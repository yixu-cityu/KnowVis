import mimetypes
import os
import time
from pathlib import Path

import PIL.Image
from google import genai
from google.genai import errors, types
from .paths import DEFAULT_CONFIG_PATH, read_gemini_api_key


def save_binary_file(image_save_path, data):
    path = Path(image_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        
    print(f"Image saved to: {image_save_path}")


class GeminiAssistant:
    global_input_tokens = 0
    global_output_tokens = 0
    global_thought_tokens = 0

    def __init__(self, model_name=None, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.client = genai.Client(
            api_key=read_gemini_api_key(config_path),
            vertexai=False,
        )
        self.system_instruction = "You are a helpful assistant."
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.cache = None
        self.chat = None
        self.llm_config = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.thought_tokens = 0
    
    def upload_and_wait(self, file_paths):
        uploaded_files = []
        for path in file_paths:
            print(f"[File] Uploading (size: {os.path.getsize(path)}): {os.path.basename(path)}...")
            file = self.client.files.upload(file=path)
            state = file.state.name if file.state is not None else None
            print(f"[File] Initial State: {file.name} - {state}")
            deadline = time.monotonic() + 300
            while state == "PROCESSING":
                file_name = file.name
                if not file_name:
                    raise RuntimeError(f"Upload returned no file name: {path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"File processing timed out: {file_name}")
                time.sleep(2)
                try:
                    file = self.client.files.get(name=file_name)
                except Exception as e:
                    raise RuntimeError(f"Unable to check uploaded file: {file_name}") from e
                state = file.state.name if file.state is not None else None

            if state != "ACTIVE":
                raise ValueError(f"File processing failed: {file.name} (state={state})")
        
            uploaded_files.append(file)
        return uploaded_files
    
    
    def reset_chat_with_history(self, response_type, history=None):
        self.stop()
        config = self.llm_config or types.GenerateContentConfig(temperature=0.0, seed=0)
        self.llm_config = config.model_copy(update={"response_mime_type": response_type})
        self.chat = self.client.chats.create(
            model=self.model_name,
            history=history, 
            config=self.llm_config
        )


    def find_existing_cache_id(self, cache_id):
        for c in self.client.caches.list():
            if c.display_name == cache_id and (c.model or '').removeprefix('models/') == self.model_name.removeprefix('models/'):
                return c
        return None
    
    def create_session_with_cache(self, response_type, temperature=0.0, image_paths=None, cache_id=None, history=None, max_token=None):
        self.llm_config = None
        self.cache = None
        if image_paths and cache_id:
            existing_cache = self.find_existing_cache_id(cache_id)
            if existing_cache:
                print(f"Found existing cache for {cache_id}, skipping upload.")
                self.cache = existing_cache
            else:
                files = self.upload_and_wait(image_paths)
                cache_contents = []
                for i, file_obj in enumerate(files):
                    cache_contents.append(f"Slide {os.path.basename(image_paths[i])}:")
                    cache_contents.append(file_obj)

                try:
                    self.cache = self.client.caches.create(
                        model=self.model_name,
                        config=types.CreateCachedContentConfig(
                            display_name=cache_id,
                            system_instruction="You have access to a series of slide images from a lecture. Each image is preceded by a label like '[Slide X]'",
                            contents=cache_contents
                        )
                    )
                except errors.ClientError as error:
                    # Short lectures can be below the explicit-cache token minimum.
                    if error.code != 400 or not any(word in str(error).lower() for word in ('too small', 'minimum', 'min_total_token')):
                        raise
                    parts = []
                    for label, file_obj in zip(cache_contents[::2], files):
                        parts.extend([types.Part.from_text(text=label), types.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type)])
                    history = [
                        types.Content(role='user', parts=parts),
                        types.Content(role='model', parts=[types.Part.from_text(text='Data Received.')]),
                        *(history or []),
                    ]

            
            self.llm_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=0,
                max_output_tokens=max_token,
                response_mime_type=response_type,
                cached_content=self.cache.name if self.cache else None
            )
        if self.llm_config is None:
            self.llm_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=0,
                max_output_tokens=max_token,
                response_mime_type=response_type
            ) 
        self.chat = self.client.chats.create(
            model=self.model_name,
            history=history, 
            config=self.llm_config
        )

    def generate_image(self, prompt: str = "", image_save_path: str | Path = "", input_file_paths = None, thinking_level="HIGH", cache_id=None) -> Path:
        if cache_id:
            raise ValueError("Image generation does not support explicit caching; pass input_file_paths instead")
        parts = [types.Part.from_text(text=prompt)]

        VIDEO_MIME_PREFIXES = ("video/",)

        for path in input_file_paths or []:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            
            if 'slides' in str(path):
                file_label = f"\nSlide {os.path.basename(path)}\n"
                parts.append(types.Part.from_text(text=file_label))

            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                mime_type = "image/jpeg"

            if mime_type.startswith(VIDEO_MIME_PREFIXES):
                # Video: upload via Files API and reference by URI
                [uploaded] = self.upload_and_wait([path])
                parts.append(
                    types.Part.from_uri(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type,
                    )
                )
            else:
                # Image (or other binary): send inline
                with open(path, "rb") as f:
                    image_bytes = f.read()
                parts.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    )
                )
                    
        contents: types.ContentListUnionDict = [
            types.Content(
                role="user",
                parts=parts,
            ),
        ]
        image_config = types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(thinking_level),
            ),
            image_config = types.ImageConfig(
                image_size="1K",
            ),
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
        )

        saved_image = False
        for chunk in self.client.models.generate_content_stream(
            model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"),
            contents=contents,
            config=image_config,
        ):
            usage = chunk.usage_metadata
            if usage:
                if usage.prompt_token_count:
                    GeminiAssistant.global_input_tokens += usage.prompt_token_count
                    self.input_tokens += usage.prompt_token_count
                if usage.candidates_token_count:
                    GeminiAssistant.global_output_tokens += usage.candidates_token_count
                    self.output_tokens += usage.candidates_token_count
                if usage.thoughts_token_count:
                    GeminiAssistant.global_thought_tokens += usage.thoughts_token_count
                    self.thought_tokens += usage.thoughts_token_count

            for part in chunk.parts or []:
                if not part.thought and part.inline_data and part.inline_data.data and (part.inline_data.mime_type or '').startswith('image/'):
                    save_binary_file(image_save_path, part.inline_data.data)
                    saved_image = True
        if not saved_image:
            raise RuntimeError("Gemini returned no image; check the model response and safety settings")
        return Path(image_save_path)

    def send_message(self, prompt, response_type,temperature = 0.0, input_image_path = None, max_token = None) -> str:
        if not self.chat:
            self.create_session_with_cache(response_type, temperature=temperature, max_token=max_token)
        if self.llm_config is None or self.chat is None:
            raise RuntimeError('Gemini chat initialization did not create a configured session')

        if self.llm_config.response_mime_type != response_type:
            current_history = self.chat.get_history()
            self.reset_chat_with_history(response_type=response_type, history = current_history)

        chat = self.chat
        if chat is None:
            raise RuntimeError('Gemini chat reset did not create a session')
        if input_image_path is None:
            response = chat.send_message(prompt)
        else:
            with PIL.Image.open(input_image_path) as img:
                response = chat.send_message([img, prompt])
        usage = response.usage_metadata
        if usage:
            GeminiAssistant.global_input_tokens += usage.prompt_token_count if usage.prompt_token_count else 0
            GeminiAssistant.global_output_tokens += usage.candidates_token_count if usage.candidates_token_count else 0
            GeminiAssistant.global_thought_tokens += usage.thoughts_token_count if usage.thoughts_token_count else 0
            self.thought_tokens += usage.thoughts_token_count if usage.thoughts_token_count else 0
            self.input_tokens += usage.prompt_token_count if usage.prompt_token_count else 0
            self.output_tokens += usage.candidates_token_count if usage.candidates_token_count else 0
        text = response.text
        if not text or not text.strip():
            raise RuntimeError('Gemini returned an empty text response')
        return text
    

    
    def stop(self):
        self.chat = None
        
        print(f"Tokens: Session Input ({self.input_tokens}), Output ({self.output_tokens+self.thought_tokens}) | Total Input ({GeminiAssistant.global_input_tokens}), Total Output ({GeminiAssistant.global_output_tokens+GeminiAssistant.global_thought_tokens})")
        self.input_tokens = 0
        self.output_tokens = 0
        self.thought_tokens = 0
