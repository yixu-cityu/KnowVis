
import json
import os
from pathlib import Path
from ..common.vlm_service import GeminiAssistant
from ..common.paths import DEFAULT_CONFIG_PATH, resolve_path

def generate_visual_by_direct_prompting(anchor_concept, retrived_content, slide_image_paths, save_path, *, config_path=DEFAULT_CONFIG_PATH):
    save_path = Path(save_path)
    if save_path.exists():
        print(f"File {save_path} already exists.")
        return 
    
    save_path.parent.mkdir(parents=True, exist_ok=True) 
    prompt = f"Create a visual illustration for {anchor_concept} based on the given content: {retrived_content} "
    vlmAssistant = GeminiAssistant(config_path=config_path)
    vlmAssistant.generate_image(prompt=prompt, image_save_path=save_path, input_file_paths=slide_image_paths, thinking_level="HIGH")
    return


class KnowledgeVisualizationPipeline:
    def __init__(self, vid, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.vid = vid
        self.config_path = resolve_path(config_path)

    def get_retrieving_video_content_prompt(self, knowledge_triples, transcript, slides_image_paths):
        
        retrieving_video_content_prompt = {
            "task": (
                "You are a helpful assistant."
                "Your objective is to retrieve specific content from a video lecture that strongly aligns with a provided set of knowledge triples.\n"
                "I will provide the data requried for this task in three parts: (1) Knowledge Triples, (2) Video Transcript\n"
                "INSTRUCTION: Please acknowledge receipt of each data part by responding strictly with the phrase 'Data Received'. "
                "Do not perform any retrieval or extraction until explicitly instructed to do so in the final step."
            ),
            "send_knowledge_triples": f"Part 1 - Knowledge Triples:\n{knowledge_triples}\n",
            "send_video_transcript": f"Part 2 - Video Transcript:\n{transcript}\n",
            "send_retrieve_task": (
                f"Context: You have access to all slide images in the context cache, labeled sequentially from "
                f"[Slide {os.path.basename(slides_image_paths[0])}] to [Slide {os.path.basename(slides_image_paths[-1])}].\n\n"
                "--- YOUR TASK ---\n"
                "Now that all data and slide images are available, please execute the following:\n"
                "1. Extract segments from the transcript that are strongly relevant to the provided knowledge triples.\n"
                "2. Identify the slide image labels that are strongly relevant to the provided knowledge triples.\n\n"
                "--- CONSTRAINTS ---\n"
                "- Only output content that demonstrates a direct, strong relationship to the knowledge triples.\n"
                "- Return the output in strict JSON format using this schema: "
                "{\"transcript\": \"<retrieved transcript text>\", \"slide_images\": [\"<slide image labels>\"]}\n"
                "- Strict Output Rule: Do NOT include Markdown code blocks (e.g., no ```json or ```). Do NOT include any introductory, explanatory, or concluding text.\n"
                r"- Empty Results: If no content is found related to the knowledge triples, return an empty JSON object: \{\}."
            )
        }
        return retrieving_video_content_prompt
        
    def retrive_content(self, knowledge_triples, transcript, slides_image_paths):
            
        vlmAssistant = GeminiAssistant(config_path=self.config_path)
        ret = {}
        prompt = self.get_retrieving_video_content_prompt(knowledge_triples, transcript, slides_image_paths)
        vlmAssistant.create_session_with_cache(response_type=None, image_paths=slides_image_paths, cache_id=f"slides_{self.vid}", history=None)
        res1 = vlmAssistant.send_message(prompt["task"],response_type=None)
        res2 = vlmAssistant.send_message(prompt["send_knowledge_triples"],response_type=None)
        res3 = vlmAssistant.send_message(prompt["send_video_transcript"],response_type=None)
        res4 = vlmAssistant.send_message(prompt["send_retrieve_task"],response_type="application/json")
        vlmAssistant.stop()
        ret = json.loads(res4)  
        ret['triples'] = knowledge_triples
        return ret
    

    def check_visual(self, anchor_concept, input_image_path, output_image_path):
        input_image_path = Path(input_image_path)
        output_image_path = Path(output_image_path)
        identify_issue_prompt = '''
            Analyze the provided illustration for the following errors:
            1. Repetitions: Duplicated text, redundant panels, or ghosting elements.
            2. Logical inconsistency across elements (count, equations, cause-and-effect).

            If errors found: Output ONLY a revised prompt using Location-Based Structuring method that fixes these issues by removing duplications and correcting logical inconsistencies. 
            If no errors: Output ONLY the word 'PASS'.
        '''                   
     
        vlmAssistant = GeminiAssistant(config_path=self.config_path)
        vlmAssistant.create_session_with_cache(response_type=None)
        resp = vlmAssistant.send_message(identify_issue_prompt,response_type=None,temperature=1.0, input_image_path=input_image_path)
        vlmAssistant.stop()

        if resp and resp.strip().upper() == 'PASS':
            print("Visual check passed. No issues detected.")
            input_image_path.replace(output_image_path)
            return
        if not resp:
            raise RuntimeError('Visual review returned an empty response')
        
        style_prompt = "Generate a visual illustration with a visually appealing design.  Avoid serious or boring vibes; keep it vivid, approachable, and clear.\n"

        vlmAssistant = GeminiAssistant(config_path=self.config_path)
        vlmAssistant.create_session_with_cache(response_type=None)
        vlmAssistant.generate_image(prompt=style_prompt+resp, image_save_path=Path(output_image_path), input_file_paths=[input_image_path], thinking_level="MINIMAL")
        vlmAssistant.stop()




    def generate_visual(self, anchor_concept,  knowledge_units_info={}, storyboard_json = {}, slide_image_paths = None, save_visual_path: str | Path = "", force_regen = False):
        save_visual_path = Path(save_visual_path)
        
        if save_visual_path.exists() and not force_regen:
            print(f"File {save_visual_path} already exists.")
            return save_visual_path 

        print(f"Generating {save_visual_path} .")
        save_visual_path.parent.mkdir(parents=True, exist_ok=True) 

        
        
        generate_visual_prompt = (
            f"Generate a visual educational illustration for the concept: '{anchor_concept}' based on the given visual storyboard outline:\n"
            f"{storyboard_json}\n"
            f"Follow the outline strictly. DO NOT add any extra text or numbers outside the storyboard outline.\n"
            f"Create a visually appealing design. Avoid serious or boring vibes; keep it vivid, approachable, and clear.\n"
            f"Enrich the illustration with informative visual metaphors, expressive icons without adding extra text.\n"
            f"Render text in legible format using friendly font."
            f"Don't restricted to a rigid grid layout. Arrange the panels layout flexibly to optimize narrative flow and visual balance.\n"
        )


        vlmAssistant = GeminiAssistant(config_path=self.config_path)
        middle_path = Path(save_visual_path.with_name(f"pending_{save_visual_path.name}"))

        vlmAssistant.generate_image(prompt=generate_visual_prompt, image_save_path=middle_path)

        self.check_visual(anchor_concept,middle_path, save_visual_path)
        return save_visual_path
    

    def generate_storyboard(self, anchor_concept, knowledge_units_info = {}, slide_image_paths = None):
        generate_storyboard_prompt = {
            "system_setup": (
                f"Your task is to generate a visual storyboard explaining '{anchor_concept}' for a novice learner."
                f"I will provide the source materials first. Acknowledge receipt with 'Data Received.', and do not generate the storyboard until instructed."
            ),
            "send_source_materials" :  (
                f"source materials:\n"
                f"- Transcript: {knowledge_units_info['transcript']}\n"
                f"- Slide Images (in context cache): {knowledge_units_info['slide_images']}\n"
                f"- Knowledge Triples: {knowledge_units_info['triples']}\n"
            ),
            "generate_storyboard": (
                f"Generate a concise storyboard for a visual illustration explaining '{anchor_concept}' for a novice learner.\n"
                f"Focus on the illustrating the knowledge triples. Take the transcript and slide images as fatucal basis.\n"
                f"Please follow the steps below:\n"
                "1. GENRE & STRATEGY: Choose one most appropriate genre (Magazine Style, Annotated Chart, Partitioned Poster, Flow Chart, or Comic Strip) to best illustrate the core concepts.\n"
                "2. OUTLINE: Generate a concise storyboard outline by synthesizing the provided knowledge triples into their most essential visual narrative arc. Be faithful to the source material. Use minimum number of panels to capture the core logic without unnecessary details. \n"
                "3. PANEL BREAKDOWN: For each panel, provide:\n"
                "   - Panel ID: [e.g., Panel-1]\n"
                "   - Panel Triple To Be Focused: [Note which triples this panel focuses on]\n"
                "   - Panel Text To Be Rendered: [Provide the EXACT, concise text that should appear in the panel]\n"
                "   - Panel Visual Description: [Detailed description of the visual elements to be illustrated]"
            ),
            "check_generated_storyboard": (
                f"Review the generated storyboard. Verify each panel against the following criteria:\n"
                f"1. The 'Panel Text To Be Rendered' must NOT be mentioned or described again inside the 'Panel Visual Description'.\n"
                f"2. Text should be a concise.\n"
                f"3. Ensure all content is faithful to the source transcript and slides.\n"
                f"4. Describe the scene physically rather than conceptually. Explicitly state the placement of objects (e.g., in the foreground, on the left) and specific actions (e.g., a hand holding a pen) rather than abstract ideas (e.g., showing the process).\n"
                f"If any panel fails criteria, rewrite it. Output the final, corrected storyboard."
            )
        }
        vlmAssistant = GeminiAssistant(config_path=self.config_path)
        
        vlmAssistant.create_session_with_cache(response_type=None, temperature=0.5, image_paths=slide_image_paths, cache_id=f"slides_{self.vid}", history=None)
        
        resp = ''
        for k, k_prompt in generate_storyboard_prompt.items():
            resp = vlmAssistant.send_message(k_prompt,response_type=None)
        
        vlmAssistant.stop()
        example_json = '''
        {
            "genre": "string (e.g., Comic Strip)",
            "panel_count": "integer",
            "storyboard_outline": "string",
            "panels": [
                {
                    "panel_id": "integer",
                    "panel_text_to_render": "string",
                    "panel_visual": "string"
                }
            ]
        }
        '''
        vlmAssistant.create_session_with_cache(response_type="application/json")
        storyboard_json = vlmAssistant.send_message(f"Format the following content as json: {resp} in this structure: {example_json}. Do not include any headers, conversational text, or explanation. Only output json",response_type="application/json")
        vlmAssistant.stop()

        return {"storyboard": storyboard_json, "triples": knowledge_units_info.get('triples', [])}



            
