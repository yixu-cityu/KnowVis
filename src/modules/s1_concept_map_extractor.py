
import json
import os
import pickle
import time
from pathlib import Path

from google.genai import types
from ..common.vlm_service import GeminiAssistant
from ..common.paths import DATA_DIR, DEFAULT_CONFIG_PATH, resolve_path

from ..common.utils import remove_dup_concepts, filter_concepts, get_nodes_from_triples, filter_generic_concepts, merge_relations,remove_isolated_triples, get_plurals

class VLMDirectConceptMapGenerator:
    def __init__(self, vid = "", config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.vid = vid
        self.config_path = resolve_path(config_path)
        
    def generate_concept_map(self):
        processed_data_dir = DATA_DIR / "processed" / self.vid
    
        print(f"[VLMDirectConceptMapGenerator] Generating Concept Map from video: {self.vid}")
        chunk_path = processed_data_dir / f'transcribe_chunk_{self.vid}.json'
        chunks = json.loads(chunk_path.read_text(encoding='utf-8'))
        transcript_content = '\n'.join(chunk['transcript_text'] for chunk in chunks.values())
        slide_image_paths = sorted((processed_data_dir / 'slides').glob('*.png'))
        if not transcript_content.strip() or not slide_image_paths:
            raise ValueError('Run video preprocessing before concept extraction')

        assistant = GeminiAssistant(config_path=self.config_path)
        assistant.create_session_with_cache(response_type = "application/json", image_paths=slide_image_paths, cache_id=f"slides_{self.vid}")
        prompt_acm2 = f"Extract a detailed concept map from the provided transcript and slides. Output the map as a list of triples in the format: [head, relation, tail]. Ensure all key concepts and their interconnections are captured.\n. Transcript:{transcript_content}. You are provided with all slide images in the context cache, labeled from [Slide {os.path.basename(slide_image_paths[0])}] to [Slide {os.path.basename(slide_image_paths[-1])}].\n Output result concept map in json format where the key is index and the value is the triple, e.g{{'1': [a,b,c], '2':[d,e,f]}}. Only output json. If empty, output: {{}}"
        prompt_acm1 = f"Extract a concept map from the provided transcript and slides. Output the graph as a list of triples in the format: [head, relation, tail].\n. Transcript:{transcript_content}. You are provided with all slide images in the context cache, labeled from [Slide {os.path.basename(slide_image_paths[0])}] to [Slide {os.path.basename(slide_image_paths[-1])}].\n Output result concept map in json format where the key is index and the value is the triple, e.g{{'1': [a,b,c], '2':[d,e,f]}}. Only output json. If empty, output: {{}}"
        resp = assistant.send_message(prompt_acm1,response_type="application/json")
        assistant.stop()
        triples_dict = json.loads(resp) if isinstance(resp, str) else resp
        if isinstance(triples_dict, list):
            triples_list = triples_dict
        elif isinstance(triples_dict, dict):
            triples_list = list(triples_dict.values())
        else:
            raise ValueError('Concept extraction must return a JSON list or object')
        return triples_list
    

class VLMConceptMapGenerator:
    def __init__(self, vid: str = "", checkpoint_file_path: str | Path = "", processed_data_dir: str | Path | None = None, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.vid = vid
        self.video_fold_dir = resolve_path(processed_data_dir) if processed_data_dir else DATA_DIR / "processed" / vid
        self.checkpoint_file = resolve_path(checkpoint_file_path, self.video_fold_dir) if checkpoint_file_path else self.video_fold_dir / 'concept_map_checkpoint.pkl'
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.vlmAssistant = GeminiAssistant(config_path=config_path)
        
        chunk_path = self.video_fold_dir / f"transcribe_chunk_{vid}.json"
        with open(chunk_path, "r", encoding="utf-8") as file:
            self.chunk_dict = json.load(file) 
        if not self.chunk_dict:
            raise ValueError(f"No transcript chunks found in {chunk_path}")
        self.dup_record = ""

    def update_checkpoint(self):
        checkpoint_data = self.load_checkpoint() or {}
        checkpoint_data["dup_record"] = self.dup_record
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def save_checkpoint(self, current_chunk_id, history):
        checkpoint_data = {
            "last_chunk_id": current_chunk_id,
            "history": [m.model_dump() if hasattr(m, 'model_dump') else m for m in history],
            "dup_record": self.dup_record
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def load_checkpoint(self):
        print(f"Try load from checkpoint:",self.checkpoint_file)
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "rb") as f:
                    data = pickle.load(f)
                    if "history" in data and data["history"]:
                        data["history"] = [
                            types.Content(**m) if isinstance(m, dict) else m 
                            for m in data["history"]
                        ]
                print(f"Resuming from checkpoint: last processed chunk {data['last_chunk_id']}")
                return data
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        return None
    

    def process_chunk(self, chunk_text, chunk_slide_list):
        slides_ref = ""
        for slide in chunk_slide_list:
            slides_ref += f"Slide {slide}, "
        slides_ref = slides_ref[:-2]

        concept_extraction_prompt = (
            f"Next, I will give you a chunk. Focus on the given chunk and the corresponding slides {slides_ref}, "
            f"the task is to extract knowledge concepts from the provided sentences in the chunk. "
            f"The knowledge concepts should be an informative and meaningful entities with clear semantic significance, "
            f"no more than three words. Output result concepts in list format: CL = [A, B, C]"
            f"Here is the chunk:\n{chunk_text}"
        )
        time.sleep(2)
        concept_list_str = self.vlmAssistant.send_message(prompt=concept_extraction_prompt, response_type=None)
        print(f"Concept List: {concept_list_str}")

        
        triple_extraction_prompt = (
            f"Focus on concepts in CL, explore relationships between concepts based on the sentences in the given chunk and the corresponding slides: {slides_ref}."
            f"Generate knowledge triples among these concepts in the format (head, relation, tail), where the 'head' and 'tail' are concepts in CL, and 'relation' represents the extracted relationship between them. Output result triples in json format where the key is index and the value is the triple. Only output json. If empty, output: {{}}"
        )    
        result_tripples = self.vlmAssistant.send_message(triple_extraction_prompt, response_type=None)
        print("triple_extraction:", result_tripples)
        time.sleep(2)


        check_missing_triple_prompt = "Focus on all the head and tails in this json and previous json, do you miss any relationships among them that are indicated or mentioned in and across all the given chunks and the corresponding slides, such as inclusion and other logical relationships? Please add the missing triples to this json. Please only output the refined JSON for this chunk."

        refined_triples = self.vlmAssistant.send_message(check_missing_triple_prompt, response_type=None)
        print("check_missing_triple:",refined_triples)
        time.sleep(2)

        
        check_proposition_triple_prompt = "Check the triples in the JSON based on the following two principles for constructing triples: Each triple should represent a complete proposition with a clear and direct relational description. The head and tail of the triple should be distinct knowledge concepts in CL, each concept should be an informative and meaningful entities with clear semantic significance, no more than three words. If any triples do not meet these principles, refine them accordingly to ensure compliance based on the given chunk. Only output json. No explanations, no quotes, no newlines, no extra text, no markdown formatting, backticks, or the word 'json' If empty, output: {}"

        refined_triples = self.vlmAssistant.send_message(check_proposition_triple_prompt, response_type='application/json')
        print("check_proposition_triple:",refined_triples)


        check_dup_within_chunk_prompt = "Check this JSON for triples where the heads and tails may have different descriptions but refer to the same real-world entity. Identify duplicated entities that are functionally identical due to minor formatting, punctuation, or spelling variations (e.g., \"5 number summary\" and \"Five-number summary\").Only select entities that unambiguously represent the same object or label without conceptual nuance. Do not group technical synonyms (e.g., \"Q1\" and \"25th percentile\") where distinct terminology serves an educational purpose. Output a single list of lists of duplicate entities, or [] if none exist. No explanations, no quotes, no newlines, and no extra text."
        dup_concepts_within_chunk = self.vlmAssistant.send_message(check_dup_within_chunk_prompt, response_type=None)
        print("check_dup_within_chunk:", dup_concepts_within_chunk)

        check_dup_across_chunk_prompt = "Refer JSON triples across all chunks. Identify duplicated entities that are functionally identical due to minor formatting, punctuation, or spelling variations (e.g., \"5 number summary\" and \"Five-number summary\").Only select entities that unambiguously represent the same object or label without conceptual nuance. Do not group technical synonyms (e.g., \"Q1\" and \"25th percentile\") where distinct terminology serves an educational purpose. Output a single list of lists of duplicate entities, or [] if none exist. No explanations, no quotes, no newlines, and no extra text. "
        dup_concepts_across_chunk = self.vlmAssistant.send_message(check_dup_across_chunk_prompt, response_type=None)
        print("check_dup_across_chunk:", dup_concepts_across_chunk)

        self.dup_record += dup_concepts_within_chunk + dup_concepts_across_chunk
        
        chat = self.vlmAssistant.chat
        if chat is None:
            raise RuntimeError('Concept extraction chat is not initialized')
        current_history = chat.get_history()
        new_history = current_history[0:-9] + [current_history[-5]]
        self.vlmAssistant.reset_chat_with_history(response_type=None, history = new_history)
        
        return new_history

    def check_dup_concepts(self):
        if not self.dup_record:
            return []
        check_dup_concepts_prompt = (
            f"Generate a JSON object from the following lists of duplicate entities: {self.dup_record}. Use the most representative term as the key and the list of associated duplicates as the value. Output the result in valid JSON format.  Only output json. No explanations, no quotes, no newlines, no extra text, no markdown formatting, backticks, or the word 'json'. If empty, output: {{}}."
        )
        self.vlmAssistant.stop()
        self.vlmAssistant.create_session_with_cache(response_type = "application/json")
        dup_concepts_record = self.vlmAssistant.send_message(check_dup_concepts_prompt,response_type = "application/json")
        self.dup_record = dup_concepts_record

        
    def generate_concept_map(self, slides_image_paths):
        
        checkpoint = self.load_checkpoint()
        start_skipping = False
        last_id = None
        if checkpoint:
            last_id = checkpoint["last_chunk_id"]
            self.dup_record = checkpoint["dup_record"]
            initial_history = checkpoint["history"]
            start_skipping = True
        else:
            initial_history = None

        is_processed = False
        total_chunks = len(self.chunk_dict)
        for chunk_id, chunk in self.chunk_dict.items():
            if start_skipping:
                if str(chunk_id) == str(last_id):
                    start_skipping = False 
                continue
            if not self.vlmAssistant.chat:
                self.vlmAssistant.create_session_with_cache(response_type=None, image_paths=slides_image_paths, cache_id=f"slides_{self.vid}", history=initial_history)
            chunk_text = chunk["transcript_text"]
            chunk_slide_list = chunk["slides"]
            print(f"Processing chunk {chunk_id}/{total_chunks} with slides {chunk_slide_list}...")
            history = self.process_chunk(chunk_text, chunk_slide_list)
            self.save_checkpoint(str(chunk_id), history)
            is_processed = True

        # if is_processed:
        self.check_dup_concepts()
        print(f"After checking duplicated concepts: {self.dup_record}")
        self.update_checkpoint()
        self.vlmAssistant.stop()

    def parse_vlm_history(self, history_list = None):
        triples = []
        for item in history_list or []:
            role = item.get('role') if isinstance(item, dict) else item.role
            if role == 'model':
                parts = item.get('parts', []) if isinstance(item, dict) else item.parts or []
                for part in parts:
                    model_response = part.get('text') if isinstance(part, dict) else part.text
                    if not model_response or model_response.lstrip().startswith('CL'):
                        continue
                    clean_json = model_response.strip().strip('`').removeprefix('json').strip()
                    try:
                        triples_json = json.loads(clean_json)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(triples_json, dict):
                        triples.extend(triples_json.values())
                    elif isinstance(triples_json, list):
                        triples.extend(triples_json)
        return triples


    def post_process(self):
        
        ori_triples = []
        dup_record_dict = {}

        with open(self.checkpoint_file, "rb") as f:
            data = pickle.load(f)
            vlm_history = data.get("history", [])
            dup_record = data.get("dup_record") or {}
            while isinstance(dup_record, str):
                try:
                    dup_record = json.loads(dup_record)
                except json.JSONDecodeError:
                    break
            dup_record_dict = dup_record
            ori_triples = self.parse_vlm_history(vlm_history)

        if not ori_triples:
            print("Error parsing triples from VLM history.")
            return []
        triples = remove_dup_concepts(ori_triples, dup_record_dict)
        triples = filter_concepts(triples)
        concept_list = get_nodes_from_triples(triples)
        # deplural
        plural_dict = get_plurals(concept_list)
        for index in range(len(triples)):
            h, r, t = triples[index]
            if h in plural_dict.keys(): h = plural_dict[h]
            if t in plural_dict.keys(): t = plural_dict[t]
            triples[index] = [h.lower(), r.lower(), t.lower()]

        # multi relation between two concepts
        triples = filter_generic_concepts(triples)
        triples = merge_relations(triples)
        triples = remove_isolated_triples(triples)

        
        with open(self.checkpoint_file, "rb") as f:
            data = pickle.load(f)
            data["knowledge_triples"] = triples
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(data, f)

        return triples








