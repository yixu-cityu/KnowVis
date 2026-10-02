
import os
from ..common.vlm_service import GeminiAssistant
from ..common.paths import DEFAULT_CONFIG_PATH
import ast

def get_important_concepts_llm(vid, concept_list, concept_map_str, transcript, slide_image_paths,percentage_threshold=0.1, *, config_path=DEFAULT_CONFIG_PATH):
    identify_important_concepts_prompt = {
        "task": (
            f"Please evaluate a list of candidate knowledge concepts extracted from a video lecture.\n"
            f"I will provide the data requried for this task in three parts: (1) Candidate Concepts, (2) Video transcript, and (3) the concept map extracted from the video lecture.\n"
            f"NOTE: Please just acknowledge receipt of each part with 'Data Received'. Do not perform any analysis until I provide the final evaluation criteria in the last step."
        ),
        "send_concept_list" : f"Candidate Concepts: {concept_list}\n",
        "send_video_transcript" : f"Video transcript: {transcript}\n",
        "send_concept_map": f"The concept map extracted from the video lecture: {concept_map_str}\n",
        "send_evaluation_task" : (
            f"You are provided with all slide images in the context cache, labeled from [Slide {os.path.basename(slide_image_paths[0])}] to [Slide {os.path.basename(slide_image_paths[-1])}].\n"
            "Please evaluate the importance of each concept based on the following four criteria:\n"
            "1. Multimedia Signaling: Whether the concept is visually emphasized on slides (e.g. primary heading, large arrows, or highlighting).\n"
            "2. Lexicogrammatical Relevance Markers: Whether verbal cues are used to emphasize a concept's high importance. (e.g., verb patterns (\"Must remember,\"), noun phrases (e.g., \"The most important,\" \"the key thing\"), and evaluative adjectives (e.g., \"crucial,\" \"fundamental\") ).\n"
            "3. Structural Connectivity: Whether the concept is pivotal to the network's connectivity (i.e., If this concept were removed, would the logical flow or the integration of other sub-topics be significantly compromised?).\n"
            "4. Temporal Allocation: Whether the instructor invests significant narrative effort to define, elaborate, or revisit it across multiple segments or slides. \n"
            "Evaluate and score each concept based on the four defined criteria.\n"
        ),
        "send_rank_task" : (
            "Rank the candidate concepts from highest to lowest weighted importance."
        ),
        "send_check_task" : (
            "Ensure every given candidate concept is included in the output exactly as written, with no modifications to the wording or spelling. Output only the ranked list of concepts without any additional text, explanations, or formatting."
        )
    } 
    
    vlmAssistant = GeminiAssistant(config_path=config_path)
    vlmAssistant.create_session_with_cache(response_type=None, image_paths=slide_image_paths, cache_id=f"slides_{vid}", history=None)
    resp = ''
    for k, k_prompt in identify_important_concepts_prompt.items():
        resp = vlmAssistant.send_message(k_prompt,response_type=None)
    vlmAssistant.stop()
    vlmAssistant.create_session_with_cache(response_type=None)
    ranked_important_concepts_list = vlmAssistant.send_message(f"Format the following ranked concepts as a Python-style list of strings: {resp}. Output only the list (e.g., ['Concept A', 'Concept B']) without any headers, conversational text, or explanation.",response_type=None)
    vlmAssistant.stop()
    

    ranked_important_concepts_list = ast.literal_eval(ranked_important_concepts_list)
    percentage_threshold = max(1, int(len(concept_list) * percentage_threshold))
    important_concepts_list = ranked_important_concepts_list[:percentage_threshold]
    important_concepts_list = [item.strip("'\"") for item in important_concepts_list]

    print("LLM Identified Important Concepts:\n", important_concepts_list)

    
    hallucination_concepts = [c for c in important_concepts_list if c not in concept_list]
    
    if len(hallucination_concepts) > 0: 
        print("hallucination_concepts: ", hallucination_concepts) 

    final_important_concepts = [c for c in important_concepts_list if (c not in hallucination_concepts)]


    return final_important_concepts


def get_challenging_concepts_llm(vid, concept_list, concept_map_str, transcript, slide_image_paths, percentage_threshold = 0.1, *, config_path=DEFAULT_CONFIG_PATH):
    identify_challenging_concepts_prompt = {
        "task": (
            f"Please evaluate a list of candidate knowledge concepts extracted from a video lecture.\n"
            f"I will provide the data requried for this task in three parts: (1) Candidate Concepts, (2) The transcript of the video lecture, and (3) The concept map extracted from the video lecture.\n"
            f"NOTE: Please just acknowledge receipt of each part with 'Data Received'. Do not perform any analysis until I provide the final evaluation criteria in the last step."
        ),
        "send_concept_list" : f"Candidate Concepts: {concept_list}\n",
        "send_video_transcript" : f"Video transcript: {transcript}\n",
        "send_concept_map": f"The concept map extracted from the video lecture: {concept_map_str}\n",
        "send_evaluation_task" : (
            f"You are provided with all slide images in the context cache, labeled from [Slide {os.path.basename(slide_image_paths[0])}] to [Slide {os.path.basename(slide_image_paths[-1])}].\n"
            "Now that you have all the data and the slide images in your cache. "
            "Please evaluate whether each candidate concept is challenging for student Alex (a university student with little prior knowledge of the subject, politically neutral and objective) based on the following EIGHT criteria:\n"
            "1. Counterintuitive: Whether the concept goes against student’s current conceptual model of the topic.\n"
            "2. Inert: Whether the concept can not fit easily into learners’ current understanding or may conflict with their world view or beliefs, e.g., a student may not understand why anyone would smoke when they know that smoking causes disease, until the student encounters someone who experiences little power or agency\n"
            "3. Alien: Whether the student cannot connect the concept to the world around them, so then struggles to apply it meaningfully in practice or in new contexts.\n"
            "4. Ritual: Whether the concept is primarily encoded as routine answers to common questions, making it difficult for students to grasp the underlying principles? For instance, students may be able to perform a respiratory examination exactly as taught, yet lack understanding of the underlying concepts and the rationale for each step.\n"
            "5. Troublesome language: Whether the concept includes medical jargon which may be meaningless to students and render a topic incomprehensible.\n"
            "6. Conceptually difficult:  whether the concept is complex in nature. For example, many medical concepts are extremely complex.\n"
            "7. Tacit:  whether the concept is implicit within a discipline but may not be overtly explained during the video lecture.\n"
            "8. Nettlesome: whether the concept challenge cultural or individual beliefs and result in an intense emotional response.\n"
            "Evaluate each concept based on the EIGHT defined criteria.\n"
        ),
        "send_select_task" : (
            "Select the candidate concepts satisfying at lease one of the eight 'Challenging' criteria. Discard low-level details; retain only challenging concepts that are important for learning.\n"
        ),
        "send_rank_task" : (
            "Rank the selected challenging concepts from from highest to lowest level of challenging based on the EIGHT defined criteria..\n"
        ),
        "send_check_task" : (
            "Check the output. Each output concept must be verbatim from the original list, with no changes to wording, spelling, or formatting. Output only the ranked list of concepts without any additional text, explanations, or formatting."
        )

    }
    vlmAssistant = GeminiAssistant(config_path=config_path)
    vlmAssistant.create_session_with_cache(response_type=None, image_paths=slide_image_paths, cache_id=f"slides_{vid}", history=None)
    
    resp = ''
    for k, k_prompt in identify_challenging_concepts_prompt.items():
        resp = vlmAssistant.send_message(k_prompt,response_type=None)
    
    vlmAssistant.stop()
    
    vlmAssistant.create_session_with_cache(response_type=None)
    ranked_challenging_concepts_list = vlmAssistant.send_message(f"Format the following concepts as a Python-style list of strings: {resp}. Output only the list (e.g., ['Concept A', 'Concept B']) without any headers, conversational text, or explanation.",response_type=None)
    vlmAssistant.stop()
    
    
    ranked_challenging_concepts_list = ast.literal_eval(ranked_challenging_concepts_list)

    percentage_threshold = max(1, int(len(concept_list) * percentage_threshold))
    challenging_concepts_list = ranked_challenging_concepts_list[:percentage_threshold]

    challenging_concepts_list = [item.strip("'\"") for item in challenging_concepts_list]
    print("LLM Identified Challenging Concepts:\n", challenging_concepts_list)

    
    hallucination_concepts = [c for c in challenging_concepts_list if c not in concept_list]
    
    if len(hallucination_concepts) > 0: 
        print("hallucination_concepts: ", hallucination_concepts) 

    final_challenging_concepts = [c for c in challenging_concepts_list if (c not in hallucination_concepts)]

    return final_challenging_concepts
    
    
