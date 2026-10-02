

import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.semi_supervised import LabelSpreading
from ..common.utils import get_triple_from_di_graph
import json
import os
import re
from typing import Any, cast
from ..common.paths import DATA_DIR, DEFAULT_CONFIG_PATH
from ..common.vlm_service import GeminiAssistant

def get_ku_by_direct_prompting(vid, concept_map_str, *, config_path=DEFAULT_CONFIG_PATH):
    processed_data_dir = DATA_DIR / "processed" / vid

    print(f"get_clusters_by_direct_prompting from video: {vid}")
    chunks = json.loads((processed_data_dir / f'transcribe_chunk_{vid}.json').read_text(encoding='utf-8'))
    transcript_content = '\n'.join(chunk['transcript_text'] for chunk in chunks.values())
    slide_image_paths = sorted((processed_data_dir / 'slides').glob('*.png'))
    if not transcript_content.strip() or not slide_image_paths:
        raise ValueError('Run video preprocessing before knowledge unit extraction')
    
    assistant = GeminiAssistant(config_path=config_path)
    assistant.create_session_with_cache(image_paths=slide_image_paths, cache_id=f"slides_{vid}", response_type = None)
    prompt = f"Extract knowledge units from the provided concept map, where each unit is a cluster of triples(i.e. [nodeA, edge, nodeB]) for understanding a specific anchor concept, paired with its relevant transcript segments and slide IDs. Use the provided transcript ({transcript_content}) and the slide images available in the context cache, which are labeled from [Slide {os.path.basename(slide_image_paths[0])}] to [Slide {os.path.basename(slide_image_paths[-1])}]. The concept map is {concept_map_str} Output strictly JSON in the format {{'KU1_[anchor concept]': {{'transcript': '', 'slide_images': [], 'triples': [[]]}}}}, ensuring keys are uniquely numbered. Return only the JSON object, or {{}} if empty."
    resp = assistant.send_message(prompt, response_type = None)
    assistant.stop()
    assistant = GeminiAssistant(config_path=config_path)
    assistant.create_session_with_cache(image_paths=slide_image_paths, cache_id=f"slides_{vid}", response_type = "application/json")
    
    resp = assistant.send_message(f"Format this: {resp}.\n Output strictly JSON in the format {{'KU1_[anchor concept]': {{'transcript': '', 'slide_images': [], 'triples': [[]]}}}}, ensuring keys are uniquely numbered. Return only the JSON object, or {{}} if empty.", response_type = "application/json")
    ku_dict = json.loads(resp) if isinstance(resp, str) else resp

    return ku_dict



def get_clusters_by_label_propagation(triples, important_list, challenging_list):
    
    nonDirectedGraph = nx.Graph()
    for edge in triples:  
        nonDirectedGraph.add_edge(edge[0], edge[2], relationship = edge[1])

    directedGraph = nx.DiGraph()
    for edge in triples:  
        directedGraph.add_edge(edge[0], edge[2], relationship = edge[1])
        
    
    G = nonDirectedGraph.copy()

    seed_nodes = [node for node in dict.fromkeys(important_list + challenging_list) if node in G.nodes()]
    if len(seed_nodes) == 0:
        print("Warning: No seed nodes found in the graph. Please check important_list and challenging_list.")
        return {}
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    y = np.full(num_nodes, -1, dtype=int)
    label_id_to_seed = {}
    
    for i, seed in enumerate(seed_nodes):
        idx = node_to_idx[seed]
        y[idx] = i
        label_id_to_seed[i] = seed

    X_kernel = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
    custom_kernel = lambda X, Y=None: X
    # sklearn documents callable kernels; its unannotated default is inferred as str.
    ls_model = LabelSpreading(kernel=cast(Any, custom_kernel), alpha=0.9, max_iter=1000, tol=1e-3)
    ls_model.fit(X_kernel, y)
    probs = ls_model.label_distributions_
    knowledge_units = defaultdict(list)

    for node_idx, node_p in enumerate(probs):
        node_name = idx_to_node[node_idx]
        valid_seed_indices = np.where(node_p >= np.max(node_p) * 0.8)[0] if np.max(node_p) > 1e-5 else []
        if node_name in seed_nodes:
            valid_seed_indices = [idx for idx in valid_seed_indices if label_id_to_seed[idx] != node_name]
            for seed_idx in valid_seed_indices:
                seed_name = label_id_to_seed[seed_idx]
                knowledge_units[seed_name].append((node_name, float(node_p[seed_idx])))
        else:
            
            for seed_idx in valid_seed_indices:
                seed_name = label_id_to_seed[seed_idx]
                knowledge_units[seed_name].append((node_name, float(node_p[seed_idx])))

    final_clusters = {}
    idx = 0
    for anchor in knowledge_units.keys():
        nodes = set([n for (n, p) in knowledge_units[anchor]])
        nodes.add(anchor)
        subgraph = nonDirectedGraph.subgraph(nodes)
        is_connected = nx.is_connected(subgraph)
        if is_connected:
            final_clusters[f"{anchor}"] =directedGraph.subgraph(nodes)
            # utils.get_triple_from_di_graph(self.directedGraph.subgraph(nodes))
            idx += 1
        else:
            components = list(nx.connected_components(subgraph))
            valid_nodes = set()
            for comp in components:
                if anchor in comp:
                    valid_nodes = comp
                    if len(valid_nodes)>2:
                        final_clusters[f"{anchor}"] = directedGraph.subgraph(valid_nodes)
                        # utils.get_triple_from_di_graph(self.directedGraph.subgraph(valid_nodes))
                        idx += 1
                    break
    sorted_clusters = sorted(final_clusters.items(), 
                            key=lambda x: (x[1].number_of_edges(), x[1].number_of_nodes()), 
                            reverse=True)
    unique_clusters = {}
    processed_subgraphs = []
    idx = 0
    for anchor, G_current in sorted_clusters:
        G_triples = get_triple_from_di_graph(G_current)
        G_triples_set = set(tuple(x) for x in G_triples)
        is_subset_graph = False
        
        for G_existing in processed_subgraphs:
            if G_triples_set.issubset(set(tuple(x) for x in G_existing)):
                is_subset_graph = True
                break
        
        if not is_subset_graph:
            safe_anchor = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', anchor).rstrip(' .')
            name = f"KU{idx}_{safe_anchor}"
            unique_clusters[name] = G_triples
            processed_subgraphs.append(G_triples)
            idx += 1

    return unique_clusters
