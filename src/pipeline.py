"""Run the existing research stages in order, saving each expensive result."""

import json
from pathlib import Path
from .common.paths import DEFAULT_CONFIG_PATH, resolve_path

from .common.utils import (
    convert_graph_to_pydot_string,
    get_network_di_graph,
    get_nodes_from_triples,
)
from .modules.s1_concept_map_extractor import VLMConceptMapGenerator
from .modules.s2_concept_scorer import get_challenging_concepts_llm, get_important_concepts_llm
from .modules.s3_concept_cluster import get_clusters_by_label_propagation
from .modules.s4_concept_visualizer import KnowledgeVisualizationPipeline


def _load_or_create(path, create):
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    result = create()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return result


def run_knowVis_pipeline(video_id, chunks, output_dir, config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = resolve_path(config_path)
    output_dir = resolve_path(output_dir)
    slide_paths = sorted((output_dir / 'slides').glob('*.png'))
    if not chunks or not slide_paths:
        raise ValueError('The full pipeline requires transcript chunks and extracted slides')
    transcript = '\n'.join(chunk['transcript_text'] for chunk in chunks.values())
    if not transcript.strip():
        raise ValueError('No transcript text was found in this video')

    def extract_concepts():
        generator = VLMConceptMapGenerator(video_id, output_dir / 'concept_map_checkpoint.pkl', processed_data_dir=output_dir, config_path=config_path)
        generator.generate_concept_map(slide_paths)
        return generator.post_process()

    print('[1/4] Extracting concept map', flush=True)
    triples = _load_or_create(output_dir / 'concept_map.json', extract_concepts)
    if not triples:
        raise ValueError('Concept extraction returned an empty map; inspect concept_map_checkpoint.pkl')
    concept_list = get_nodes_from_triples(triples)
    concept_map = convert_graph_to_pydot_string(get_network_di_graph(triples))

    print('[2/4] Scoring concepts', flush=True)
    def score_concepts():
        args = (video_id, concept_list, concept_map, transcript, slide_paths)
        return {
            'important': get_important_concepts_llm(*args, config_path=config_path),
            'challenging': get_challenging_concepts_llm(*args, config_path=config_path),
        }
    scores = _load_or_create(output_dir / 'concept_scores.json', score_concepts)

    print('[3/4] Clustering knowledge units', flush=True)
    units = _load_or_create(
        output_dir / 'knowledge_units.json',
        lambda: get_clusters_by_label_propagation(triples, scores['important'], scores['challenging']),
    )
    if not units:
        raise ValueError('No knowledge units were formed; inspect the concept map and scores')

    print('[4/4] Generating visual summaries', flush=True)
    visualizer = KnowledgeVisualizationPipeline(video_id, config_path=config_path)
    visuals = []
    for name, unit_triples in units.items():
        # Cluster names are filename-safe; also check cached data before writing.
        if Path(name).name != name or any(c in name for c in '<>:"/\\|?*'):
            raise ValueError(f'Invalid knowledge unit filename: {name}')
        anchor = name.split('_', 1)[-1]
        content = _load_or_create(
            output_dir / 'knowledge_units' / f'{name}.json',
            lambda: visualizer.retrive_content(unit_triples, transcript, slide_paths),
        )
        if not content.get('transcript'):
            raise ValueError(f'No source content retrieved for {name}')
        storyboard = _load_or_create(
            output_dir / 'storyboards' / f'{name}.json',
            lambda: visualizer.generate_storyboard(anchor, content, slide_paths),
        )
        visual_path = output_dir / 'visuals' / f'{name}.png'
        visualizer.generate_visual(anchor, content, storyboard['storyboard'], slide_paths, visual_path)
        if not visual_path.is_file():
            raise RuntimeError(f'Visual generation did not produce {visual_path}')
        visuals.append(str(visual_path.resolve()))
    return {'video_id': video_id, 'visuals': visuals}
