"""Deterministic, conservative cleanup of concept triples.

The original utility file was empty. These rules intentionally avoid semantic
synonym guessing: only explicit aliases and observed singular forms are merged.
"""

import json

import networkx as nx


def filter_concepts(triples):
    """Drop malformed, empty, duplicate and self-referencing triples."""
    result = []
    seen = set()
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        if not all(isinstance(value, str) and value.strip() for value in triple):
            continue
        head, relation, tail = (value.strip() for value in triple)
        key = (head, relation, tail)
        if head.casefold() != tail.casefold() and key not in seen:
            result.append(list(key))
            seen.add(key)
    return result


def get_nodes_from_triples(triples):
    return list(dict.fromkeys(node for h, _, t in filter_concepts(triples) for node in (h, t)))


def remove_dup_concepts(triples, duplicate_groups):
    if not isinstance(duplicate_groups, dict):
        raise ValueError("Duplicate concepts must be a mapping of names to alias lists")
    aliases = {}
    for canonical, variants in duplicate_groups.items():
        if not isinstance(canonical, str) or not isinstance(variants, list):
            continue
        aliases[canonical.casefold()] = canonical
        for variant in variants:
            if isinstance(variant, str):
                aliases[variant.casefold()] = canonical
    return filter_concepts([
        [aliases.get(h.casefold(), h), r, aliases.get(t.casefold(), t)]
        for h, r, t in filter_concepts(triples)
    ])


def get_plurals(concepts):
    """Merge only simple plural/singular pairs already present in the graph."""
    observed = {concept.casefold(): concept for concept in concepts}
    result = {}
    for concept in concepts:
        word = concept.casefold()
        candidates = []
        if word.endswith('ies'):
            candidates.append(word[:-3] + 'y')
        if word.endswith(('ches', 'shes', 'xes', 'zes', 'sses')):
            candidates.append(word[:-2])
        if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
            candidates.append(word[:-1])
        for candidate in candidates:
            if candidate in observed:
                result[concept] = observed[candidate]
                break
    return result


def filter_generic_concepts(triples):
    """Exclude explicit placeholder labels, without guessing domain relevance."""
    placeholders = {'', 'none', 'null', 'n/a', 'unknown', 'something', 'anything'}
    return [
        [h, r, t] for h, r, t in filter_concepts(triples)
        if h.casefold() not in placeholders and t.casefold() not in placeholders
    ]


def merge_relations(triples):
    grouped = {}
    for head, relation, tail in filter_concepts(triples):
        relations = grouped.setdefault((head, tail), [])
        if relation not in relations:
            relations.append(relation)
    return [[h, '; '.join(relations), t] for (h, t), relations in grouped.items()]


def get_network_di_graph(triples):
    graph = nx.DiGraph()
    for head, relation, tail in merge_relations(triples):
        graph.add_edge(head, tail, relationship=relation)
    return graph


def get_triple_from_di_graph(graph):
    return [[h, data['relationship'], t] for h, t, data in graph.edges(data=True)]


def remove_isolated_triples(triples):
    """Remove disconnected one-edge fragments; keep a standalone small map."""
    triples = filter_concepts(triples)
    graph = get_network_di_graph(triples).to_undirected()
    components = list(nx.connected_components(graph))
    connected_nodes = set().union(*(nodes for nodes in components if len(nodes) > 2))
    if not connected_nodes:
        return triples
    return [[h, r, t] for h, r, t in triples if h in connected_nodes and t in connected_nodes]


def convert_graph_to_pydot_string(graph):
    """Serialize labels as quoted DOT strings without requiring Graphviz."""
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ['digraph concepts {']
    lines.extend(f'  {quote(node)};' for node in graph.nodes)
    lines.extend(
        f'  {quote(h)} -> {quote(t)} [label={quote(data["relationship"])}];'
        for h, t, data in graph.edges(data=True)
    )
    return '\n'.join([*lines, '}'])
