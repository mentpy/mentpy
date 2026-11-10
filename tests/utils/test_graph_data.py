"""Regression tests for graph analysis and random-state datasets."""

import networkx as nx
import numpy as np
import pytest

import mentpy as mp


def test_local_clifford_equivalence():
    singleton = mp.GraphState()
    singleton.add_node("a")
    assert mp.utils.are_lc_equivalent(singleton, singleton)[0]
    path = nx.path_graph(3)
    triangle = nx.complete_graph(3)
    assert mp.utils.are_lc_equivalent(path, triangle)[0]
    assert not mp.utils.are_lc_equivalent(path, nx.empty_graph(3))[0]


def test_flow_space_is_edge_mask_hypercube():
    with pytest.warns(UserWarning, match="No flow found"):
        space = mp.utils.FlowSpace(3, [0], [2])
    assert set(space.total_graph_space) == set(range(8))
    assert space.total_graph_space.number_of_edges() == 12
    for left, right in space.total_graph_space.edges:
        assert (left ^ right).bit_count() == 1


@pytest.mark.parametrize(
    "generate",
    [
        mp.utils.generate_random_dataset,
        mp.utils.generate_random_dataset_noisy,
    ],
)
def test_random_dataset_accepts_single_sample(generate):
    (inputs, targets), (test_inputs, test_targets) = generate(np.eye(2), 1)
    assert len(inputs) == len(targets) == 1
    np.testing.assert_allclose(inputs, targets)
    assert np.isclose(np.linalg.norm(inputs[0]), 1)
    assert test_inputs == test_targets == []


def test_train_test_split_shuffles_paired_lists():
    train, test = mp.utils.train_test_split(
        [1, 2, 3], [10, 20, 30], test_size=0.5, randomize=True
    )
    for inputs, targets in (train, test):
        assert [10 * x for x in inputs] == targets


def _local_complement(graph, vertex):
    from itertools import combinations

    result = graph.copy()
    for u, v in combinations(graph[vertex], 2):
        if result.has_edge(u, v):
            result.remove_edge(u, v)
        else:
            result.add_edge(u, v)
    return result


def _check_clifford(left, right, clifford):
    # A graph's stabilizers are the columns of [adjacency; identity].
    n = len(left)
    g = nx.to_numpy_array(left, dtype=int, weight=None)
    h = nx.to_numpy_array(right, nodelist=list(left), dtype=int, weight=None)
    source = np.vstack((g, np.eye(n, dtype=int)))
    target = np.vstack((h, np.eye(n, dtype=int)))
    p = np.block(
        [
            [np.zeros((n, n), int), np.eye(n, dtype=int)],
            [np.eye(n, dtype=int), np.zeros((n, n), int)],
        ]
    )
    np.testing.assert_array_equal((clifford.T @ p @ clifford) % 2, p)
    np.testing.assert_array_equal((source.T @ clifford.T @ p @ target) % 2, 0)


def test_lc_matches_all_four_vertex_local_complementation_orbits():
    from itertools import combinations

    edges = list(combinations(range(4), 2))

    def mask(graph):
        return sum(
            1 << index for index, edge in enumerate(edges) if graph.has_edge(*edge)
        )

    graphs = []
    for bits in range(64):
        graph = nx.empty_graph(4)
        graph.add_edges_from(
            edge for index, edge in enumerate(edges) if bits & (1 << index)
        )
        graphs.append(graph)
    orbits = {}
    for bits, graph in enumerate(graphs):
        if bits in orbits:
            continue
        seen, pending = {bits}, [graph]
        while pending:
            current = pending.pop()
            for vertex in current:
                other = _local_complement(current, vertex)
                key = mask(other)
                if key not in seen:
                    seen.add(key)
                    pending.append(other)
        for key in seen:
            orbits[key] = seen
    for i, left in enumerate(graphs):
        for j, right in enumerate(graphs):
            actual, certificate = mp.utils.are_lc_equivalent(left, right, "global")
            assert actual == (j in orbits[i]), (i, j)
            if actual:
                _check_clifford(left, right, certificate)


def test_lc_wide_and_disconnected_certificates():
    for graph in (
        nx.path_graph(65),
        nx.star_graph(65),
        nx.empty_graph(5),
        nx.disjoint_union(nx.path_graph(4), nx.path_graph(3)),
    ):
        other = graph
        for vertex in list(graph)[::3]:
            other = _local_complement(other, vertex)
        actual, certificate = mp.utils.are_lc_equivalent(graph, other, "global")
        assert actual
        _check_clifford(graph, other, certificate)
    # Correspondence follows labels even if the two graphs have different order.
    graph = nx.relabel_nodes(nx.path_graph(4), {i: f"q{i}" for i in range(4)})
    other = nx.Graph()
    other.add_nodes_from(reversed(list(graph)))
    other.add_edges_from(_local_complement(graph, "q1").edges)
    actual, certificate = mp.utils.are_lc_equivalent(graph, other, "global")
    assert actual
    _check_clifford(graph, other, certificate)


def test_lc_rejects_invalid_graphs_and_native_input():
    from mentpy import _rust

    loop = nx.Graph([(0, 0)])
    for graph in (loop, nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)])):
        with pytest.raises(ValueError, match="simple undirected"):
            mp.utils.are_lc_equivalent(graph, graph)
    for data in (b"", b"\x00\x01\x00\x00", b"\x01\x00\x00\x00", b"\x00\x02\x02\x00"):
        with pytest.raises(ValueError):
            _rust.lc_equivalent(data, bytes(4), 2)
    assert _rust.lc_equivalent(b"", b"", 0) is None
