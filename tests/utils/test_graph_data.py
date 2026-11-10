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
