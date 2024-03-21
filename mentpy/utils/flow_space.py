# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""A module to study graphs with flow."""
import itertools
import networkx as nx

from mentpy.mbqc import MBQCircuit, GraphState

__all__ = ["FlowSpace"]


class FlowSpace:
    r"""The flow space graph of a MBQCircuit.

    Each node corresponds to a possible graph over ``n_qubits`` qubits.
    Each edge between nodes represent going from one graph to another via adding or removing edges.

    Args
    ----
    n_qubits: int
        The number of qubits in the graph state.
    input_nodes: list
        The input nodes of the graph state.
    output_nodes: list
        The output nodes of the graph state.

    .. note::  ``flow_space()`` will only work for MBQCircuit with less
    than 8 qubits.

    Group
    -----
    utils
    """

    def __init__(
        self, n_qubits, input_nodes, output_nodes, allow_any_size_graphs: bool = False
    ):
        """Creates the flow graph space of a graph state circuit."""

        if n_qubits > 7 and (not allow_any_size_graphs):
            raise UserWarning(
                f"Expected a graph_state of size 7 or less, but {n_qubits} "
                "was given."
            )

        self.number_nodes = n_qubits
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        total_graph, wf_graph, wof_graph = self._generate_graph_spaces()
        self.total_graph_space = total_graph
        self.flow_graph_space = wf_graph
        self.no_flow_graph_space = wof_graph

    def __repr__(self) -> str:
        r"""Returns the representation of the flow space"""
        return (
            f"Flow space over {self.number_nodes} nodes with {self.input_nodes} input nodes "
            f"and {self.output_nodes} output nodes."
        )

    def _generate_all_graphs(self):
        """Generator that yields all possible graphs for the specified number of nodes."""
        n = self.number_nodes
        total_edges = int(n * (n - 1) / 2)
        complete_graph = nx.complete_graph(n)
        edge_combinations = itertools.product([0, 1], repeat=total_edges)

        for edge_selection in edge_combinations:
            g = GraphState()
            g.add_nodes_from(range(n))
            edges = [
                edge
                for i, edge in enumerate(complete_graph.edges())
                if edge_selection[i]
            ]
            g.add_edges_from(edges)
            yield g

    def _generate_graph_spaces(self):
        """
        Generates the total graph space and its subgraphs for states with and without flow.

        Returns:
            tuple: A tuple containing three nx.Graph objects (total_graph_space, flow_graph_space, no_flow_graph_space).
        """
        graphs_list = list(self._generate_all_graphs())
        graph_space = nx.Graph()
        with_flow, without_flow = [], []

        for idx, graph in enumerate(graphs_list):
            mbqc_graph = MBQCircuit(
                graph, input_nodes=self.input_nodes, output_nodes=self.output_nodes
            )
            mbqc_graph.flow.initialize_flow()
            layers = mbqc_graph.flow.layers

            if layers is not None:
                graph_space.add_node(idx, flow=True, mbqc_circuit=mbqc_graph)
                with_flow.append(idx)
            else:
                graph_space.add_node(idx, flow=False, mbqc_circuit=mbqc_graph)
                without_flow.append(idx)

        for (idx1, graph1), (idx2, graph2) in itertools.combinations(
            enumerate(graphs_list), 2
        ):
            if len(set(graph2.edges()).symmetric_difference(graph1.edges())) == 1:
                graph_space.add_edge(idx1, idx2)

        return (
            graph_space,
            graph_space.subgraph(with_flow),
            graph_space.subgraph(without_flow),
        )


if __name__ == "__main__":
    import cProfile
    import pstats
    import os
    import subprocess

    n_qubits = 4
    input_nodes = [0]
    output_nodes = [3]
    profile_filename = "profile_stats.prof"
    cProfile.run("FlowSpace(n_qubits, input_nodes, output_nodes)", profile_filename)

    dot_filename = "profile_output.dot"
    png_filename = "profile_output.png"
    with open(dot_filename, "w") as f:
        subprocess.run(["gprof2dot", "-f", "pstats", profile_filename], stdout=f)

    # Convert .dot to .png
    subprocess.run(["dot", "-Tpng", dot_filename, "-o", png_filename])

    # Optional: Remove the .dot file if not needed
    os.remove(dot_filename)

    print(f"Profile image saved as {png_filename}")
