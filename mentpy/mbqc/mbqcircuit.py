# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""The graph_state module"""

import copy
from functools import cached_property, reduce
from typing import Optional, List, Tuple, Callable, Dict

import networkx as nx

from mentpy.operators import Ment, ControlMent, MentOutcome
from mentpy.mbqc.states.graphstate import GraphState
from mentpy.mbqc.flow import Flow

__all__ = ["MBQCircuit", "merge", "hstack", "vstack"]


class MBQCircuit:
    r"""The MBQCircuit class that deals with operations and manipulations of graph states

    Parameters
    ----------
    graph: mp.GraphState
        The graph state of the MBQC circuit.
    input_nodes: list
        The input nodes of the MBQC circuit.
    output_nodes: list
        The output nodes of the MBQC circuit.
    measurements: dict
        The measurements of the MBQC circuit. The keys are the nodes and the values are the measurements.
    default_measurement: mp.Ment
        The default measurement of the MBQC circuit if no `measurements` are given.
    initialize_flow: bool
        Whether to find a flow during construction. Set to ``False`` when a
        caller only needs fixed-outcome tensor-network contraction and wants to
        avoid eager flow solving for large generated patterns.

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.GraphState()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        state = mp.MBQCircuit(g, input_nodes=[0], output_nodes=[4])


    See Also
    --------
    :class:`mp.GraphState`

    Group
    -----
    mbqc
    """

    def __init__(
        self,
        graph: GraphState,
        input_nodes: Optional[List[int]] = None,
        output_nodes: Optional[List[int]] = None,
        measurements: Optional[Dict[int, Ment]] = None,
        default_measurement: Optional[Ment] = None,
        relabel_indices: bool = True,
        initialize_flow: bool = True,
    ) -> None:
        """Initializes a graph state"""

        input_nodes = [] if input_nodes is None else list(input_nodes)
        output_nodes = [] if output_nodes is None else list(output_nodes)
        if len(set(input_nodes)) != len(input_nodes) or len(set(output_nodes)) != len(
            output_nodes
        ):
            raise ValueError(
                "Input and output node lists must each contain unique nodes."
            )
        if nx.number_of_selfloops(graph):
            raise ValueError("MBQC circuit graphs cannot contain self-loops.")
        default_measurement = (
            Ment("XY") if default_measurement is None else default_measurement
        )
        if relabel_indices:
            graph, input_nodes, output_nodes, measurements = self._relabel_graph(
                graph, input_nodes, output_nodes, measurements
            )

        self._graph = graph
        self._setup_nodes(input_nodes, output_nodes)
        self._setup_measurements(measurements, default_measurement)
        self._update_attributes()

        # create flow
        self._flow = Flow(
            graph,
            input_nodes,
            output_nodes,
            {v: m.plane for v, m in self.measurements.items() if m is not None},
        )
        self._initialize_flow = initialize_flow
        if initialize_flow:
            self._flow.initialize_flow()

        self._measurement_order = None if initialize_flow else list(self.graph.nodes)

    def _relabel_graph(
        self,
        graph: nx.Graph,
        input_nodes: List[int],
        output_nodes: List[int],
        measurements: Optional[Dict[int, "Ment"]],
    ) -> Tuple[nx.Graph, List[int], List[int], Optional[Dict[int, "Ment"]]]:
        """Relabels the graph nodes and updates related node indices.

        Args:
            graph (nx.Graph): The original graph.
            input_nodes (List[int]): Original input node indices.
            output_nodes (List[int]): Original output node indices.
            measurements (Optional[Dict[int, 'Ment']]): Original node measurements.

        Returns:
            tuple: A tuple containing the relabeled graph, input nodes, output nodes, and measurements.
        """
        N = graph.number_of_nodes()
        try:
            nodes = sorted(graph.nodes)
        except TypeError:
            nodes = list(graph.nodes)
        mapping = dict(zip(nodes, range(N)))
        if not set(input_nodes + output_nodes) <= mapping.keys():
            raise ValueError("Input and output nodes must be in the graph.")
        if measurements is not None and not measurements.keys() <= mapping.keys():
            raise ValueError("Measurement nodes must be in the graph.")
        graph = nx.relabel_nodes(graph, mapping)
        input_nodes = [mapping[i] for i in input_nodes]
        output_nodes = [mapping[i] for i in output_nodes]
        if measurements is not None:
            measurements = _remap_measurements(measurements, mapping)

        return graph, input_nodes, output_nodes, measurements

    def _setup_nodes(self, input_nodes: List[int], output_nodes: List[int]) -> None:
        """Setup the input and output nodes of the MBQCircuit"""
        if not all([v in self.graph.nodes for v in input_nodes]):
            raise ValueError(
                f"Input nodes {input_nodes} are not in the graph. Graph nodes are {self.graph.nodes}"
            )
        if not all([v in self.graph.nodes for v in output_nodes]):
            raise ValueError(
                f"Output nodes {output_nodes} are not in the graph. Graph nodes are {self.graph.nodes}"
            )

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

    def _setup_measurements(
        self, measurements: Dict[int, Ment], default_measurement: Ment
    ) -> None:
        """Setup the measurements of the MBQCircuit"""
        # Type check default_measurement
        if not isinstance(default_measurement, Ment):
            raise ValueError(
                f"Default measurement {default_measurement} is not an instance of Ment."
            )
        self._default_measurement = default_measurement

        # Type check measurements
        if measurements is None:
            measurements = {node: default_measurement for node in self.outputc}
            for node in self.output_nodes:
                measurements[node] = None
        else:
            if not all([v in self.graph.nodes for v in measurements.keys()]):
                nodes_not_in_graph = [
                    v for v in measurements.keys() if v not in self.graph.nodes
                ]
                raise ValueError(f"Nodes {nodes_not_in_graph} are not in the graph.")
            if not all(
                [isinstance(v, Ment) or v is None for v in measurements.values()]
            ):
                raise ValueError(
                    f"Values {measurements.values()} are not instances of Ment."
                )

            measurements = measurements.copy()
            for node in self.graph.nodes:
                if node not in measurements:
                    measurements[node] = (
                        self._default_measurement if node in self.outputc else None
                    )

        self._measurements = {
            node: self._copy_measurement(node, measurement)
            for node, measurement in measurements.items()
        }

    @staticmethod
    def _copy_measurement(node, measurement):
        if measurement is None:
            return None
        measurement = copy.deepcopy(measurement)
        measurement.node_id = node
        return measurement

    def __repr__(self) -> str:
        """Return the representation of the current MBQC circuit state"""
        return f"MBQCircuit with {self.graph.number_of_nodes()} qubits."

    def __len__(self) -> int:
        """Return the number of nodes in the MBQCircuit"""
        return len(self.graph)

    def __getattr__(self, name):
        # If the attribute is not found in the MBQCircuit, try to find it in the graph or flow
        try:
            return getattr(object.__getattribute__(self, "_graph"), name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_flow"), name)

    def __setitem__(self, key, value):
        r"""Set the value of the measurement of the node with index key."""
        if key not in self.graph.nodes:
            raise ValueError(f"Node {key} is not in the graph.")
        if not isinstance(value, Ment):
            raise ValueError(f"Value {value} is not a Measurement object.")

        controlled = isinstance(self._measurements[key], ControlMent) or isinstance(
            value, ControlMent
        )
        self._measurements[key] = self._copy_measurement(key, value)
        self._update_attributes()
        if self._update_flow_planes() or controlled:
            if self._initialize_flow:
                self._measurement_order = None

    def _update_flow_planes(self):
        planes = {
            node: ment.plane
            for node, ment in self.measurements.items()
            if ment is not None
        }
        if planes != self._flow.planes:
            self._flow.planes = planes
            self._flow.flow_initialized = False
            return True
        return False

    def __getitem__(self, key):
        r"""Return the value of the measurement of the node with index key."""
        try:
            return self._measurements[key]
        except KeyError:
            raise ValueError(f"Node {key} is not in the graph.")

    def __delitem__(self, key):
        """Delete the measurement of the node with index key."""

        if key not in self.graph.nodes:
            raise ValueError(f"Node {key} is not in the graph.")

        self._measurements[key] = None
        self._update_attributes()
        self._update_flow_planes()
        if self._initialize_flow:
            self._measurement_order = None

    @property
    def measurements(self) -> Dict[int, Ment]:
        r"""Return the measurements of the MBQC circuit."""
        return self._measurements

    @measurements.setter
    def measurements(self, measurements: Dict[int, Ment]) -> None:
        r"""Set the measurements of the MBQC circuit."""
        self._setup_measurements(measurements, self._default_measurement)
        self._update_attributes()
        self._update_flow_planes()
        if self._initialize_flow:
            self._measurement_order = None

    @property
    def graph(self) -> GraphState:
        r"""Return the graph of the resource state."""
        return self._graph

    @property
    def input_nodes(self) -> List[int]:
        r"""Return the input nodes of the MBQC circuit."""
        return self._input_nodes

    @property
    def output_nodes(self) -> List[int]:
        r"""Return the output nodes of the MBQC circuit."""
        return self._output_nodes

    @property
    def quantum_output_nodes(self) -> List[int]:
        r"""Return the output nodes of the MBQC circuit."""
        return self._quantum_output_nodes

    @property
    def classical_output_nodes(self) -> List[int]:
        r"""Return the output nodes of the MBQC circuit."""
        return self._classical_output_nodes

    @property
    def trainable_nodes(self) -> List[int]:
        r"""Return the trainable nodes of the MBQC circuit."""
        return self._trainable_nodes

    @trainable_nodes.setter
    def trainable_nodes(self, trainable_nodes: List[int]) -> None:
        r"""Set the trainable nodes of the MBQC circuit."""
        if not all([v in self.graph.nodes for v in trainable_nodes]):
            raise ValueError(
                f"Trainable nodes {trainable_nodes} are not in the graph. Graph nodes are {self.graph.nodes}"
            )
        self._trainable_nodes = trainable_nodes

    @property
    def controlled_nodes(self) -> List[int]:
        r"""Return the controlled nodes of the MBQC circuit."""
        return self._controlled_nodes

    @property
    def flow(self) -> Flow:
        r"""Return the flow function of the MBQC circuit."""
        return self._flow

    @property
    def partial_order(self) -> Callable:
        r"""Return the partial order function of the MBQC circuit."""
        if not self._flow.flow_initialized:
            self._flow.initialize_flow()
        order = self._flow.partial_order
        if order is not None and self.controlled_nodes:
            return _create_new_partial_order(
                self.controlled_nodes, self.measurements, order
            )
        return order

    @property
    def depth(self) -> int:
        r"""Return the depth of the MBQC circuit."""
        if not self._flow.flow_initialized:
            self._flow.initialize_flow()
        return self._flow.depth

    @property
    def measurement_order(self) -> List[int]:
        r"""Return the measurement order of the MBQC circuit."""
        if not self._initialize_flow and self._measurement_order is not None:
            return self._measurement_order
        if self._flow.flow_initialized is False:
            self._flow.initialize_flow()
        if self._measurement_order is None:
            self._measurement_order = self.calculate_order()

        return self._measurement_order

    @measurement_order.setter
    def measurement_order(self, measurement_order: List[int]) -> None:
        r"""Set the measurement order of the MBQC circuit."""
        order = list(measurement_order)
        if (
            len(order) != len(self.graph)
            or set(order) != set(self.graph)
            or not _check_measurement_order(order, self.partial_order)
        ):
            raise ValueError(f"Invalid measurement order {measurement_order}.")
        self._measurement_order = order

    @cached_property
    def outputc(self) -> List:
        r"""Returns :math:`O^c`, the complement of output nodes."""
        return [v for v in self.graph.nodes() if v not in self.output_nodes]

    @cached_property
    def inputc(self) -> List:
        r"""Returns :math:`I^c`, the complement of input nodes."""
        return [v for v in self.graph.nodes() if v not in self.input_nodes]

    def ordered_layers(self, train_indices=False) -> List[List[int]]:
        r"""Returns the layers of the MBQC circuit."""
        self._flow.initialize_flow()
        if self._flow.func is None:
            return None
        if train_indices:
            indices = {node: i for i, node in enumerate(self.trainable_nodes)}
            return [
                [indices[node] for node in layer if node in indices]
                for layer in self._flow.layers
                if any(node in indices for node in layer)
            ]
        return self._flow.layers

    def _update_attributes(self) -> None:
        trainable_nodes = []
        controlled_nodes = []
        quantum_outputs = []
        classical_outputs = []

        for nodei, menti in self._measurements.items():
            if menti is not None:
                if isinstance(menti, ControlMent):
                    controlled_nodes.append(nodei)

                if menti.is_trainable():
                    trainable_nodes.append(nodei)

                if nodei in self._output_nodes:
                    classical_outputs.append(nodei)
            else:
                if nodei in self._output_nodes:
                    quantum_outputs.append(nodei)

        self._trainable_nodes = trainable_nodes
        self._controlled_nodes = controlled_nodes
        self._quantum_output_nodes = [
            n for n in self.output_nodes if n in quantum_outputs
        ]
        self._classical_output_nodes = [
            n for n in self.output_nodes if n in classical_outputs
        ]

    def calculate_order(self):
        r"""Returns the order of the measurements"""
        if not self.graph:
            return []
        partial_order = self.partial_order
        if partial_order is None:
            return None
        if self.controlled_nodes:
            nodes = list(self.graph)
            inputs = set(self.input_nodes)
            dependencies = nx.DiGraph()
            dependencies.add_nodes_from(nodes)
            dependencies.add_edges_from(
                (u, v) for u in nodes for v in nodes if partial_order(u, v)
            )
            try:
                return list(
                    nx.lexicographical_topological_sort(
                        dependencies, key=lambda node: node not in inputs
                    )
                )
            except nx.NetworkXUnfeasible as exc:
                raise ValueError("Measurement dependencies contain a cycle.") from exc
        layers = {
            node: index
            for index, layer in enumerate(self._flow.layers)
            for node in layer
        }
        order = sorted(self.graph, key=layers.__getitem__)
        inputs = set(self.input_nodes)
        # NumPy windows expect the input nodes first when the flow permits it.
        last_input_layer = max((layers[node] for node in inputs), default=-1)
        if any(layers[node] < last_input_layer for node in order if node not in inputs):
            return order
        input_order = sorted(self.input_nodes, key=layers.__getitem__)
        return input_order + [node for node in order if node not in inputs]

    def add_edge(self, u, v):
        r"""Add an edge between nodes u and v."""
        self.add_edges_from([(u, v)])

    def add_edges_from(self, edges, **kwargs):
        r"""Add edges while preserving the existing measurements."""
        new_graph = self.graph.copy()
        new_graph.add_edges_from(edges, **kwargs)
        updated = MBQCircuit(
            new_graph,
            self.input_nodes,
            self.output_nodes,
            measurements=self.measurements,
            default_measurement=self._default_measurement,
            relabel_indices=False,
            initialize_flow=self._initialize_flow,
        )
        self.__dict__.pop("inputc", None)
        self.__dict__.update(updated.__dict__)


def _check_measurement_order(measurement_order: List, partial_order: Callable):
    r"""Checks that the measurement order is valid"""
    if partial_order is None:
        return True
    for i in range(len(measurement_order)):
        for j in range(i + 1, len(measurement_order)):
            if partial_order(measurement_order[j], measurement_order[i]):
                return False
    return True


def merge(
    state1: MBQCircuit,
    state2: MBQCircuit,
    along=(),
    initialize_flow: bool = True,
) -> MBQCircuit:
    """Merge two MBQC circuits into a larger MBQC circuit. This is, the input and
    output of the new MBQC circuit will depend on the concat_indices.

    Group
    -----
    mbqc
    """

    along = list(along)
    if not along:
        return _vstack2(state1, state2, initialize_flow=initialize_flow)
    for i, j in along:
        if i not in state1.output_nodes or j not in state2.input_nodes:
            raise ValueError(f"Cannot merge states at indices {i} and {j}")

    connections = dict(along)
    if len(connections) != len(along) or len(set(connections.values())) != len(along):
        raise ValueError("Each merge boundary node can be joined only once.")

    graph, left, right, measurements = _disjoint_parts(state1, state2)
    junctions = {left[i]: right[j] for i, j in along}
    input_nodes = [junctions.get(left[i], left[i]) for i in state1.input_nodes]
    consumed = set(connections.values())
    input_nodes.extend(right[i] for i in state2.input_nodes if i not in consumed)

    output_nodes = []
    paired_outputs = dict(zip(state2.input_nodes, state2.output_nodes))
    inserted_outputs = set()
    for output in state1.output_nodes:
        if output not in connections:
            output_nodes.append(left[output])
        else:
            joined_input = connections[output]
            if joined_input in paired_outputs:
                paired_output = paired_outputs[joined_input]
                output_nodes.append(right[paired_output])
                inserted_outputs.add(paired_output)
    output_nodes.extend(
        right[i] for i in state2.output_nodes if i not in inserted_outputs
    )

    graph, measurements = _join_nodes(graph, measurements, junctions)

    return MBQCircuit(
        graph,
        input_nodes,
        output_nodes,
        measurements=measurements,
        initialize_flow=initialize_flow,
    )


def vstack(states, initialize_flow: bool = True) -> MBQCircuit:
    """Vertically stack a list of graph states into a larger graph state. This is,
    the input of the new MBQC circuit is the input of the first state, and the output
    is the output of the last state.

    Group
    -----
    mbqc
    """
    if len(states) == 0:
        raise ValueError("Cannot vertically stack an empty list of states.")
    if len(states) == 1:
        return states[0]
    return reduce(
        lambda state1, state2: _vstack2(
            state1, state2, initialize_flow=initialize_flow
        ),
        states,
    )


def hstack(states, initialize_flow: bool = True) -> MBQCircuit:
    """Horizontally stack a list of graph states into a larger graph state. This is,
    the input of the new MBQC circuit is the input of the first state, and the output
    is the output of the last state.

    Args
    ----
    initialize_flow: bool
        Whether to find a flow while constructing intermediate stacked states.

    Group
    -----
    mbqc
    """
    if len(states) == 0:
        raise ValueError("Cannot horizontally stack an empty list of states.")
    if len(states) == 1:
        return states[0]
    return reduce(
        lambda state1, state2: _hstack2(
            state1, state2, initialize_flow=initialize_flow
        ),
        states,
    )


def _vstack2(
    state1: MBQCircuit,
    state2: MBQCircuit,
    initialize_flow: bool = True,
) -> MBQCircuit:
    """Vertically stack two graph states into a larger graph state. This is,
    the input of the new MBQC circuits is both the input of the first and second
    state, and the output is the output of the first and second state.

    Group
    -----
    mbqc
    """
    graph, left, right, measurements = _disjoint_parts(state1, state2)
    input_nodes = [left[i] for i in state1.input_nodes]
    input_nodes.extend(right[i] for i in state2.input_nodes)
    output_nodes = [left[i] for i in state1.output_nodes]
    output_nodes.extend(right[i] for i in state2.output_nodes)

    return MBQCircuit(
        graph,
        input_nodes,
        output_nodes,
        measurements=measurements,
        initialize_flow=initialize_flow,
    )


def _hstack2(
    state1: MBQCircuit,
    state2: MBQCircuit,
    initialize_flow: bool = True,
) -> MBQCircuit:
    """Horizontally stack two graph states into a larger graph state. This is,
    the input of the new MBQC circuit is the input of the first state, and the
    output is the output of the second state.

    Args
    ----
    state1: MBQCircuit
        The first state to stack.
    state2: MBQCircuit
        The second state to stack.

    Group
    -----
    mbqc
    """
    # check that the size of the output of the first state is the same as the
    # size of the input of the second state
    if len(state1.output_nodes) != len(state2.input_nodes):
        raise ValueError(
            "The output of the first state must be the same size as the input "
            "of the second state."
        )

    graph, left, right, measurements = _disjoint_parts(state1, state2)
    junctions = {
        left[output]: right[input_node]
        for output, input_node in zip(state1.output_nodes, state2.input_nodes)
    }
    input_nodes = [junctions.get(left[i], left[i]) for i in state1.input_nodes]
    output_nodes = [right[i] for i in state2.output_nodes]

    graph, measurements = _join_nodes(graph, measurements, junctions)

    return MBQCircuit(
        graph,
        input_nodes,
        output_nodes,
        measurements=measurements,
        initialize_flow=initialize_flow,
    )


def _disjoint_parts(state1, state2):
    left = {node: i for i, node in enumerate(state1.graph)}
    right = {node: i for i, node in enumerate(state2.graph, start=len(left))}
    measurements = _remap_measurements(state1.measurements, left)
    measurements.update(_remap_measurements(state2.measurements, right))
    return nx.disjoint_union(state1.graph, state2.graph), left, right, measurements


def _join_nodes(graph, measurements, junctions):
    mapping = {node: junctions.get(node, node) for node in graph}
    measurements = _remap_measurements(
        {node: ment for node, ment in measurements.items() if node not in junctions},
        mapping,
    )
    for old, new in junctions.items():
        graph = nx.contracted_nodes(graph, new, old, self_loops=False)
    return graph, measurements


def _remap_measurements(measurements, mapping):
    if all(old == new for old, new in mapping.items()):
        return measurements.copy()

    def remap_condition(condition):
        dependencies = {mapping[node] for node in condition.cond_nodes}

        def evaluate(outcomes):
            return condition(
                {old: outcomes[new] for old, new in mapping.items() if new in outcomes}
            )

        return MentOutcome(evaluate, cond_nodes=dependencies)

    remapped = {}
    for node, measurement in measurements.items():
        if isinstance(measurement, ControlMent) and isinstance(
            measurement.condition, MentOutcome
        ):
            measurement = measurement.copy()
            measurement.condition = remap_condition(measurement.condition)
        remapped[mapping[node]] = measurement
    return remapped


def _create_new_partial_order(controlled_nodes, measurements, old_partial_order):
    dependencies = {
        (source, node)
        for node in controlled_nodes
        for source in getattr(measurements[node].condition, "cond_nodes", ())
    }

    def new_partial_order(i, j):
        return (i, j) in dependencies or old_partial_order(i, j)

    return new_partial_order
