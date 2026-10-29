# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""A module for drawing MBQC circuits."""

from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.mbqc.states.graphstate import GraphState

import warnings

__all__ = ["draw", "draw_with_wires"]

DEFAULT_NODE_COLOR = "#FFBD59"
INPUT_NODE_COLOR = "#ADD8E6"
OUTPUT_NODE_COLOR = "#ADD8E6"
CONTROLLED_NODE_COLOR = "#A88FE8"
UNTRAINABLE_NODE_COLOR = "#CCCCCC"


def get_node_colors(state, style="default"):
    """Return node colors based on the state and style."""

    possible_styles = ("default", "black_and_white", "blue_inputs")
    assert style in possible_styles, f"Style must be one of {possible_styles}"

    node_colors = {}

    # Base Coloring
    for i in state.graph.nodes():
        if i in state.controlled_nodes:
            node_colors[i] = CONTROLLED_NODE_COLOR
        elif i in state.quantum_output_nodes:
            node_colors[i] = OUTPUT_NODE_COLOR
        elif i in set(state.nodes()) - set(state.trainable_nodes):
            node_colors[i] = UNTRAINABLE_NODE_COLOR
        else:
            node_colors[i] = DEFAULT_NODE_COLOR

    # Style-based Adjustments
    if style == "black_and_white":
        node_colors = {i: "#FFFFFF" for i in state.graph.nodes()}
    elif style == "blue_inputs":
        for i in state.input_nodes:
            node_colors[i] = INPUT_NODE_COLOR

    return node_colors


def get_options(kwargs) -> dict:
    """Returns default options updated with user-defined values."""
    default_options = {
        "node_color": "white",
        "font_family": "Dejavu Sans",
        "font_weight": "medium",
        "font_size": 10,
        "edgecolors": "k",
        "node_size": 500,
        "edge_color": "grey",
        "edge_color_control": "#CCCCCC",
        "with_labels": True,
        "label": "indices",
        "transparent": True,
        "figsize": (8, 3),
        "show_controls": True,
        "show_flow": True,
        "pauliop": None,
        "style": "default",
        "position": None,
        "layout": None,
        "title": None,
    }

    # Update default options with any provided by the user
    default_options.update(kwargs)

    return default_options


def draw(state: Union[MBQCircuit, GraphState], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Draws mbqc circuit with flow.

    Group
    -----
    mbqc
    """

    options = get_options(kwargs)

    show_controls = options.pop("show_controls")
    show_flow = options.pop("show_flow")
    pauliop = options.get("pauliop", None)
    edge_color_control = options.pop("edge_color_control")
    style = options.pop("style")
    position = options.pop("position")
    layout = options.pop("layout")
    title = options.pop("title")

    if layout == "pauli" and pauliop is None:
        pauliop = _infer_pauliop_from_template(state)
        options["pauliop"] = pauliop

    if pauliop is not None:
        if len(pauliop) != 1:
            raise ValueError("pauliop must be a single Pauli operator")
        options["label"] = "pauliop"

    if layout == "pauli":
        options.pop("label", None)
        options.pop("pauliop", None)
        if "labels" in options:
            options.pop("labels")
    elif "labels" not in options:
        options["labels"] = process_labels(state, options)
    else:
        options.pop("pauliop")

    transp = options.pop("transparent")
    fig, ax = plt.subplots(figsize=options.pop("figsize"))

    if transp:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    if isinstance(state, GraphState):
        nx.draw(state, position, ax=ax, **options)

    elif isinstance(state, MBQCircuit):
        if layout == "pauli":
            return _draw_pauli_template(
                state,
                ax=ax,
                title=title,
                pauliop=pauliop,
                options=options,
            )
        if state.flow is None:
            nx.draw(state.graph, position, ax=ax, **options)
        elif state.flow.name.lower() == "cflow":
            plt.close(fig)
            return draw_with_wires(state, **kwargs)
        else:
            layers = state.flow.layers
            node_colors = get_node_colors(state, style=style)
            options["node_color"] = [node_colors[node] for node in state.graph.nodes()]

            position_xy = {}
            for i, layer in enumerate(layers):
                for j, node in enumerate(layer):
                    position_xy[node] = (i, -j)

            nx.draw(state.graph, ax=ax, pos=position_xy, **options)

            if show_flow:
                nx.draw(_graph_with_flow(state), pos=position_xy, ax=ax, **options)

            if show_controls:
                dashed_edges = []
                for node in state.controlled_nodes:
                    for k in state.measurements[node].condition.cond_nodes:
                        dashed_edges.append((node, k))
                nx.draw_networkx_edges(
                    state.graph,
                    pos=position_xy,
                    edge_color=edge_color_control,
                    width=1.5,
                    edgelist=dashed_edges,
                    style="dashed",
                )

    return fig, ax


def draw_with_wires(
    state: Union[MBQCircuit, GraphState], fix_wires=None, **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Draws mbqc circuit with flow.

    TODO: Add support for graphs without flow, but with gflow
    TODO: Improve fix when there are control nodes

    Group
    -----
    mbqc
    """
    options = get_options(kwargs)

    show_controls = options.pop("show_controls")
    show_flow = options.pop("show_flow")
    pauliop = options.get("pauliop", None)
    edge_color_control = options.pop("edge_color_control")
    style = options.pop("style")
    position = options.pop("position")
    options.pop("layout")
    title = options.pop("title")

    if pauliop is not None:
        if len(pauliop) != 1:
            raise ValueError("pauliop must be a single Pauli operator")
        options["label"] = "pauliop"

    if "labels" not in options:
        options["labels"] = process_labels(state, options)
    else:
        options.pop("pauliop")

    transp = options.pop("transparent")
    fig, ax = plt.subplots(figsize=options.pop("figsize"))

    if transp:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    if fix_wires is None and isinstance(state, MBQCircuit):
        if state.flow.name.lower() != "cflow":
            raise ValueError("Only cflow is supported at the moment")
        if state.flow is not None:
            fix_wires = []
            for inp in state.input_nodes:
                is_output = False
                wire = [inp]
                while not is_output:
                    out = state.flow(wire[-1])
                    wire.append(out)
                    if out in state.output_nodes:
                        is_output = True
                fix_wires.append(tuple(wire))

    if isinstance(state, GraphState):
        nx.draw(state, position, ax=ax, **options)

    elif isinstance(state, MBQCircuit):
        node_color = options.pop("node_color")
        if node_color is None:
            node_color = get_node_colors(state, style=style)
        node_colors = get_node_colors(state, style=style)
        options["node_color"] = [node_colors[node] for node in state.graph.nodes()]

        fixed_nodes = state.input_nodes + state.output_nodes
        position_xy = {}
        for indx, p in enumerate(state.input_nodes):
            position_xy[p] = (0, -1 * indx)

        separation = len(state.outputc) // len(state.output_nodes)
        if fix_wires is not None:
            for wire in fix_wires:
                if len(wire) + 2 > separation:
                    separation = len(wire) + 2
        for indx, p in enumerate(state.output_nodes):
            position_xy[p] = (2 * (separation) - 2, -1 * indx)

        if fix_wires is not None:
            x = [list(x) for x in fix_wires]

            fixed_nodes += sum(x, [])

            for indw, wire in enumerate(fix_wires):
                for indx, p in enumerate(wire):
                    if p != "*":
                        position_xy[p] = (2 * (indx + 1), -1 * indw)

        # remove all '*' from fixed_nodes
        fixed_nodes = [x for x in fixed_nodes if x != "*"]

        node_pos = nx.spring_layout(
            state.graph, pos=position_xy, fixed=fixed_nodes, k=1 / len(state.graph)
        )

        nx.draw(state.graph, ax=ax, pos=node_pos, **options)
        if state.flow is not None and show_flow:
            nx.draw(_graph_with_flow(state), pos=node_pos, ax=ax, **options)
        if show_controls:
            dashed_edges = []
            for node in state.controlled_nodes:
                for k in state.measurements[node].condition.cond_nodes:
                    dashed_edges.append((node, k))
            nx.draw_networkx_edges(
                state.graph,
                pos=node_pos,
                edge_color=edge_color_control,
                width=1.5,
                edgelist=dashed_edges,
                style="dashed",
            )

    return fig, ax


def _infer_pauliop_from_template(state):
    if isinstance(state, MBQCircuit):
        try:
            from mentpy.operators import PauliOp

            return PauliOp(_pauli_template_text(state, None, len(state.input_nodes)))
        except Exception:
            return None
    return None


def _draw_pauli_template(state, ax, title, pauliop, options):
    n_wires = len(state.input_nodes)
    _validate_pauli_template(state, n_wires)

    parity_node = 3 * n_wires
    angle_node = parity_node + 1
    pauli_text = _pauli_template_text(state, pauliop, n_wires)

    positions = {}
    labels = {}
    for q in range(n_wires):
        y = -q
        positions[3 * q] = (0, y)
        positions[3 * q + 1] = (1, y)
        positions[3 * q + 2] = (2, y)
        labels[3 * q] = f"q{q}"
        labels[3 * q + 1] = pauli_text[q]
        labels[3 * q + 2] = f"q{q}'"

    positions[parity_node] = (1, -n_wires)
    positions[angle_node] = (2, -n_wires)
    labels[parity_node] = "P"
    labels[angle_node] = r"$\theta$"

    draw_options = dict(options)
    draw_options["labels"] = labels
    draw_options.setdefault("edge_color", "#6c6c6c")
    draw_options.setdefault("edgecolors", "black")
    draw_options.setdefault("linewidths", 1.2)
    draw_options.setdefault("node_size", 1120)
    draw_options.setdefault("font_size", 12)
    draw_options.setdefault("width", 1.8)
    draw_options["node_color"] = draw_options.get(
        "node_color",
        [
            _pauli_template_node_color(
                node, state.input_nodes, state.output_nodes, parity_node, angle_node
            )
            for node in state.graph.nodes
        ],
    )

    nx.draw_networkx(state.graph, positions, ax=ax, **draw_options)
    if title is not None:
        ax.set_title(title)
    elif pauli_text:
        ax.set_title(f"MBQC Pauli-rotation layer: {pauli_text}")
    ax.axis("off")
    return ax.figure, ax


def _validate_pauli_template(state, n_wires):
    if n_wires == 0:
        raise ValueError("Pauli template layout requires at least one input wire.")
    expected_nodes = set(range(3 * n_wires + 2))
    if set(state.graph.nodes) != expected_nodes:
        raise ValueError(
            "layout='pauli' expects a circuit produced by templates.from_pauli."
        )
    if list(state.input_nodes) != [3 * q for q in range(n_wires)]:
        raise ValueError(
            "layout='pauli' expects input nodes from templates.from_pauli."
        )
    if list(state.output_nodes) != [3 * q + 2 for q in range(n_wires)]:
        raise ValueError(
            "layout='pauli' expects output nodes from templates.from_pauli."
        )


def _pauli_template_text(state, pauliop, n_wires):
    if pauliop is not None:
        return pauliop.txt

    parity_node = 3 * n_wires
    chars = []
    for q in range(n_wires):
        has_x = state.graph.has_edge(3 * q + 1, parity_node)
        has_z = state.graph.has_edge(3 * q, parity_node)
        if has_x and has_z:
            chars.append("Y")
        elif has_x:
            chars.append("X")
        elif has_z:
            chars.append("Z")
        else:
            chars.append("I")
    return "".join(chars)


def _pauli_template_node_color(
    node, input_nodes, output_nodes, parity_node, angle_node
):
    if node in input_nodes or node in output_nodes:
        return INPUT_NODE_COLOR
    if node == parity_node:
        return "#F0D784"
    if node == angle_node:
        return DEFAULT_NODE_COLOR
    return UNTRAINABLE_NODE_COLOR


def process_labels(state: Union[MBQCircuit, GraphState], options: dict):
    """Process and return the appropriate labels for the nodes based on the given options."""

    label_option = options.pop("label", "index")
    pauliop = options.pop("pauliop", None)

    if label_option in ("index", "indices"):
        return None  # No modification necessary
    elif label_option in ("plane", "planes"):
        labels = {
            node: ("" if state[node] is None else state[node].plane)
            for node in state.graph.nodes()
        }
        for node in state.controlled_nodes:
            labels[node] = "Ctrl"
        return labels
    elif label_option in ("arrow", "arrows"):
        plane2arrow = {
            "X": r"$\uparrow$",
            "Y": r"$\rightarrow$",
            "XY": r"$\nearrow$",
            "Z": r"$\cdot$",
            "XZ": r"$\nwarrow$",
            "YZ": r"$\nwarrow$",
            "XYZ": r"$\nwarrow \nearrow$",
            "": "",
        }
        labels = {
            node: plane2arrow[("" if state[node] is None else state[node].plane)]
            for node in state.graph.nodes()
        }
        for node in state.controlled_nodes:
            labels[node] = "Ctrl"
        return labels
    elif label_option in ("angles", "angle"):
        labels = {}
        for node in state.graph.nodes():
            if state.measurements[node] is not None:
                if state.measurements[node].angle is not None:
                    labels[node] = round(state.measurements[node].angle, 3)
                else:
                    labels[node] = r"$\theta$"
            else:
                labels[node] = ""
        return labels
    elif label_option == "pauliop":
        obj_nodes = (
            state.graph.nodes() if isinstance(state, MBQCircuit) else state.nodes()
        )
        labels = {node: pauliop.txt[node] for node in obj_nodes}
        return labels
    else:
        raise ValueError(
            f"label must be one of ['index', 'plane', 'arrow', 'angles', 'pauliop'], not {label_option}"
        )


def _graph_with_flow(state):
    """Return digraph with flow (but does not have all CZ edges!)"""
    if state.flow.name.lower() != "cflow":
        return state.graph
    g = state.graph
    dflow = nx.DiGraph()
    dflow.add_nodes_from(g.nodes())
    for node in state.outputc:
        next_nodes = state.flow(node)
        vs = []
        if isinstance(next_nodes, int):
            vs = [next_nodes]
        else:
            # check indx where the flow is 1
            vs = [i for i, x in enumerate(next_nodes) if x == 1]

        for v in vs:
            dflow.add_edge(node, v)
    return dflow
