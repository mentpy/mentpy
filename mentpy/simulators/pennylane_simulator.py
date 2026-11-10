# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""PennyLane simulator for MBQC circuits."""

from functools import reduce
from operator import and_

import pennylane as qml

from mentpy.mbqc.measurement_angles import MeasurementAngleResolver
from .numpy_simulator import NumpySimulator as PennylaneSimulator

__all__ = ["PennylaneSimulator"]


def _prepare_state(state, wires):
    """Prepare an input state across PennyLane versions."""
    if len(wires) == 0:
        return
    if hasattr(qml, "StatePrep"):
        qml.StatePrep(state, wires=wires)
    else:
        qml.QubitStateVector(state, wires=wires)


def mbqcircuit_to_circuit(gsc, device=None, circuit_noise=None, *args, **kwargs):
    """Converts a MBQCircuit to a PennyLane circuit."""
    gr = gsc.graph
    resolver = MeasurementAngleResolver(gsc)
    resolver.validate_dependencies()
    corrections = resolver._flow_corrections()
    device = device or ("default.mixed" if circuit_noise else "default.qubit")
    dev = qml.device(device, wires=list(gr))
    n_angles = len(gsc.trainable_nodes)
    positions = {node: i for i, node in enumerate(resolver.measurement_nodes)}

    @qml.qnode(dev)
    def circuit(param, output="density", st=None):
        if output != "density":
            raise ValueError(f"Unsupported PennyLane output {output!r}.")
        if len(param) != n_angles:
            if len(param) != len(positions):
                raise ValueError(f"Expected {n_angles} trainable measurement angles.")
            # Older exports accepted one entry per measurement.
            param = [param[positions[node]] for node in gsc.trainable_nodes]
        if st is None:
            for node in gsc.input_nodes:
                qml.Hadamard(node)
        else:
            _prepare_state(st, wires=gsc.input_nodes)

        for j in gsc.inputc:
            qml.Hadamard(j)
        for i, j in gr.edges():
            qml.CZ(wires=[i, j])

        if circuit_noise is not None:
            for i in gr:
                if circuit_noise == "depolarizing":
                    qml.DepolarizingChannel(wires=i, *args, **kwargs)
                elif circuit_noise == "amplitude_damping":
                    qml.AmplitudeDamping(wires=i, *args, **kwargs)
                elif circuit_noise == "phase_damping":
                    qml.PhaseDamping(wires=i, *args, **kwargs)
                elif circuit_noise == "phase_flip":
                    qml.PhaseFlip(wires=i, *args, **kwargs)
                elif circuit_noise == "generalized_amplitude_damping":
                    qml.GeneralizedAmplitudeDamping(wires=i, *args, **kwargs)
                else:
                    raise ValueError(f"Unrecognized circuit noise: {circuit_noise}")

        outcomes = {}
        for node in resolver.measurement_nodes:
            for angle, controls in resolver._xy_rotations(node, param):
                if controls:
                    condition = reduce(
                        and_, (outcomes[control] for control in controls)
                    )
                    qml.cond(condition, qml.RZ)(angle, wires=node)
                else:
                    qml.RZ(angle, wires=node)
            qml.Hadamard(wires=node)
            outcomes[node] = qml.measure(node)
            for target, pauli in corrections.get(node, ()):
                gate = qml.PauliX if pauli == "x" else qml.PauliZ
                qml.cond(outcomes[node], gate)(wires=target)

        return qml.density_matrix(gsc.quantum_output_nodes)

    return circuit
