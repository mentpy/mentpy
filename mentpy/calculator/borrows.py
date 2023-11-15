# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
import pennylane as qml


def fidelity(a, b):
    """Borrows the fidelity function from PennyLane."""
    return qml.math.fidelity(a, b)
