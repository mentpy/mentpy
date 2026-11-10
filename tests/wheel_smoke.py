"""Exercise the standard installation outside the repository checkout."""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import mentpy as mp
from mentpy import _rust

assert _rust.gf2_inverse([[1, 1], [0, 1]]) == [[1, 1], [0, 1]]
assert mp.utils.lie_algebra_completion(mp.PauliOp("X;Z")).txt == "X\nZ\nY\nI"
assert mp.utils.are_lc_equivalent(nx.empty_graph(3), nx.empty_graph(3))[0]
sim = mp.PatternSimulator(mp.templates.linear_cluster(2), backend="rust")
value, gradient = sim.expectation_and_grad([0.3], "Y")
np.testing.assert_allclose(value, np.sin(0.3), atol=1e-12)
np.testing.assert_allclose(gradient, [np.cos(0.3)], atol=1e-12)
assert sim.sample([0.3], shots=17, seed=4).shots == 17
pauli = mp.PatternSimulator(mp.templates.linear_cluster(2), backend="pauli")
np.testing.assert_allclose(
    pauli.expectation_and_grad([0.3], "Y")[1], [np.cos(0.3)], atol=1e-12
)
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'
pattern = mp.from_qasm(qasm)
assert len(pattern.quantum_output_nodes) == 1
np.testing.assert_allclose(jax.grad(lambda x: jnp.sin(x))(0.3), np.cos(0.3), atol=1e-6)
print(
    "Standard install: native flow, algebra, simulation, gradients, shots, QASM, and JAX passed."
)
