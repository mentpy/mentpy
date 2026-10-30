<p align="center">
  <img src="https://github.com/mentpy/mentpy/blob/main/docs/_static/logo.png?raw=true" alt="MentPy: A Measurement-Based Quantum computing simulator." width="70%">
</p>

<div align="center">
    <a href="https://pypi.org/project/mentpy">
        <img src="https://img.shields.io/pypi/v/mentpy">
    </a>
    <a href="https://pypi.org/project/mentpy">
        <img src="https://img.shields.io/pypi/wheel/mentpy">
    </a>
    <a href='https://docs.mentpy.com/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/mentpy/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/mentpy/mentpy" > 
        <img src="https://codecov.io/gh/mentpy/mentpy/graph/badge.svg?token=3EM0A3Q4MG"/> 
    </a>
    <a href="https://journals.aps.org/pra/abstract/10.1103/2snk-m8c6">
        <img alt="Paper" src="https://img.shields.io/badge/Paper-Physical%20Review%20A-blue">
    </a>
</div>

The `mentpy` library is open-source software for building, simulating, and
training measurement-based quantum computing circuits. It includes graph-state
tools, NumPy simulators, a JAX tensor-network backend with autodiff for
supported forced-outcome patterns, Pauli observables and chemistry utilities, circuit
interoperability, and OpenQASM-to-MBQC compilation with optional ZX simplification.

## Installation

You can install the stable release of `mentpy` from PyPI using:

```bash
pip install mentpy
```

For the latest, potentially unstable version, you can install directly from the source:

```bash
pip install git+https://github.com/mentpy/mentpy.git
```

If you're contributing to `mentpy` and need to install development dependencies, you can do so using:

```bash
git clone https://github.com/mentpy/mentpy.git
cd mentpy
pip install -e '.[dev]'
```

This command installs `mentpy` in editable mode with its additional development dependencies.

## Usage
To simulate a measurement pattern, you can use the `mp.PatternSimulator`.
```python
import numpy as np
import mentpy as mp

st = mp.templates.grid_cluster(2,4)
ps = mp.PatternSimulator(st)
output = ps(np.random.rand(len(st.outputc)))
```

For differentiable costs, install `mentpy[jax]` and use the tensor-network backend:

```python
import jax
import jax.numpy as jnp

sim = mp.PatternSimulator(st, backend="jax-tn")
observable = mp.Observable("ZZ")
cost = lambda angles: sim.expectation(angles, observable)
value_and_grad = jax.jit(jax.value_and_grad(cost))
value, gradient = value_and_grad(jnp.zeros(len(st.trainable_nodes)))
shots = sim.sample(np.zeros(len(st.trainable_nodes)), shots=1024, seed=7)
print(shots.counts)
```

Install `mentpy[compiler]` to import a unitary OpenQASM 2 program with
`pattern = mp.from_qasm(qasm)`. Use `mp.compile(qasm).pattern` to apply PyZX
simplification first. The `trajectory` backend executes causal-flow XY patterns
while keeping only the active frontier in memory. See the
[Grover tutorial](docs/tutorials/grover-qasm.rst) for Qiskit and PennyLane examples,
exact probabilities, finite shots, Pauli measurements, and output ordering.

For visualization of circuits, you can use the `mp.draw(st)` function

![image](https://user-images.githubusercontent.com/52287586/230715389-bf280971-c841-437d-8772-bf59557b0875.png)

To calculate the lie algebra of a model $G_\theta$, you can use the `mp.utils.calculate_lie_algebra` function

```python
lie_alg = mp.utils.calculate_lie_algebra(st)
len(lie_alg)
>> 16
```

## Documentation

The documentation for `mentpy` can be found <a href="https://mentpy.readthedocs.io/en/latest/" target="_blank">here</a>.

## Contributing

We welcome contributions to `mentpy`! Please see our [contributing guidelines](./CONTRIBUTING.md) for more information.

## How to cite

If you use `mentpy` in your research, please cite it as follows:

```
@article{mantilla2025mbqml,
  title = {Measurement-based quantum machine learning},
  author = {Mantilla Calder\'on, Luis and Raussendorf, Robert and Feldmann, Polina and Bondarenko, Dmytro},
  journal = {Phys. Rev. A},
  volume = {113},
  issue = {4},
  pages = {042421},
  numpages = {14},
  year = {2026},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/2snk-m8c6},
  url = {https://link.aps.org/doi/10.1103/2snk-m8c6}
}
```

