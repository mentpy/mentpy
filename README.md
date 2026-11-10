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
tools, native tensor-network simulation and autodiff, sampled measurement
trajectories, Pauli observables and chemistry utilities, circuit interoperability,
and OpenQASM-to-MBQC compilation with ZX simplification.

## Installation

You can install the latest published release of `mentpy` from PyPI using:

```bash
uv pip install mentpy
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
output = ps(np.random.rand(len(st.trainable_nodes)))
```

Native simulation, autodiff, Pauli propagation, OpenQASM compilation, and JAX compatibility are
included in the standard install. For differentiable observable costs:

```python
sim = mp.PatternSimulator(st)
observable = mp.Observable("ZZ")
value, gradient = sim.expectation_and_grad(
    np.zeros(len(st.trainable_nodes)), observable
)
shots = sim.sample(np.zeros(len(st.trainable_nodes)), shots=1024, seed=7)
print(shots.counts)
```

Import a unitary OpenQASM 2 program with
`pattern = mp.from_qasm(qasm)`. Use `mp.compile(qasm).pattern` to apply PyZX
simplification first. The default Rust backend uses tensor contraction for
postselected patterns and keeps only active qubits for sampled trajectories
(`force0=False`). See the
[Grover tutorial](https://github.com/mentpy/mentpy/blob/main/docs/tutorials/grover-qasm.rst) for Qiskit and PennyLane examples,
exact probabilities, finite shots, Pauli measurements, and output ordering.

Use `backend="pauli"` for sparse output expectations on causal-flow X/Y/XY
patterns. On supported Linux systems, `device="cuda:0"` selects cuPauliProp;
its runtime libraries come with the standard install. A compatible NVIDIA
driver is required. Dense Pauli output methods reuse the Rust state simulator.
See the [native execution guide](docs/native.rst) for support limits and benchmarks.

For visualization of circuits, you can use the `mp.draw(st)` function

![image](https://user-images.githubusercontent.com/52287586/230715389-bf280971-c841-437d-8772-bf59557b0875.png)

To calculate the lie algebra of a model $G_\theta$, you can use the `mp.utils.calculate_lie_algebra` function

```python
lie_alg = mp.utils.calculate_lie_algebra(st)
len(lie_alg)
>> 16
```

## Documentation

The documentation for `mentpy` can be found <a href="https://docs.mentpy.com/en/latest/" target="_blank">here</a>.

## Contributing

We welcome contributions to `mentpy`! Please see our [contributing guidelines](https://github.com/mentpy/mentpy/blob/main/CONTRIBUTING.md) for more information.

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

