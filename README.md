<p align="center">
  <img src="https://github.com/BestQuark/mentpy/blob/main/docs/_static/logo.png?raw=true" alt="MentPy: A Measurement-Based Quantum computing simulator." width="70%">
</p>

<div align="center">
    <a href="https://pypi.org/project/mentpy">
        <img src="https://img.shields.io/pypi/v/mentpy">
    </a>
    <a href="https://pypi.org/project/mentpy">
        <img src="https://img.shields.io/pypi/wheel/mentpy">
    </a>
    <a href="https://mentpy.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/mentpy/badge/?version=latest" alt="Documentation Status">
    </a>
    <a href="https://codecov.io/gh/BestQuark/mentpy">
        <img src="https://codecov.io/gh/BestQuark/mentpy/graph/badge.svg?token=TDWFQPW3KD">
    </a>
    <a href="https://twitter.com/mentpy">
        <img src="https://img.shields.io/twitter/follow/mentpy?label=mentpy&style=flat&logo=twitter">
    </a>
    <a href="https://discord.gg/HNA36hmEE5">
      <img alt="Discord" src="https://img.shields.io/discord/1158882999551676586?style=social&logo=discord">
    </a>
</div>

The `MentPy` library is an open-source software for simulations of 
measurement-based quantum computing circuits. Currently, this package is in its alpha version and many features are still in development.

## Installation

The `MentPy` library can be installed using `pip` with

```bash
pip install mentpy
```

or directly from the source code for the latest version with

```bash
pip install git+https://github.com/BestQuark/mentpy.git
```

## Usage
To simulate a measurement pattern, you can use the `mp.PatternSimulator`.
```python
import mentpy as mp

st = mp.templates.grid_cluster(2,4)
ps = mp.PatternSimulator(st)
output = ps(np.random.rand(len(st.outputc)))
```

For visualization of circuits, you can use the `mp.draw(st)` function

![image](https://user-images.githubusercontent.com/52287586/230715389-bf280971-c841-437d-8772-bf59557b0875.png)

To calculate the lie algebra of a model $G_\theta$, you can use the `mp.utils.calculate_lie_algebra` function

```python
lie_alg = mp.utils.calculate_lie_algebra(st)
len(lie_alg)
>> 16
```

## Documentation

The documentation for `MentPy` can be found <a href="https://mentpy.readthedocs.io/en/latest/" target="_blank">here</a>.

## Contributing

We welcome contributions to `MentPy`! Please see our [contributing guidelines](./CONTRIBUTING.md) for more information.

## License

`MentPy` is licensed under the [GNU General Public License v3.0](./LICENSE).

