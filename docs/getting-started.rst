Installation
============

MentPy requires Python 3.10 or above. Install it into your environment with:

.. code-block:: bash

   uv pip install mentpy

In a uv project, use ``uv add mentpy``. With pip, use
``python -m pip install --pre mentpy`` while releases are in alpha.

The standard install includes native flow algorithms, Rust tensor-network
simulation and reverse-mode autodiff, OpenQASM/ZX compilation, and the existing
JAX interface. No ``[jax]`` or ``[compiler]`` extra is needed.

Intel macOS installations use Numba 0.62, the last release providing binaries
for that platform. Its Python support ends at 3.13. Other platforms use the
current compatible dependency versions.

Integrations
------------

Install external toolkits when using their integration examples:

.. code-block:: bash

   uv pip install "mentpy[chem]"       # OpenFermion/PySCF molecular integrals
   uv pip install "mentpy[synthesis]"  # pygridsynth Clifford+T synthesis
   uv pip install "mentpy[interop]"    # Cirq, Qiskit, and OpenFermion helpers
   uv pip install "mentpy[cudaq]"      # CUDA-Q export and execution helpers

Native builds
-------------

Platform wheels contain the compiled Rust extension; using a wheel does not
require a Rust compiler. Building from source requires a Rust toolchain and a
C/C++ compiler. The native extension is required, so a failed build reports an
error instead of silently omitting native features.

.. code-block:: bash

   git clone https://github.com/mentpy/mentpy
   cd mentpy
   uv pip install -e '.[dev,chem]'

Native code lives in ``mentpy/_native``. Flow discovery uses packed GF(2)
operations and reuses one elimination for each generalized-flow layer. Tensor
contraction uses an iterative reverse-mode tape on CPU, with GEMM kernels for
larger products. The Python flow algorithms remain as independent references
for validation.

CUDA tensor contraction is available in source builds made with the Cargo
``cuda`` feature and a compatible CUDA toolkit. It uses Candle for tensor
operations and differentiation. CPU wheels do not contain this CUDA build.
For a source build, add ``features = ["cuda"]`` to the existing
``[[tool.setuptools-rust.ext-modules]]`` table in ``pyproject.toml`` before
installing. The CUDA toolkit must be on the compiler and library search paths.
Select a GPU explicitly with ``backend="rust", device="cuda:0"``; requesting
CUDA from a CPU-only build raises an error. Driver and toolkit availability are
separate from the standard Python dependencies.

Development checks
------------------

.. code-block:: bash

   python -m black --check .
   python -m pytest -q
   python -m sphinx -b html docs docs/_build/html
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test --lib

Run ``python -m benchmarks.flow`` for complete flow-solving timings and
``python -m benchmarks.native --jax`` to compare native and JAX gradients.
The simulator benchmark reports setup, first-call, and warmed timings
separately. Results depend on the graph, output size, and hardware.
