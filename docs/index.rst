.. MentPy documentation master file, created by
   sphinx-quickstart on Tue Sep  6 11:39:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. container:: mentpy-logo-light

   .. image:: _static/logo-light.png
      :alt: MentPy
      :align: center
      :width: 70%

.. container:: mentpy-logo-dark

   .. image:: _static/logo.png
      :alt: MentPy
      :align: center
      :width: 70%

Welcome to MentPy's documentation
=================================

MentPy is an open-source Python package for building, simulating, and training
measurement-based quantum circuits. Rust runs the flow solvers, tensor
contractions, gradients, and sampled quantum trajectories. A sparse Pauli
backend evaluates observables on causal-flow patterns, with CPU execution and
CUDA acceleration through cuPauliProp. MentPy also includes chemistry utilities,
OpenQASM import with ZX simplification, and a compatible JAX interface.


Features
--------

* Manipulation of graph states.
* Calculate causal, generalized, and Pauli flow with native GF(2) solvers.
* Simulate postselected patterns and sampled trajectories with the Rust backend.
* Differentiate tensor-network and Pauli expectation values in Rust.
* Propagate sparse Pauli observables on CPU or compatible NVIDIA GPUs.
* Use the existing NumPy and JAX backends for comparison and interoperability.
* Optimize measurement angles in MBQC ansatzes used for QML.
* Build molecular Hamiltonians and optimize ansatzes with explicit training loops.
* Import and export supported circuits with PennyLane, Cirq, Qiskit, and CUDA-Q.
* Import unitary OpenQASM 2 circuits as executable MBQC patterns.
* Simplify circuits with PyZX before compiling them to MBQC.
* Collect output bitstrings, counts, and finite-shot Pauli estimates.
* Create data and noisy data for training QML models.
* Compute Pauli Lie closure and graph-state local-Clifford equivalence in Rust.

Roadmap
-------
The native flow, contraction, gradient, trajectory, and sparse Pauli engines are
implemented. The next steps extend the patterns and resource states they can
represent:

* Extend Pauli circuit extraction beyond causal-flow X/Y/XY patterns.
* Add controlled approximation with reported error bounds and efficient joint sampling.
* Introduce MPS resources and local measurement instruments, starting with an AKLT wire.
* Improve contraction ordering for wider patterns and distribute the CUDA tensor build.
* Expand hardware-aware compilation and dynamic-circuit execution examples.

See :doc:`native` for supported execution paths, validation, and remaining limits.


Contributing
------------
If you would like to contribute to this project, please feel free to open an issue or pull request 😄.

Acknowledgements
----------------

Luis would like to thank his M.Sc. supervisors, Dr. Dmytro Bondarenko, Dr. Polina Feldmann, and Dr. Robert Raussendorf 
for their guidance during the development of this library.


Citation
--------

If you find MentPy useful in your research, please consider citing us 🙂

.. md-tab-set::
   .. md-tab-item:: BibTeX

      .. code-block:: latex

         @article{mantilla2025mbqml,
           title = {Measurement-based quantum machine learning},
           author = {Mantilla Calder\'on, Luis and Raussendorf, Robert and Feldmann, Polina and Bondarenko, Dmytro},
           journal = {Phys. Rev. A},
           volume = {113},
           number = {4},
           pages = {042421},
           year = {2026},
           month = {Apr},
           publisher = {American Physical Society},
           doi = {10.1103/2snk-m8c6},
           url = {https://link.aps.org/doi/10.1103/2snk-m8c6}
         }

   .. md-tab-item:: AIP

      .. code-block:: text

         L. Mantilla Calderón, R. Raussendorf, P. Feldmann, and D. Bondarenko, Measurement-based quantum machine learning, Phys. Rev. A 113, 042421 (2026). https://doi.org/10.1103/2snk-m8c6


   .. md-tab-item:: APA

      .. code-block:: text

         Mantilla Calderón, L., Raussendorf, R., Feldmann, P., & Bondarenko, D. (2026). Measurement-based quantum machine learning. Physical Review A, 113(4), 042421. https://doi.org/10.1103/2snk-m8c6

   .. md-tab-item:: MLA

      .. code-block:: text

         Mantilla Calderón, Luis, et al. "Measurement-based quantum machine learning." Physical Review A, vol. 113, no. 4, 2026, p. 042421. https://doi.org/10.1103/2snk-m8c6

.. toctree::
   :caption: Getting Started
   :hidden:

   getting-started
   native

.. toctree::
   :caption: Basic usage
   :hidden:

   basic-usage/measurements-in-qm.rst
   basic-usage/intro-to-graphstates.rst
   basic-usage/intro-to-mbqc.rst
   basic-usage/simulating-mbqc-circuits.rst

.. toctree::
   :caption: Tutorials
   :hidden:

   tutorials/index.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api
