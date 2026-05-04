.. MentPy documentation master file, created by
   sphinx-quickstart on Tue Sep  6 11:39:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/logo.png
   :align: center
   :width: 70%

Welcome to MentPy's documentation
=================================

The :obj:`mentpy` library is an open-source Python package for creating and training quantum machine learning (QML) models 
in the measurement-based quantum computing (MBQC) framework. This library contains functions
to automatically calculate the causal flow or generalized flow of a graph and tools to analyze the 
expressivity of the MBQC ansatzes. MentPy now includes production-usable MBQC
construction tools, NumPy simulators, a JAX tensor-network backend for
low-treewidth forced-outcome patterns, chemistry/VQE helpers, circuit
interoperability, and optional compiler integrations. Public APIs may still
evolve as the project grows.


Features
--------

* Manipulation of graph states.
* Automatically calculate the causal flow or generalized flow of a graph.
* Simulate MBQC circuits with NumPy and JAX tensor-network backends.
* Evaluate tensor-network observable costs with JAX autodiff.
* Optimize measurement angles in MBQC ansatzes used for QML.
* Build chemistry Hamiltonians and MBQC UCCSD-style VQE ansatzes.
* Import and export supported circuits with PennyLane, Cirq, Qiskit, and CUDA-Q.
* Compile QASM/PyZX circuits through optional ZX-calculus reductions.
* Create data and noisy data for training QML models.
* Determine the lie algebra of an MBQC ansatz.

Roadmap
-------
* Stabilize compiler passes that convert between gate circuits, ZX diagrams, and MBQC resources.
* Benchmark and optimize tensor-network contractions for larger low-treewidth patterns.
* Broaden support for general MBQC resource states and non-forced measurement workflows.
* Expand hardware-aware export paths and dynamic-circuit execution examples.
* Continue increasing coverage around simulators, gradients, interop, chemistry, and compiler utilities.


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

         @software{mantilla2023mentpy,
            title = {{MentPy: A Python package for parametrized MBQC circuits}},
            author = {Mantilla Calderón, Luis},
            year = {2023},
            url = {https://github.com/mentpy/mentpy},
         }

   .. md-tab-item:: AIP

      .. code-block:: text

         L. Mantilla Calderón, MentPy: A Python package for parametrized MBQC circuits, (2023). https://github.com/mentpy/mentpy
     

   .. md-tab-item:: APA

      .. code-block:: text

         Mantilla Calderón, L. (2023). MentPy: A Python package for parametrized MBQC circuits. Retrieved from https://github.com/mentpy/mentpy
   
   .. md-tab-item:: MLA

      .. code-block:: text

         Mantilla Calderón, Luis. MentPy: A Python package for parametrized MBQC circuits. 2023. Web. https://github.com/mentpy/mentpy



.. toctree::
   :caption: Getting Started
   :hidden:

   getting-started

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
