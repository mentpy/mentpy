.. MentPy documentation master file, created by
   sphinx-quickstart on Tue Sep  6 11:39:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/logo.png
   :align: center
   :width: 70%

Welcome to MentPy's documentation
=================================

.. admonition:: Note
   :class: warning
   
   MentPy is in its alpha version and is under active development.

The :obj:`mentpy` library is an open-source Python package for creating and training quantum machine learning (QML) models 
in the measurement-based quantum computing (MBQC) framework. This library contains functions
to automatically calculate the causal flow or generalized flow of a graph and tools to analyze the 
expressivity of the MBQC ansatzes.


Features
--------

* Manipulation of graph states.
* Automatically calculate the causal flow or generalized flow of a graph.
* Simulate MBQC circuits.
* Optimize measurement angles in MBQC ansatzes used for QML.
* Create data and noisy data for training QML models.
* Determine the lie algebra of an MBQC ansatz.

Roadmap
-------
* Improve current simulators for MBQC circuits.
* Increase code coverage.
* Add autodiff support for MBQC circuits.
* Add support for more general MBQC states.
* Integrate with `pyzx` to optimize resources in MBQC circuits.


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

   tutorials/intro-to-mbqml.rst
   tutorials/intro-to-mbqml-parallel.rst
   tutorials/classify-classical-data.rst
   tutorials/classify-fisher.rst
   tutorials/hea-discrete.rst
   tutorials/learn-instrument.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api
