Installation
=============

The :doc:`getting-started` guide is intended to assist users with installing the library.

Install using ``pip``
---------------------
The :obj:`mentpy` library requires Python 3.9 or above. It can be installed from 
`PyPI <https://pypi.org/project/mentpy/>`_ using ``pip``.

.. code-block:: bash

   python3 -m pip install mentpy

Install from Source
-------------------

To install from source, you can ``git clone`` the repository and then use ``pip`` to handle the installation. This method will also ensure that all necessary dependencies as specified in ``pyproject.toml`` are installed.

.. code-block:: bash

   git clone https://github.com/mentpy/mentpy
   cd mentpy
   python3 -m pip install -e .

Now, you can verify the installation by importing the :obj:`mentpy` package and checking its version.

.. ipython:: python

   import mentpy as mp
   mp.__version__

Development Installation
------------------------

Developers who wish to contribute to :obj:`mentpy` or use the development version can set up a development environment. This involves cloning the repository and installing the necessary dependencies, including those required for testing and documentation.

.. code-block:: bash

   git clone https://github.com/mentpy/mentpy
   cd mentpy
   python3 -m pip install -e '.[dev]'


This command installs :obj:`mentpy` in an "editable" mode and also installs additional development dependencies specified under `[dev-dependencies]` in `pyproject.toml`. Now, you are set up to make changes, run tests, and build documentation.

Testing and Development
-----------------------

Before submitting changes, you can run the test suite to ensure everything is functioning correctly:

.. code-block:: bash

   pytest

If you're adding new features or fixing any bugs, it's a good idea to include new tests that cover your changes.

Additionally, you can build the documentation locally to check for any errors and see how it looks before pushing changes:

.. code-block:: bash

   cd docs
   make html

This process generates HTML documentation in `docs/_build/html`, which you can open in a web browser to review.

Contributors are encouraged to follow the :doc:`contributing guidelines <CONTRIBUTING>` to submit their enhancements or bug fixes.
