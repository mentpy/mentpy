Single qubit measurements
=========================

.. meta::
   :description: Measurements in quantum mechanics using python
   :keywords: mbqc, measurement-based quantum computation, quantum computing


Measurement-based quantum computation (MBQC) is a paradigm of quantum computing where the computation is performed by performing single-qubit measurements on a large entangled state known as a resource state. In MentPy, measurements are represented by the :obj:`mp.Ment` class.

.. ipython:: python

    m1 = mp.Ment('X')
    m1.matrix()

The :obj:`mp.Ment` class can be initialized with a string representing a pauli operator, or with an optional angle and a string representing the plane of rotation. The default angle is ``None`` which represents a trainable parameter. The default plane of rotation is ``XY``. 

.. ipython:: python

    m2 = mp.Ment(np.pi/2, 'XY')
    m2.matrix()

We can get the POVM elements of a measurement with the :meth:`get_povm` method

.. ipython:: python

    p0, p1 = m2.get_povm()
    print(p0)
    print(p1)

This object will be used as the basic building block for MBQC circuits.