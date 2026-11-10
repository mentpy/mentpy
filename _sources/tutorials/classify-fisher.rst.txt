Fisher information classifier
=============================

Quantum Fisher information measures how sensitively a state responds to a
parameterized evolution. For a pure state evolving under a Hermitian
generator :math:`H`, it is
:math:`F_Q = 4(\langle H^2\rangle - \langle H\rangle^2)`.

This example uses the generator :math:`H=(ZI+IZ)/2`. The product state
:math:`|00\rangle` is an eigenstate and has zero Fisher information. The Bell
state :math:`(|00\rangle+|11\rangle)/\sqrt{2}` has Fisher information four.
Since :math:`H^2=(II+ZZ)/2`, we can evaluate the score with two Pauli
observables.

.. ipython:: python

    generator = mp.Observable({"ZI": 0.5, "IZ": 0.5})
    generator_squared = mp.Observable({"ZZ": 0.5}, constant=0.5)
    product_state = np.array([1., 0., 0., 0.])
    bell_state = np.array([1., 0., 0., 1.]) / np.sqrt(2)

Two length-three wires teleport the input to the output nodes when their
measurement angles are zero. Run each input through the same MBQC pattern,
evaluate the observables, and classify it with a threshold of two.

.. ipython:: python

    pattern = mp.templates.many_wires([3, 3])
    sim = mp.PatternSimulator(pattern)
    angles = np.zeros(len(pattern.trainable_nodes))

    for label, initial in [("Product", product_state), ("Bell", bell_state)]:
        state = sim.run(angles, input_state=initial)
        fisher = 4 * (generator_squared(state) - generator(state) ** 2)
        print(f"{label}: F_Q = {fisher:.1f}, class = {int(fisher > 2)}")
