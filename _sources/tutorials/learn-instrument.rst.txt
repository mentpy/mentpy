An adaptive quantum instrument
==============================

A quantum instrument returns both a classical outcome and a conditional
quantum state. In MBQC, a later measurement can use that classical outcome to
select its basis. Here, a three-node wire illustrates the difference between
sampling a branch and averaging over branches.

Measure a wire
--------------

The input is ``|0>``. Measure node 0 in X, then measure node 1 in XY. When the
first outcome is zero, use angle π. When it is one, use angle zero. The flow
corrections take care of the teleportation byproducts.

.. ipython:: python

    wire = mp.templates.linear_cluster(3)
    wire[0] = mp.Ment("X")
    wire[1] = mp.ControlMent(~wire[0].outcome, true_angle=np.pi)
    sim = mp.PatternSimulator(wire, [1, 0], force0=False, seed=7)
    state = sim.run()
    print(sim.outcomes)
    print(np.round(state, 3))

The first outcome is random. The final qubit is ``|1>`` when that outcome is
zero and ``|0>`` otherwise. To sample the instrument repeatedly, use
``sample()`` with the same simulator.

.. ipython:: python

    samples = sim.sample(shots=256, seed=7)
    print(samples.counts)

If you only need the average output state, export the instrument to a
PennyLane QNode. It applies the same conditional rotation and returns the
density matrix averaged over intermediate measurement outcomes.

.. ipython:: python

    instrument = mp.utils.to_pennylane(wire)
    print(np.round(instrument([], st=np.array([1, 0])), 3))
