An introduction to MB-QML
=========================

.. meta::
   :description: An introduction to measurement-based quantum machine learning
   :keywords: mb-qml, measurement-based quantum machine learning, quantum machine learning, mbqc

**Author(s):** `Luis Mantilla <https://x.com/realmantilla>`_

Quantum machine learning (QML) is a field that studies how to use parametrized quantum circuits to 
learn to identify patterns in quantum data. In measurement-based quantum machine learning (MB-QML) 
[#thesis]_, we use a MBQC circuit with parametrized measurement angles to solve QML problems. 

In :mod:`mentpy`, MB-QML models are defined using the :class:`MBQCircuit` class. We can define a model from scratch
or use one of the templates provided in :mod:`mentpy.templates`. Here, we use the MuTA template with two 
input qubits, and fix two of the parameters to be fixed (qubits 3 and 8).

.. ipython:: python

    import numpy as np
    import mentpy as mp

    gs = mp.templates.muta(2,1, one_column=True)
    gs[3] = mp.Ment('X')
    gs[8] = mp.Ment('X')
    ps = mp.PatternSimulator(gs)
    @savefig muta_mbqml.png width=1000px
    mp.draw(gs)

To optimize the parameters of the model, we need to define a loss function. Here, we will use the 
average infidelity between the target states and the output states of the model.

.. ipython:: python

    def loss(output, target):
        avg_fidelity = 0
        for sty, out in zip(target, output):
            sty = mp.calculator.pure2density(sty)
            avg_fidelity += 1-mp.calculator.fidelity(sty, out)
        ans = (avg_fidelity/len(target))
        return ans

    def prediction_single_state(thetas, st):
        ps.reset(input_state=st)
        st = ps(thetas)
        return st

    def prediction(thetas, statesx):
        output = [prediction_single_state(thetas, st) for st in statesx]
        return output

    def cost(thetas, statesx, statesy):
        outputs = prediction(thetas, statesx)
        return loss(outputs, statesy)

Be aware that the loss function we are using in this example is a global operation, which can induce barren plateaus. We will ignore this issue for now. Having defined a model and a loss function, 
we can now use some data to train our model. We will use the :func:`generate_random_dataset` function 
to generate a random dataset of states :math:`\left\{(\rho_i, \sigma_i)_i \right\}_i^{N}`
where the input and target states are related by a given unitary :math:`\sigma_i = U \rho_i U^\dagger`.

.. code-block:: python

    runs_train = {}
    runs_test = {}

    NUM_STEPS = 100
    NUM_RUNS = 20

    for i in range(NUM_RUNS):
        gate2learn = np.kron(mp.gates.random_su(1), np.eye(2))
        # Replace with the following line to learn an IsingXX(π/2) gate
        # gate2learn = mp.gates.ising_xx(np.pi/2)
        (x_train, y_train), (x_test, y_test) = mp.utils.generate_random_dataset(gate2learn, 10, test_size = 0.3)

        cost_train, cost_test = [], []

        def callback(params, iter):
            cost_train.append(cost(params, x_train, y_train))
            cost_test.append(cost(params, x_test, y_test))
        
        theta = np.random.rand(len(gs.trainable_nodes))
        opt = mp.optimizers.AdamOpt(step_size=0.08)
        theta = opt.optimize(lambda params: cost(params, x_train, y_train), theta, num_iters=NUM_STEPS, callback=callback)
        post_cost = cost(theta, x_test, y_test)

        runs_train[i] = cost_train
        runs_test[i] = cost_test

.. admonition:: Code for plotting learning curve
    :class: codeblock
    :collapsible:

    If you do not have seaborn installed, you can either install it by running `pip install --upgrade seaborn` or comment out the seaborn-style lines.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("white")

        runs_train_array = np.array(list(runs_train.values()))
        runs_test_array = np.array(list(runs_test.values()))

        train_means = np.mean(runs_train_array, axis=0)
        train_stds = np.std(runs_train_array, axis=0)
        test_means = np.mean(runs_test_array, axis=0)
        test_stds = np.std(runs_test_array, axis=0)

        train_lower = np.maximum(train_means - train_stds, 0) 
        train_upper = train_means + train_stds
        test_lower = np.maximum(test_means - test_stds, 0)
        test_upper = test_means + test_stds

        fig, ax = plt.subplots()
        ax.plot(train_means, label='Train cost mean', color='blue')
        ax.fill_between(range(len(train_means)), train_lower, train_upper, alpha=0.1, color='blue')
        ax.plot(test_means, label='Test cost mean', linestyle='--', color='green')
        ax.fill_between(range(len(test_means)), test_lower, test_upper, alpha=0.1, color='green')

        ax.legend(fontsize=16)
        ax.set_xlabel('Steps', fontsize=16)
        ax.set_ylabel('Cost', fontsize=16)
        ax.set_title(r"Random local unitary, $U_{Haar} \otimes I$", fontsize=18)
        # Title for training an IsingXX(π/2) gate
        # ax.set_title(r"IsingXX($\pi/2$)", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

Finally, we can average over the runs and plot the results! In our next tutorial, we wil see how to parallelize the training process and study the robustness of the model to noise.

References
----------

.. [#thesis] Mantilla Calderón, L. C. (2023). Measurement-based quantum machine learning (T). University of British Columbia. 