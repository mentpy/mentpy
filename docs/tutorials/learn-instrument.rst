Learning a quantum instrument
=============================

.. meta::
    :description: Learning a quantum instrument for teleporation
    :keywords: quantum, quantum machine learning, measurement-based quantum computing

.. admonition:: Note
   :class: warning
   
   MentPy is under active development. This tutorial might break in future versions.


In this tutorial we use the object :class:`mp.ControlMent` to learn a quantum instrument for teleportation. In general, a quantum instrument is a map :math:`\mathcal{I}: \operatorname{End}(\mathcal{H}_1) \rightarrow \operatorname{End}(\mathcal{H}_2) \otimes \operatorname{End}(\mathbb{C}^{|X|})` that measures a state :math:`\rho` and stores the measurement outcome,

.. math::
    \mathcal{I}(\rho) = \sum_{x\in X} E_x (\rho) \otimes \ket{x}\bra{x},
    
where :math:`\left\{E_x\right\}_{x\in X}` is a collection of completely positive maps such that :math:`\operatorname{Tr}\left(\sum_{x\in X} E_x(\rho)\right) = \operatorname{Tr}(\rho)`. 


.. ipython:: python
    :okwarning:

    import numpy as np
    import mentpy as mp

    list_of_wires = [5,5]
    gs = mp.templates.many_wires(list_of_wires)
    gs.add_edge(1,5)
    gs.add_edge(1,7)

    mgs = mp.merge(gs, gs, along = [(4, 5)])

    mgs = mp.merge(mgs, mp.templates.linear_cluster(5), along=[(8,0)])
    mgs[3] = mp.Ment('X')
    mgs[7] = mp.Ment('X')
    mgs[11] = mp.Ment('X')
    mgs[16] = mp.Ment('X')
    mgs[12] = mp.Ment("Z")
    mgs[17] = mp.Ment("Z")
    mgs[18] = mp.ControlMent(mgs[17].outcome + 1)
    mgs[19] = mp.ControlMent(mgs[17].outcome + 1)
    mgs[20] = mp.ControlMent(mgs[12].outcome + 1)
    mgs[21] = mp.ControlMent(mgs[12].outcome + 1)
    
    @savefig teleportation_ansatz.png width=1000px
    mp.mbqc.view.draw_old(mgs, label='indices', edge_color_control='black', figsize = (12,4), fix_wires=[(8, '*', '*', '*','*', 9,10,11,12), (0,1,2,3, 13,14,15,16,17), (4,5,6,7, "*", "*","*", "*", 18, 19,20,21,22)])

Usually we do not have access to the analytical solution of a learning problem, but in this case, it is possible to find it as it is a small quantum system. In particular, to get an optimal cost value we define the following measurement pattern:

.. ipython:: python

    input_state_random = mp.utils.generate_haar_random_states(1, 1)
    state_zero = np.array([1, 0])
    state_zero_product = np.kron(state_zero, state_zero)
    input_state = np.kron(state_zero_product, input_state_random)

    wires = [[0,1,2,3,13,14,15,16,17], [4,5,6,7,18,19,20,21,22],[8,9,10,11,12]]
    schedule = [0,4,8,5,1,6,2,3,7,13,14,9,15,10,11,16,17,18,19,12,20,21, 22]

    psK = mp.PatternSimulator(mgs, backend='numpy-dm', window_size=5, dev_mode = True, wires = wires, schedule = schedule)

    psK.reset(input_state=input_state)
    angles = np.zeros(len(mgs.trainable_nodes))

    angles[4] = np.pi / 2
    angles[10] = np.pi / 2
    angles[12] = np.pi
    angles[13] = np.pi
    angles[14] = 0
    angles[15] = np.pi

    quantum_state = psK(angles)
    outcomes = ((psK.outcomes[12] + 1) % 2, (psK.outcomes[17] + 1) % 2)
    fidelity = mp.calculator.fidelity(quantum_state, np.outer(input_state_random, input_state_random.conj()))

    print(quantum_state, outcomes, fidelity)

.. admonition:: Code for plotting the exact solution
    :class: codeblock
    :collapsible:

    .. ipython:: python

        angle_to_text = {
            0: '0',
            np.pi / 2: r'$\pi/2$',
            np.pi: r'$\pi$',
            3 * np.pi / 2: r'$3\pi/2$'
        }

        labels = {node: angle_to_text[angle] for node, angle in zip(mgs.trainable_nodes, angles)}

        for node in mgs.nodes:
            if node not in mgs.trainable_nodes and node in mgs.outputc:
                labels[node] = '0'
            elif node not in mgs.quantum_output_nodes and node in mgs.output_nodes:
                labels[node] = 'Z'

        @savefig teleport_exact_solution.png width=1000px
        mp.draw(
            mgs,
            label='angles',
            labels=labels,
            edge_color_control='black',
            figsize=(12, 4),
            fix_wires=[(8, '*', '*', '*', '*', 9, 10, 11, 12), (0, 1, 2, 3, 13, 14, 15, 16, 17), (4, 5, 6, 7, "*", "*", "*", "*", 18, 19, 20, 21, 22)]
        )

We can now define a loss function, a callback, and train the ansatz to get a solution close to the analytical one.

.. code-block:: python

    def loss(output, target, apply_hadamard = False):
        avg_fidelity = 0
        for i in range(len(target)):
            sty = target[i]
            sty = np.outer(sty, np.conj(sty))
            
            out = output[i]

            if apply_hadamard:
                out = mp.gates.HGate @ out @ mp.gates.HGate

            avg_fidelity += mp.calculator.fidelity(out,sty)
    
        ans = 1 - (avg_fidelity/len(target))
        return ans

    st0 = np.array([1,0])
    def prediction(thetas, statesx):
        output = []
        for i in range(len(statesx)):
            input_kron = np.kron(st0, st0)
            input_kron = np.kron(input_kron, statesx[i])
            psK.reset(input_state=input_kron)
            statek = psK(thetas)
            output.append(statek)
        return output

    def cost(thetas, statesx, statesy):
        outputs = prediction(thetas, statesx)
        return loss(outputs, statesy)

    cost_train = []
    cost_test = []
    def create_callback(**kwargs):
        global cost_train
        global cost_test
        cost_train = []
        cost_test = []
        calliter = 0
        def callback(x,calliter):
            global cost_train
            global cost_test
            ctrain = cost(x, x_train, y_train)
            cost_train.append(ctrain)
            ctest = cost(x, x_test, y_test)
            cost_test.append(ctest)
            if calliter % 10 == 0:
                print(f"iter: {calliter}, train: {ctrain}, test: {ctest}")
        return callback

    MAX_NUM_STEPS = 180
    runs_train = {}
    runs_test = {}
    theta_ops = {}
    for i in range(0,10):
        (x_train, y_train), (x_test, y_test) = mp.utils.generate_random_dataset(np.eye(2), 50, test_size = 0.3)
        theta = np.random.rand(len(mgs.trainable_nodes))
        print("value pre-training: ", cost(theta, x_test, y_test))
        opt = mp.optimizers.AdamOpt(step_size=0.1)
        theta = opt.optimize(lambda x: cost(x, x_train, y_train), theta, callback = create_callback(), num_iters=MAX_NUM_STEPS)
        post_cost = cost(theta, x_test, y_test)
        runs_train[i] = cost_train
        runs_test[i] = cost_test
        theta_ops[i] = theta

.. admonition:: Code for plotting learning curve
    :class: codeblock
    :collapsible:
        
    .. code-block:: python
        
        plt.style.use('default')

        MAX_NUM_RUNS = 10
        MAX_NUM_STEPS = 180

        runs_train_array = np.array(list(runs_train.values()))[:, :MAX_NUM_STEPS]
        runs_test_array = np.array(list(runs_test.values()))[:, :MAX_NUM_STEPS]

        train_means = np.mean(runs_train_array, axis=0)
        train_stds = np.std(runs_train_array, axis=0)
        test_means = np.mean(runs_test_array, axis=0)
        test_stds = np.std(runs_test_array, axis=0)

        train_lower = np.maximum(train_means - train_stds, 0)  
        train_upper = train_means + train_stds
        test_lower = np.maximum(test_means - test_stds, 0)
        test_upper = test_means + test_stds

        plt.style.use('default')
        fig, ax = plt.subplots()
        ax.plot(train_means, label='Train cost mean', color='blue')
        ax.fill_between(range(MAX_NUM_STEPS), train_lower, train_upper, alpha=0.1, color='blue')
        ax.plot(test_means, label='Test cost mean', linestyle='--', color='green')
        ax.fill_between(range(MAX_NUM_STEPS), test_lower, test_upper, alpha=0.1, color='green')

        ax.legend(loc='lower left')
        plt.xlabel('Steps')
        plt.ylabel('Cost')
        plt.title("Learning curve for a quantum instrument")
        plt.savefig('TeleportLearningCurve.png', dpi=700)
        plt.show()

Finally, we can plot the learning curves for this example.
