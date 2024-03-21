Parallelizing MB-QML protocols
==============================

.. admonition:: Note
   :class: warning
   
   MentPy is under active development. This tutorial might break in future versions.

In this tutorial, we will see how to parallelize the MB-QML protocols in :mod:`mentpy`. 
Similar to the previous tutorial, we first need a MB-QML model to work with.

.. ipython:: python

   gs = mp.templates.muta(2,1, one_column=True)
   gs[3] = mp.Ment('X')
   gs[8] = mp.Ment('X')
   ps = mp.PatternSimulator(gs)
   @savefig muta_mbqml.png width=1000px
   mp.draw(gs)


Then, when we define a loss function, we can use the :mod:`pathos` package to parallelize the computation
of the infidelity between the target states and the output states.

.. ipython:: python

    from pathos.multiprocessing import ProcessingPool as Pool

    def loss(output, target):
        avg_fidelity = 0
        for sty, out in zip(target, output):
            sty = mp.calculator.pure2density(sty)
            avg_fidelity += 1-mp.calculator.fidelity(sty, out)
        ans = (avg_fidelity/len(target))
        return ans

    def prediction_single_state(thetas, st):
        ps.reset(input_state=st)
        statek = ps(thetas)
        return statek

    def prediction(thetas, statesx):
        thetas = np.copy(thetas)
        pool = Pool()
        output = pool.map(prediction_single_state, [thetas]*len(statesx), statesx)
        return output

    def cost(thetas, statesx, statesy):
        outputs = prediction(thetas, statesx)
        return loss(outputs, statesy)


Then, we define our training loop and iterate over a noise range to see how the infidelity changes with the noise strength.

.. code-block::

    gate2learn = mp.gates.ising_xx(np.pi/2)
    noise_type = 'brownian'
    # noise_type = 'bitflip'

    STEPS_NOISE = 10
    test_rounds = {}
    train_rounds = {}
    for k in range(STEPS_NOISE):
        runs_train = []
        runs_test = []
        NUM_STEPS = 200
        NUM_RUNS = 60
        for i in range(NUM_RUNS):
            (x_train, y_train), (x_test, y_test) = mp.utils.generate_random_dataset_noisy(gate2learn, 20, noise_level=0.05*k, noise_type=noise_type, test_size = 0.5)
                
            theta = np.random.rand(len(gs.trainable_nodes))
            opt = mp.optimizers.AdamOpt(step_size=0.08)
            theta = opt.optimize(lambda params: cost(params, x_train, y_train), theta, num_iters=NUM_STEPS)
            post_cost_test = cost(theta, x_test, y_test)
            post_cost_train = cost(theta, x_train, y_train)

            runs_train.append(post_cost_train)
            runs_test.append(post_cost_test)
            
        print("Finished round ", k)
        train_rounds[k] = runs_train
        test_rounds[k] = runs_test
        np.save(f'train_{k}_np.npy', runs_train)
        np.save(f'test_{k}_np.npy', runs_test)


This will significantly speed up the computation of the loss function. Finally, we can plot the learning curve.

.. admonition:: Code for plotting learning curve
    :class: codeblock
    :collapsible:

    If you do not have seaborn installed, you can either install it by running `pip install --upgrade seaborn` or comment out the seaborn-style lines.

    .. code-block:: python

        fig, ax = plt.subplots()
        means_train, means_test, sem_train, sem_test = [], [], [], []

        for indx, noise_lvl in enumerate(np.arange(STEPS_NOISE)*0.05):
            runs_train = np.load(f'train_{indx}_np.npy')
            runs_test = np.load(f'test_{indx}_np.npy')
            means_train.append(1-np.mean(runs_train))
            means_test.append(1-np.mean(runs_test))
            sem_train.append(np.std(runs_train, ddof=1) / np.sqrt(len(runs_train)))
            sem_test.append(np.std(runs_test, ddof=1) / np.sqrt(len(runs_test)))

        sem_train = np.array(sem_train)
        sem_train[sem_train < 0] = 0
        sem_train[sem_train > 1] = 1

        sem_test = np.array(sem_test)
        sem_test[sem_test < 0] = 0
        sem_test[sem_test > 1] = 1

        ax.errorbar(np.arange(STEPS_NOISE)*0.05, means_train, yerr=np.sqrt(sem_train), label='Train cost', linestyle='-', color='blue', capsize=2)
        ax.errorbar(np.arange(STEPS_NOISE)*0.05, means_test, yerr=np.sqrt(sem_test), label='Test cost', linestyle='--', color='green', capsize=2)

        ax.legend(fontsize=16)

        ax.set_xlabel('Noise strength', fontsize=16)
        ax.set_ylabel('Fidelity', fontsize=16)
        ax.set_title(r'$\operatorname{IsingXX}(\pi/2)$ (Brownian noise)', fontsize=18)
        # ax.set_title(r'$\operatorname{IsingXX}(\pi/2)$ (Bitflip noise)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)  
        plt.tight_layout()
        plt.savefig('isingxx_noise.png', dpi=500)

