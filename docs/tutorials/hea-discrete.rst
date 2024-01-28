Discrete optimization for a hardware efficient ansatz
=====================================================

.. meta::
    :description: Discrete optimization for a hardware efficient ansatz
    :keywords: quantum, quantum machine learning, measurement-based quantum computing

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction


**Author(s):**  `Luis Mantilla <https://x.com/realmantilla>`_

In this tutorial, we will explore how to optimize the angles of a MB-QML circuit using Deep Q-Learning and a greedy-search algorithm.

First, we need to set up our environment. We will use :mod:`gymnasium` for the environment, :mod:`stable_baselines3` for the Deep Q-Learning algorithm.

.. code-block:: python

    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    from stable_baselines3 import DQN
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    from scipy.linalg import sqrtm
    import mentpy as mp
    import matplotlib.pyplot as plt

    model_class = DQN  # Can also work with SAC, DDPG, and TD3


Quantum Circuit and loss function
---------------------------------

Let's define our quantum circuit and prepare the training and testing data.

.. code-block:: python

    gs = mp.templates.muta(2, 1, one_column=True)
    mp.draw(gs)

    ps = mp.PatternSimulator(gs, backend='numpy-dm', window_size=5)
    mygate = mp.gates.ising_xx(np.pi/4)
    mygate = np.kron(mp.gates.TGate, np.eye(2)) @ mygate
    (X_train, y_train), (X_test, y_test) = mp.utils.random_train_test_states_unitary(mygate, 30)


We define our loss and prediction functions to evaluate the performance of our quantum circuit.

.. code-block:: python

    def loss(output, target):
        avg_fidelity = 0
        for sty, out in zip(target, output):
            sty = np.outer(sty, np.conj(sty.T))
            avg_fidelity += mp.calculator.fidelity(sty, out) 
        return 1 - (avg_fidelity / len(target))

    def prediction(thetas, statesx):
        output = []
        thetas = np.copy(thetas)
        for st in statesx:
            ps.reset(input_state=st)
            statek = ps(thetas)
            output.append(statek)
        return output

    def cost(thetas, statesx, statesy):
        outputs = prediction(thetas, statesx)
        return loss(outputs, statesy)


Optimizing with Deep Q-Learning
-------------------------------

We create a custom Gym environment for our quantum system. This environment will interact with the reinforcement learning agent.

.. code-block:: python

    class QuantumGymEnvironment(gym.Env):
        def __init__(self, X_train, y_train, X_test, y_test, max_iters=300, threshold=0.99, eval=False):
            super(QuantumGymEnvironment, self).__init__()

            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.max_iters = max_iters
            self.threshold = threshold
            self.iter_to_node = [i for i in gs.measurement_order if i in gs.trainable_nodes]
            self.best_solution = None
            self.angles = [-np.pi/2, -np.pi/4, 0]
            self.num_nodes = len(gs.trainable_nodes)
            self.TOTAL_ITERS = 0

            self.action_space = spaces.Discrete(len(self.angles)) 
            low_bounds = np.full(self.num_nodes, -np.pi/2)
            low_bounds = np.append(low_bounds, 0)  

            high_bounds = np.full(self.num_nodes, 0)
            high_bounds = np.append(high_bounds, self.num_nodes) 

            self.observation_space = spaces.Box(low=low_bounds, high=high_bounds)
            self.theta = None
            self.iter = 0
            self.cost_calls = 0

        def step(self, action):
            global training_progress
            curr_ind = self.iter % len(self.iter_to_node)
            self.theta[curr_ind] = self.angles[action]
            self.iter += 1
            self.TOTAL_ITERS += 1

            done = self.iter >= self.max_iters
            outputs = prediction(self.theta, self.X_train)
            loss_value = loss(outputs, self.y_train)
            fidelity = 1 - loss_value

            reward = fidelity
            outputs_test = prediction(self.theta, self.X_test)
            loss_value_test = loss(outputs_test, self.y_test)
            fidelity_test = 1 - loss_value_test

            if self.best_solution is None or fidelity > self.best_solution['fidelity']:
                self.best_solution = {'theta': self.theta.copy(), 'fidelity': fidelity, 'fid_test': fidelity_test}

            training_progress['fidelity'].append(self.best_solution['fidelity'])
            training_progress['fidelity_test'].append(self.best_solution['fid_test'])
            training_progress['steps'].append(self.TOTAL_ITERS)

            observation, info = self._get_obs(), self._get_info()
            return observation, reward, done, False, info

        def _get_obs(self):
            return np.append(self.theta.copy(), self.iter % len(self.iter_to_node))

        def _get_info(self):
            return {"cost_calls": self.cost_calls}

        def reset(self, seed=None, options=None):
            self.theta = np.random.choice([0, -np.pi/2], self.num_nodes)
            observation = self._get_obs()
            info = self._get_info()
            self.iter = 0
            self.cost_calls = 0
            return observation, info

        def render(self, mode='human'):
            pass

        def close(self):
            pass


We train our model using the Deep Q-Learning algorithm provided by :mod:`stable_baselines3`.

.. code-block:: python

    training_progress = {'fidelity': [], 'steps': [], 'fidelity_test':[]}
    env = QuantumGymEnvironment(X_train, y_train, X_test, y_test, max_iters=200)
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=3*(3**8))


We can visualize the training progress by plotting the fidelity of the quantum state over time.

.. admonition:: Code for plotting learning curve
    :class: codeblock
    :collapsible:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.colors as mcolors

        plt.plot(training_progress['steps'][1:3**8], 1 - np.array(training_progress['fidelity'][1:3**8]), linestyle="-", color='r', marker='o', markevery=0.1, label='Train', alpha=0.5)
        plt.plot(training_progress['steps'][1:3**8], 1 - np.array(training_progress['fidelity_test'][1:3**8]), linestyle="--", color='r', label='Test', alpha=0.5)

        plt.plot([i -  training_progress2['steps'][0] for i in training_progress2['steps']], 1 - np.array(training_progress2['fidelity']), linestyle="-", color='b', marker='o', markevery=0.1, label='Train', alpha=0.5)
        plt.plot(training_progress2['steps'], 1 - np.array(training_progress2['fidelity_test']), linestyle="--", color='b', label='Test', alpha=0.5)

        plt.plot(training_progress3['steps'], 1 - np.array(training_progress3['fidelity']), linestyle="-", color='g', marker='o', markevery=0.1, label='Train', alpha=0.5)
        plt.plot(training_progress3['steps'], 1 - np.array(training_progress3['fidelity_test']), linestyle="--", color='g', label='Test', alpha=0.5)

        plt.plot(training_progress4['steps'], 1 - np.array(training_progress4['fidelity']), linestyle="-", color='y', marker='o', markevery=0.1, label='Train', alpha=0.5)
        plt.plot(training_progress4['steps'], 1 - np.array(training_progress4['fidelity_test']), linestyle="--", color='y', label='Test', alpha=0.5)

        plt.axvline(x=3**8, color='r', linestyle='--', label='Worst case random search')
        plt.xlabel("Steps", fontsize=15)
        plt.ylabel("Cost", fontsize=15)
        plt.title('Deep Q Learning', fontsize=16)

        plt.ylim(0, 3**8 + 500)
        train_line = mlines.Line2D([], [], color='k', marker='o', markersize=5, label='Train', linestyle="-")
        test_line = mlines.Line2D([], [], color='k', linestyle="--", markersize=5, label='Test')

        worst_case_line = mlines.Line2D([], [], color='r', linestyle='--', label='Worst case random search')

        plt.legend(handles=[train_line, test_line, worst_case_line], fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.ylim(0, 1)
        plt.savefig("DQN_DISCRETE.png", dpi=500, bbox_inches="tight")
        plt.show()


In the plot, you should observe the cost (1 - fidelity) decreasing over time, indicating that the model is learning to optimize the quantum gate angles.


Optimizing with Greedy Search
-----------------------------

We can also use a greedy search algorithm to find the optimal angles for our quantum circuit. We will use the same environment as before, but we will use a different algorithm to find the optimal angles.


.. code-block:: python

    import itertools as it

    class GreedyLayerOptimizer:
        def __init__(self, layers, discrete_angles, eps=0.05, max_iters=1000000, max_layers=3):
            self.layers = layers
            self.discrete_angles = discrete_angles
            self.eps = eps
            self.max_iters = max_iters
            self.max_layers = max_layers
            self.n_steps = 0

        def optimize(self, cost, angles, num_iters=3, callback=None, verbose=False):
            self.n_steps = 0
            done = False

            for i in range(num_iters):
                if done:
                    break

                angles = np.random.choice([0, -np.pi/2], len(gs.trainable_nodes))
                new_angles = angles.copy()

                for n in range(1, self.max_layers + 1):
                    if verbose:
                        print(f"Optimizing {n} layers. Max: {self.max_layers}")
                    new_angles, new_cost = self.layer_opt(cost, new_angles, n, callback, verbose)

                    if self.n_steps >= self.max_iters:
                        print("Max iterations reached")
                        done = True
                        angles = new_angles
                        break
                    if new_cost < 0.01:
                        print("Cost below threshold")
                        done = True
                        angles = new_angles
                        break

                    angles = new_angles

                if verbose:
                    print(f"Iteration {i + 1} of {num_iters}: {angles} with value {cost(angles)}")
            return angles

        def layer_opt(self, cost, angles, n, callback=None, verbose=False):
            new_angles = angles.copy()

            for i in range(len(self.layers) - n + 1):
                merged_layer = sum(self.layers[i:i + n], [])
                best_cost = cost(new_angles)
                best_angles = new_angles.copy()

                for angle_combination in it.product(self.discrete_angles, repeat=len(merged_layer)):
                    self.n_steps += 1
                    for layer, angle in zip(merged_layer, angle_combination):
                        new_angles[layer] = angle

                    curr_cost = cost(new_angles)
                    if curr_cost < best_cost or np.random.rand() < self.eps:
                        if verbose:
                            print(f"New best cost: {curr_cost} < {best_cost}")
                        best_cost = curr_cost
                        best_angles = new_angles.copy()

                    if callback is not None:
                        callback(best_angles, self.n_steps)

                    if best_cost < 0.01:
                        break

                new_angles = best_angles
            return new_angles, best_cost


We can now train our model using the greedy search algorithm.

.. code-block:: python

    runs_train = {}
    runs_test = {}
    steps_runs = {}
    max_cost_calls = {}
    thetas_op = {}
    for i in range(5):
        theta = np.random.choice([0, -np.pi/2], len(gs.trainable_nodes))
        global_cost_calls = 0

        cost_train = []
        cost_test = []
        step = []

        my_callback = create_callback(X_train, y_train,X_test, y_test)
        opt = GreedyLayerOptimizer(eps=0, layers =gs.ordered_layers(train_indices=True), discrete_angles = [0, -np.pi/2, -np.pi/4], max_layers=5)
        theta = opt.optimize(lambda x: cost(x, X_train, y_train), theta, callback = my_callback)

        runs_train[i] = cost_train.copy()
        runs_test[i] = cost_test.copy()
        steps_runs[i] = step.copy()
        thetas_op[i] = theta.copy()

        cost_train.clear()
        cost_test.clear()
        step.clear()
        
        max_cost_calls[i] = global_cost_calls


Finally, we can plot the learning curve for the greedy search algorithm.

.. admonition:: Code for plotting learning curve
    :class: codeblock
    :collapsible:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.colors as mcolors

        cmap = mcolors.LinearSegmentedColormap.from_list("viridis", plt.get_cmap("viridis").colors)

        num_colors = 5
        colors = [cmap(i) for i in np.linspace(0, 1, num_colors+1)]

        for i in range(num_colors):
            color = colors[i]
            plt.plot(steps_runs[i], runs_train[i], linestyle="-", color=color, marker='o', markevery=0.1, alpha=0.5)
            plt.plot(steps_runs[i], runs_test[i], color=color, linestyle="--", markevery=0.1, alpha=0.5)
            plt.plot(steps_runs[i][-1], runs_test[i][-1], marker='o', c='b')
            plt.plot(steps_runs[i][-1], runs_train[i][-1], marker='*', c='r')

        train_line = mlines.Line2D([], [], color='k', marker='o', markersize=5, label='Train', linestyle="-")
        test_line = mlines.Line2D([], [], color='k', linestyle="--", markersize=5, label='Test')
        plt.axvline(x=3**8, color='r', linestyle='--', label='Worst case random search')

        plt.xlabel("Steps", fontsize=15)
        plt.ylabel("Cost", fontsize=15)
        plt.title("Greedy Layer Optimizer", fontsize=16)

        worst_case_line = mlines.Line2D([], [], color='r', linestyle='--', label='Worst case random search')

        plt.legend(handles=[train_line, test_line, worst_case_line], fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig("greedy_layer_optimizer_all.png", dpi=500, bbox_inches='tight')


We can get a detailed view of the learning curve of one run of the greedy search algorithm.

.. admonition:: Code for plotting learning curve
    :class: codeblock
    :collapsible:

    .. code-block:: python

        i = 1
        plt.plot(steps_runs[i], runs_train[i], label=f"Train", linestyle="-", color='k', marker='o', markevery=0.1)
        plt.plot(steps_runs[i], runs_test[i], label=f"Test", color='k', linestyle="--",)
        plt.plot(steps_runs[i][-1], runs_test[i][-1], marker='o', c='b')
        plt.plot(steps_runs[i][-1], runs_train[i][-1], marker='*', c='r')

        plt.xlabel("Steps", fontsize=15)
        plt.ylabel("Cost", fontsize=15)
        plt.title("Greedy Layer Optimizer", fontsize=16)
        plt.legend(fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig("greedy_layer_optimizer.png", dpi=500, bbox_inches='tight')

Conclusion
----------

In this tutorial, we demonstrated how to use Deep Q-Learning and a Greedy layer optimizer to learn the angles in a measurement pattern to implement a quantum gate. 