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
    mygate = mp.gates.IsingXX(np.pi/4)
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


Training with Deep Q-Learning
-----------------------------

We train our model using the Deep Q-Learning algorithm provided by :mod:`stable_baselines3`.

.. code-block:: python

    training_progress = {'fidelity': [], 'steps': [], 'fidelity_test':[]}
    env = QuantumGymEnvironment(X_train, y_train, X_test, y_test, max_iters=200)
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=3*(3**8))


We can visualize the training progress by plotting the fidelity of the quantum state over time.

.. admonition:: Code for plotting learning curve
    :class: info
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

Conclusion
----------

In this tutorial, we demonstrated how to use Deep Q-Learning to optimize the angles of quantum gates in a quantum circuit. By training our model in a custom Gym environment, we can find the parameters that maximize the fidelity of our quantum state, paving the way for more efficient quantum computing.