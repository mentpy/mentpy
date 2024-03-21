Fisher information classifier
=============================


.. meta::
   :description: MB-QML to classify quantum states using Fisher information.
   :keywords: mb-qml, mbqc, measurement-based quantum machine learning, qml

**Author(s):** `Luis Mantilla <https://x.com/realmantilla>`_, Polina Feldmann, Dmytro Bondarenko

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction

Here, we will learn now to perform a classification of quantum states according to their Fisher information. First, we generate a dataset of states 

.. code-block:: python
   
   import os
   import numpy as np
   import pickle
   import uuid
   import datetime
   import argparse
   import logging

   import mentpy as mp

.. code-block:: python

   HH = np.kron(mp.gates.HGate, mp.gates.HGate)

   def two_deg_poly(x, coeffs):
      extended_x = np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])
      return np.sum(extended_x * coeffs)

   def get_probabilities(st, angles):
      pattern_simulator.reset(input_state=st)
      qdensity = pattern_simulator(angles)
      qdensity = HH @ qdensity @ HH
      return np.abs(qdensity[0,0]), np.abs(qdensity[-1,-1])

   def fisher_prediction(st, angles, polycoeffs):
      p00, p11 = get_probabilities(st, angles)
      return np.real(two_deg_poly(np.array([p00, p11]), polycoeffs))

   def prob_prediction(st, angles, polydeg=2, polycoeffs=None):
      return get_probabilities(st, angles)

   def generate_data(n_states):
      def create_states(n, state1, state2):
         tensor_00 = np.kron(state1, state1)
         tensor_01 = np.kron(state1, state2)
         tensor_10 = np.kron(state2, state1)
         tensor_11 = np.kron(state2, state2)

         angles = 2 * np.pi * np.random.rand(n)
         phases = 2 * np.pi * np.random.rand(n)

         states = np.zeros((n, 4), dtype='complex')
         for k in range(n):
               angle, phase = angles[k], phases[k]
               states[k] = (np.cos(angle) * tensor_00 + np.sin(angle) * tensor_01 +
                           np.cos(phase) * tensor_10 + np.sin(phase) * tensor_11)
               states[k] /= np.linalg.norm(states[k])
         return states

      half_n = n_states // 2

      basis0_1, basis1_1 = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
      states_1 = create_states(half_n, basis0_1, basis1_1)

      basis0_2, basis1_2 = np.array([1, 0]), np.array([0, 1])
      states_2 = create_states(half_n, basis0_2, basis1_2)

      return np.concatenate((states_1, states_2))

   def get_targets_or_fishers(states, hamilt, return_fisher=False):
      results = []
      for st in states:
         idealfisher = 4 * (np.conj(st.T) @ np.conj(hamilt.T) @ hamilt @ st - (np.conj(st.T) @ hamilt @ st)**2)
         if return_fisher:
               results.append(idealfisher)
         else:
               results.append(1 if idealfisher > 2 else 0)
      return results
      
   get_targets = lambda sts, hamilt: get_targets_or_fishers(sts, hamilt, return_fisher=False)
   get_fishers = lambda sts, hamilt: get_targets_or_fishers(sts, hamilt, return_fisher=True)

   def get_optimizer(opt_name, step_size, momentum, adaptive=False):
      opts = {'SGD': mp.optimizers.SGDOpt, 'Adam': mp.optimizers.AdamOpt, 'RCD': mp.optimizers.RCDOpt}
      return opts[opt_name](step_size=step_size, momentum=momentum) if opt_name == 'SGD' else opts[opt_name](step_size=step_size, adaptive=adaptive)

   def get_stochastic_batch(x, y, batch_size):
      idx = np.random.randint(0, len(x), batch_size)
      return x[idx], y[idx]

   def parallel_run(worker_id, args):
      np.random.seed(worker_id)

      states = np.array(gen_states(args.num_data) if args.data_dist == '2dfamily' else mp.utils.generate_haar_random_states(2, args.num_data))
      hamiltonian = (np.kron(np.array([[1,0], [0,-1]]), np.eye(2)) + np.kron(np.eye(2), np.array([[1,0], [0,-1]]))) / 2
      targets = np.array(get_targets(states, hamiltonian))
      x_train, x_test, y_train, y_test = mp.utils.train_test_split(states, targets, test_size=0.2, randomize=True)

      opt = get_optimizer(args.optimizer, args.step_size, args.momentum, adaptive=args.optimizer == 'RCD')

      theta = np.concatenate((np.random.rand(3) * 2 * np.pi, np.random.rand(6)))

      cost_train, cost_test = [], []
      for step in range(args.num_steps):
         batch_x, batch_y = (x_train, y_train) if not args.stochastic else get_stochastic_batch(x_train, y_train, 50)
         
         theta = opt.step(lambda x: cost(x, batch_x, batch_y, 2, asymmetric=args.asymmetric, regularization=args.regularization), theta, step)
         
         if step % 10 == 0:
            cost_train.append(cost(theta, x_train, y_train, asymmetric=args.asymmetric))
            cost_test.append(cost(theta, x_test, y_test, testing=True, asymmetric=args.asymmetric))

      return cost_train, cost_test, theta, (x_train, x_test)

   def parallel_run_wrapper(args):
      return parallel_run(*args)