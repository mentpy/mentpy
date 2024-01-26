Learning a Quantum Instrument
=============================


.. meta::
   :description: Learning a Quantum Instrument using Measurement-Based Quantum Computation
   :keywords: mb-qml, measurement-based quantum machine learning, quantum machine learning, mbqc

**Author(s):** `Luis Mantilla <https://x.com/realmantilla>`_

.. admonition:: Note
   :class: warning
   
   This tutorial is under construction


.. code-block:: python

    import mentpy as mp
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set_theme()

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
    
    mp.draw(mgs, label='indices', edge_color_control='black', figsize = (12,4), fix_wires=[(8, '*', '*', '*','*', 9,10,11,12), (0,1,2,3, 13,14,15,16,17), (4,5,6,7, "*", "*","*", "*", 18, 19,20,21,22)])
    plt.savefig('teleportation.png', dpi=500)

    wires = [[0,1,2,3,13,14,15,16,17], [4,5,6,7,18,19,20,21,22],[8,9,10,11,12]]
    schedule = [0,4,8,5,1,6,2,3,7,13,14,9,15,10,11,16,17,18,19,12,20,21, 22]

    psK = mp.PatternSimulator(mgs, backend='numpy-dm', window_size=5, dev_mode = True, wires = wires, schedule = schedule)

    INPUT_STATE = mp.utils.generate_haar_random_states(1,1)
    st0 = np.array([1,0])
    st0xst0 = np.kron(st0, st0)
    MYINPUTSTATE = np.kron(st0xst0, INPUT_STATE)

    psK.reset(input_state=MYINPUTSTATE)
    myangles = np.zeros(len(mgs.trainable_nodes))
    myangles[4] = np.pi/2
    myangles[10] = np.pi/2

    myangles[12] = np.pi
    myangles[13] = np.pi
    myangles[14] = 0
    myangles[15] = np.pi

    psi = psK(myangles)
    outcomes= (psK.outcomes[12]+1) %2, (psK.outcomes[17]+1) %2
    print(psi, outcomes, mp.calculator.fidelity(psi, np.outer(INPUT_STATE, INPUT_STATE.conj())))

    num2text = {0: '0', np.pi/2: r'$\pi/2$', np.pi: r'$\pi$', 3*np.pi/2: r'$3\pi/2$'}
    labels = {v:num2text[m] for v,m in zip(mgs.trainable_nodes, myangles)}

    for i in mgs.nodes:
        if i not in mgs.trainable_nodes and i in mgs.outputc:
            labels[i] = '0'
        elif i not in mgs.quantum_output_nodes and i in mgs.output_nodes:
            labels[i] = 'Z'



    mp.draw(mgs, label='indices',labels=labels, edge_color_control='black', figsize = (12,4), fix_wires=[(8, '*', '*', '*','*', 9,10,11,12), (0,1,2,3, 13,14,15,16,17), (4,5,6,7, "*", "*","*", "*", 18, 19,20,21,22)])
    plt.savefig('teleportationEXACTSOLUTION.png', dpi=500)

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

    #set normal theme
    # sns.set_theme()
    # default
    plt.style.use('default')
    # plt.style.use('seaborn-whitegrid')
    # number of steps
    MAX_NUM_RUNS=10
    MAX_NUM_STEPS=180
    num_steps = MAX_NUM_STEPS
    # calculate means and variances at each step across all runs
    train_means = [np.mean([runs_train[i][j] for i in range(MAX_NUM_RUNS)]) for j in range(MAX_NUM_STEPS)]
    train_vars = [np.var([runs_train[i][j] for i in range(MAX_NUM_RUNS)]) for j in range(MAX_NUM_STEPS)]
    test_means = [np.mean([runs_test[i][j] for i in range(MAX_NUM_RUNS)]) for j in range(MAX_NUM_STEPS)]
    test_vars = [np.var([runs_test[i][j] for i in range(MAX_NUM_RUNS)]) for j in range(MAX_NUM_STEPS)]

    min_vals1 = np.array(train_means) - np.sqrt(train_vars)
    # replace with 0 where min_vals is negative
    min_vals1[min_vals1 < 0] = 0

    min_vals2 = np.array(test_means) - np.sqrt(test_vars)
    # replace with 0 where min_vals is negative
    min_vals2[min_vals2 < 0] = 0

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(train_means, label='Train cost mean', color='blue')
    ax.fill_between(range(num_steps), min_vals1, 
                    np.array(train_means) + np.sqrt(train_vars), alpha=0.1, color='blue')
    ax.plot(test_means, label='Test cost mean', linestyle='--', color='green')
    ax.fill_between(range(num_steps), min_vals2, 
                    np.array(test_means) + np.sqrt(test_vars), alpha=0.1, color='green')

    # ax.legend()
    # put legend in lower left corner
    ax.legend(loc='lower left')


    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.title("Learning curve for a quantum instrument")
    plt.savefig('TeleportLearningCurve3.png', dpi=700)