from enum import Enum

import numpy as np
import networkx as nx
# from scipy import sparse

# NetworkX graph to adjacency matrix
def graph_to_adj_matrix(graph):
    return nx.to_numpy_array(graph)

# Adjacency matrix to NetworkX graph
def adj_matrix_to_graph(adj_matrix):
    return nx.from_numpy_array(adj_matrix)

# NetworkX graph to edge list
def graph_to_edge_list(graph):
    return list(graph.edges)

# Edge list to NetworkX graph
def edge_list_to_graph(edge_list):
    g = nx.Graph()
    g.add_edges_from(edge_list)
    return g

# NetworkX graph to adjacency list
def graph_to_adj_list(graph):
    return list(map(list, dict(graph.adjacency()).values()))

# Adjacency list to NetworkX graph
def adj_list_to_graph(adj_list):
    graph = nx.Graph()
    for index, neighbors in enumerate(adj_list):
        for neigh in neighbors:
            graph.add_edge(index, neigh)
    return graph

# Adjacency matrix to edge list
def adj_matrix_to_edge_list(adj_matrix):
    g = nx.from_numpy_array(adj_matrix)
    return list(g.edges)

# Edge list to adjacency matrix
def edge_list_to_adj_matrix(edge_list, num_nodes=None):
    if num_nodes is None:
        num_nodes = max(max(u, v) for u, v in edge_list) + 1
    g = nx.Graph()
    g.add_edges_from(edge_list)
    return nx.to_numpy_array(g, nodelist=range(num_nodes))

# # NetworkX graph to sparse matrix (CSR)
# def graph_to_sparse_matrix(graph):
#     return nx.to_scipy_sparse_matrix(graph, format='csr')

# # Sparse matrix (CSR) to NetworkX graph
# def sparse_matrix_to_graph(sparse_matrix):
#     return nx.from_scipy_sparse_matrix(sparse_matrix)

# # NetworkX graph to set of edges
# def graph_to_set_of_edges(graph):
#     return set(graph.edges)

# # Set of edges to NetworkX graph
# def set_of_edges_to_graph(edges_set):
#     g = nx.Graph()
#     g.add_edges_from(edges_set)
#    return g

class Measurement(Enum):
    X = 1
    Y = 2
    Z = 3

    XY = 12
    XZ = 13
    YZ = 23

class Q(Enum):
    A = 1
    B = 2
    C = 3
    def measure(self, m = Measurement.X):
        return m
"""
What is P-Flow?
"""

def print_assert(a, b):
    print(a, b)
    assert(a == b)
    return a == b

universe = { Q.A, Q.B, Q.C }
complement = lambda c: universe - c
input_nodes = { Q.A }
output_nodes = { Q.C }
edges = { ( Q.A, Q.B ), ( Q.B, Q.C ) }

I_C = lambda input_nodes: complement(input_nodes)
O_C = lambda output_nodes: complement(output_nodes)

G = nx.Graph()
G.add_edges_from({(Q.A, Q.B), (Q.B, Q.C)})

print(graph_to_adj_matrix(G))

"""
What does the matrix Gamma represent in the P-Flow algorithm?
- The adjacency matrix of the qubit network
"""

# { A -> B, B -> C }, reflected
gamma = graph_to_adj_matrix(G)
print(gamma)

"""
What does NGamma represent?
- Neighbours of a node Q
"""
# NGamma(u)
# u = A => { B }
# u = B => { A, C }
# u = C => { B }
def NGamma(u):
    return set(G.neighbors(u))

# print_assert(NGamma(Q.A), {Q.B})
# print_assert(NGamma(Q.B), {Q.A, Q.C})
# print_assert(NGamma(Q.C), {Q.B})

# TODO: what is A?
def OddG(graph, A):
    odd_vertices = [v for v in graph.nodes() if len(set(graph.neighbors(v)) & set(A)) % 2 == 1]
    return odd_vertices

"""
What controls the size of the identity matrix in the P-Flow algorithm?
"""
identity_matrix = np.identity(len(gamma))
identity = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]

# print_assert(identity_matrix, identity)

"""
How to define the intersection operator from a matrix representation of the network?
- Consider the input and output vector spaces
"""

# /\^p_u = { v \el O^c | v != u ^ lam(v) = P }
# TODO: I don't understand the intent behind these functions
# TODO: What are the measurement patterns expected in these results?
measurement_at_node = lambda node: node.measure()
lam_p_u = lambda P, u: { v for v in O_C(output_nodes) if v != u and measurement_at_node(v) == P }
lam_x_u = lambda u: lam_p_u(Measurement.X, u)
lam_y_u = lambda u: lam_p_u(Measurement.Y, u)
lam_z_u = lambda u: lam_p_u(Measurement.Z, u)

# Correct?
# print_assert(lam_x_u(Q.A), {Q.B})

# Prove that the cardinality of the edges selected are precisely the size 
# of the subspaces requried for the matricies.

# TODO: what is A?
K_a_u = lambda A, u: (A | lam_x_u(u) | lam_y_u(u)) & I_C(input_nodes)
P_a_u = lambda A, u: universe - (A | lam_y_u(u) | lam_z_u(u))
Y_a_u = lambda A, u: lam_y_u(u) - A

"""
To construct the block matrix, we are finding the sparse/filtered versions
"""

import numpy as np

# Create identity matrix
identity_matrix = np.identity(len(gamma))

def PauliFlowAux(V, gamma, I, lam, A={}, B=output_nodes, k=0):
    """
    Construct the solutions matrix for 
    """
    for u in complement(B):

        """
        MA,u :=
        [ Γ ∩ KA,u × PA,u
        _______________________
        (Γ + Id) ∩ KA,u × YA,u ]

        \begin{pmatrix}
        \Gamma \cap KA,u \times PA,u \\
        (\Gamma + Id) \cap KA,u \times YA,u
        \end{pmatrix}
        """
        # Initialize the mask based on G's shape
        mask_top = np.zeros_like(gamma, dtype=bool)
        mask_bot = np.zeros_like(gamma, dtype=bool)

        # Obtain the node to index mapping (assume gamma is from a NetworkX graph G)
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

        # Set the mask to True for adjacencies between nodes in K_a_u and P_a_u
        for k, p in zip(K_a_u(A, u), P_a_u(A, u)):
            if k in node_to_idx and p in node_to_idx: # Check if both nodes exist in the graph
                mask_top[node_to_idx[k], node_to_idx[p]] = True
                mask_top[node_to_idx[p], node_to_idx[k]] = True  # If needed, for undirected graph

        # TODO: parallel iteration
        # Set the mask to True for adjacencies between nodes in K_a_u and Y_a_u
        for k, p in zip(K_a_u(A, u), Y_a_u(A, u)):
            if k in node_to_idx and p in node_to_idx: # Check if both nodes exist in the graph
                mask_bot[node_to_idx[k], node_to_idx[p]] = True
                mask_bot[node_to_idx[p], node_to_idx[k]] = True  # If needed, for undirected graph

        # Use mask to set selected positions in gamma to 0 and combine with identity
        M_a_u_top = np.where(mask_top, 0, gamma)
        M_a_u_bot = np.where(mask_bot, 0, gamma + identity_matrix)
        M_a_u = np.vstack((M_a_u_top, M_a_u_bot))

        if u.measurement() in { Measurement.X, Measurement.Y, Measurement.XY }:
            # Top of the matrix is the one-hot vector for {u}
            # Bottom of the matrix is zeros
            # The matrix must fill out to be in the column space for
            pass
        elif u.measurement() in { Measurement.X, Measurement.Z, Measurement.XZ }:
            # Top of the matrix is:
            # (NGamma(u) & P_a_u(u)) | {u}
            # Bottom of the matrix is:
            # (NGamma(u) & Y_a_u(u)
            pass
        elif u.measurement() in { Measurement.Y, Measurement.Z, Measurement.YZ }:
            # Top of the matrix is:
            # (NGamma(u) & P_a_u(u))
            # Bottom of the matrix is:
            # (NGamma(u) & Y_a_u(u)
            pass

# (NGamma(u) & P_a_u(u)) | { u }
# (NGamma(u) & Y_a_u(u))

import numpy as np

def calculate_S_tilde_1(λ_tilde, u, PA_u, YA_u, NΓ):
    if λ_tilde == "XY":
        return [{u}, set()]
    elif λ_tilde == "XZ":
        neighbors = set(NΓ[u])  # Get neighbors of u
        return [(neighbors & PA_u) | {u}, neighbors & YA_u]
    elif λ_tilde == "YZ":
        neighbors = set(NΓ[u])  # Get neighbors of u
        return [neighbors & PA_u, neighbors & YA_u]

def calculate_S_tilde_2(λ_tilde, u, PA_u, YA_u, NΓ):
    num_nodes = len(NΓ)
    S_tilde_top = np.zeros((num_nodes, num_nodes))
    S_tilde_bot = np.zeros((num_nodes, num_nodes))

    neighbors_u = set(NΓ[u])
    PA_u_set = set(PA_u)
    YA_u_set = set(YA_u)

    if λ_tilde == "XY":
        S_tilde_top[u, u] = 1
    elif λ_tilde == "XZ":
        intersect_PA_u = neighbors_u & PA_u_set
        for i in intersect_PA_u:
            S_tilde_top[u, i] = 1
        S_tilde_top[u, u] = 1
        intersect_YA_u = neighbors_u & YA_u_set
        for j in intersect_YA_u:
            S_tilde_bot[u, j] = 1
    elif λ_tilde == "YZ":
        intersect_PA_u = neighbors_u & PA_u_set
        for i in intersect_PA_u:
            S_tilde_top[u, i] = 1
        intersect_YA_u = neighbors_u & YA_u_set
        for j in intersect_YA_u:
            S_tilde_bot[u, j] = 1

    # Stack the top and bottom blocks to create the full block matrix
    S_tilde = np.vstack((S_tilde_top, S_tilde_bot))
    return S_tilde

def create_identity_graph(input_graph: nx.Graph) -> nx.Graph:
    identity_graph = nx.MultiDiGraph()  # Use MultiDiGraph to allow self-loops
    # Add nodes from the original graph
    identity_graph.add_nodes_from(input_graph.nodes())
    # Add self-loops for each node
    for node in input_graph.nodes():
        identity_graph.add_edge(node, node)
    return identity_graph

class Measurement(Enum):
    X = 1
    Y = 2
    Z = 3

    XY = 12
    XZ = 13
    YZ = 23

class Q(Enum):
    A = 1
    B = 2
    C = 3
    def measure(self, m = Measurement.Y):
        return m
"""
What is P-Flow?
"""

def print_assert(a, b):
    print(a, b)
    assert(a == b)
    return a == b

universe = { Q.A, Q.B, Q.C }
complement = lambda c: universe - c
input_nodes = { Q.A }
output_nodes = { Q.C }
edges = { ( Q.A, Q.B ), ( Q.B, Q.C ) }

I_C = lambda input_nodes: complement(input_nodes)
O_C = lambda output_nodes: complement(output_nodes)

# print_assert(I_C(input_nodes), { Q.B, Q.C })
# print_assert(O_C(output_nodes), { Q.A, Q.B })

G = nx.Graph()
G.add_edges_from({(Q.A, Q.B), (Q.B, Q.C)})

print(graph_to_adj_matrix(G))

"""
What does the matrix Gamma represent in the P-Flow algorithm?
- The adjacency matrix of the qubit network
"""

# { A -> B, B -> C }, reflected
gamma = graph_to_adj_matrix(G)
print(gamma)

"""
What does NGamma represent?
- Neighbours of a node Q
"""
# NGamma(u)
# u = A => { B }
# u = B => { A, C }
# u = C => { B }
def NGamma(u):
    return set(G.neighbors(u))

# print_assert(NGamma(Q.A), {Q.B})
# print_assert(NGamma(Q.B), {Q.A, Q.C})
# print_assert(NGamma(Q.C), {Q.B})

# TODO: what is A?
def OddG(graph, A):
    odd_vertices = [v for v in graph.nodes() if len(set(graph.neighbors(v)) & set(A)) % 2 == 1]
    return odd_vertices

"""
What controls the size of the identity matrix in the P-Flow algorithm?
"""
identity_matrix = np.identity(len(gamma))
identity = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]

# print_assert(identity_matrix, identity)

"""
How to define the intersection operator from a matrix representation of the network?
- Consider the input and output vector spaces
"""

# /\^p_u = { v \el O^c | v != u ^ lam(v) = P }
# TODO: I don't understand the intent behind these functions
# TODO: What are the measurement patterns expected in these results?
measurement_at_node = lambda node: node.measure()
lam_p_u = lambda P, u: { v for v in O_C(output_nodes) if v != u and measurement_at_node(v) == P }
lam_x_u = lambda u: lam_p_u(Measurement.X, u)
lam_y_u = lambda u: lam_p_u(Measurement.Y, u)
lam_z_u = lambda u: lam_p_u(Measurement.Z, u)

# Correct?
# print_assert(lam_x_u(Q.A), {Q.B})

# Prove that the cardinality of the edges selected are precisely the size 
# of the subspaces requried for the matricies.

K_a_u = lambda A, u: (A | lam_x_u(u) | lam_y_u(u)) & I_C(input_nodes)
P_a_u = lambda A, u: universe - (A | lam_y_u(u) | lam_z_u(u))
Y_a_u = lambda A, u: lam_y_u(u) - A

# TODO: asserts
# K_a_u(G, Q.A)
# P_a_u(G, Q.A)
# Y_a_u(G, Q.A)

# 0. Helper methods
def node_to_idx(graph, node):
    """
    Given a node from an adjacency set, find its index within its corresponding adjacency matrix
    """
    return list(graph.nodes).index(node)

def adjacency_set_to_matrix(adj_set, universe):
    """
    Given a list of adjacencies and a universe of elements, construct the corresponding adjacency matrix.
    """
    print(adj_set, universe)
    graph = nx.Graph()
    graph.add_nodes_from(universe)
    graph.add_edges_from(adj_set)
    return nx.to_numpy_array(graph, dtype=int, nodelist=(universe))

def graph_to_matrix(graph):
    """
    Converts a NetworkX graph to its corresponding adjacency matrix.
    """
    return nx.to_numpy_array(graph, dtype=int, nodelist=(graph.nodes()))

def idx_to_adj(graph, row, col):
    """
    Given matrix, coordinates, find its adjacency representation.
    NOTE: if row and col coords are the same, return an identity adjacency, even if self loops aren't in the graph.
    """
    if row == col:
        return [(row, row)]
    else:
        adj = []
        if graph.has_edge(row, col):
            adj.append((row, col))
        if graph.has_edge(col, row):
            adj.append((col, row))
    return adj

def idx_to_node(graph: nx.Graph, n: int):
    """
    Given an index n, find the node within the graph it corresponds to.

    Parameters
    ----------
    graph : networkx.Graph
        The graph from which to find the node corresponding to index n.

    n : int
        The index for which to find the corresponding node.

    Returns
    -------
    node
        The corresponding node in the graph for the given index n.
    """
    # Ensure n is within the valid range of node indices
    assert 0 <= n < len(graph), "Index out of bounds."

    # Directly return the node corresponding to the index 
    return (graph.nodes())[n]

def create_identity_graph(input_graph: nx.Graph) -> nx.Graph:
    """
    Create an identity graph based on the nodes of the input graph.

    For every node in the input graph, the identity graph will contain a corresponding
    node with a self-loop. No edges between different nodes are added. The resulting graph
    mirrors the identity matrix for an adjacency matrix, with ones on the diagonal and zeros
    elsewhere.

    Parameters
    ----------
    input_graph : networkx.Graph
        The input graph from which to create the identity graph. Only the nodes of the
        input graph are used; edges are ignored.

    Returns
    -------
    networkx.MultiDiGraph
        An identity graph where each node has a self-loop.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3)])
    >>> I = create_identity_graph(G)
    >>> I.edges()
    MultiEdgeDataView([(1, 1), (2, 2), (3, 3)])
    >>> I.number_of_edges()
    3
    """
    identity_graph = nx.MultiDiGraph()  # Use MultiDiGraph to allow self-loops
    identity_graph.add_nodes_from(input_graph.nodes())
    for node in input_graph.nodes():
        identity_graph.add_edge(node, node)
    return identity_graph

def generate_membership_vector(graph: nx.Graph, nodes=[]) -> np.ndarray:
    """
    Returns a column vector representing the membership of a set of elements in the graph.
    The empty set is supported as an argument to return a vector with no membership.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph from which the membership vector is to be generated.

    nodes : iterable, optional
        An iterable of nodes for which the membership vector is to be generated. Default is an empty list, representing no membership.

    Returns
    -------
    np.ndarray
        A column vector with '1' at indices corresponding to the nodes and '0' elsewhere. If an empty set (or list) is provided, it returns a vector with '0's.
    """
    # Get a sorted list of all nodes to maintain consistent indexing
    graph_nodes = list(graph.nodes())
    # Initialize a column vector with '0's 
    membership_vector = np.zeros((len(graph_nodes), 1), dtype=int)
    # Set '1' at the index of each node present in the nodes set, if any
    for node in nodes:
        node_idx = graph_nodes.index(node)
        membership_vector[node_idx, 0] = 1
    return membership_vector
    
def PauliFlow(V, gamma, input_nodes, output_nodes, lam = lambda node: node.measure()):

    universe = gamma.nodes() 
    complement = lambda c: universe - c

    I_C = lambda input_nodes: complement(input_nodes)
    O_C = lambda output_nodes: complement(output_nodes)
    NGamma = lambda gamma, u: set(gamma.neighbors(u))

    # Supplementary constructs for the P-Flow algorithm:

    # measurement tests
    lam_p_u = lambda P, u: { v for v in O_C(output_nodes) if v != u and lam(v) == P }
    lam_x_u = lambda u: lam_p_u(Measurement.X, u)
    lam_y_u = lambda u: lam_p_u(Measurement.Y, u)
    lam_z_u = lambda u: lam_p_u(Measurement.Z, u)

    # Set of possible elements of the witness set K
    K_a_u = lambda A, u: (A | lam_x_u(u) | lam_y_u(u)) & I_C(input_nodes)

    # Set of verticies in the past/present which should remain corrected after measuring u
    P_a_u = lambda A, u: universe - (A | lam_y_u(u) | lam_z_u(u))

    # Set of verticies for condition: ∀v <= u.u 6 = v ∧ λ (v) = Y ⇒ (v ∈ p(u) ⇔ v ∈ Odd(p(u)))
    Y_a_u = lambda A, u: lam_y_u(u) - A

    # nodes with X/Y/Z measurement alone
    L_x, L_y, L_z = set(), set(), set()

    p = {}  # node_to_correction_set_map
    d = {}  # node_to_depth_map
    
    def PauliFlowAux(V, gamma, I, lam, A={}, B=output_nodes, k=0):
        """
        Construct the solutions matrix for M_a_u * X^K = S_lam_tilde
        Strategy:
        - encode all matricies with dimension N x N qubits (but without full rank)
        - use 1/0 to encode membership/non-membership of nodes
        - solve top and bottom systems separately, then combine the solutions
        """
        # accumulator for the solution set of correctibles at the given depth
        C = set()
        for u in complement(B): 
            # 1. Constraint matrix M_a_u

            # Top section:
            # First, get the adjacency list as a set of tuples, then constrain those tuples by those that are both in the witness set
            # and connected to a node that should remain corrected after measurement.
            # Then from the set of nodes within this set, fill the sparse matrix reflecting the top
            # linear system of equations.
            M_a_u_top_set = set(gamma.edges()) & set(zip(K_a_u(A, u), P_a_u(A, u)))
            M_a_u_top_mat = adjacency_set_to_matrix(M_a_u_top_set, gamma.nodes())
            
            # Bottom section
            # The 
            M_a_u_bot_set = (set(gamma.edges()) | set(create_identity_graph(gamma).edges())) & set(zip(K_a_u(A, u), Y_a_u(A, u)))
            M_a_u_bot_mat = adjacency_set_to_matrix(M_a_u_bot_set, gamma.nodes())

            # 2. Solutions matrix S_lam_tilde 
            if lam(u) in { Measurement.X, Measurement.Y, Measurement.XY }:
                # Top of the matrix is the one-hot vector for {u}
                # Bottom of the matrix is zeros
                # The matrix must fill out to be in the column space for
                S_lam_tilde_top_set = {u}
                S_lam_tilde_top_mat = generate_membership_vector(gamma, S_lam_tilde_top_set)
                S_lam_tilde_bot_set = {}
                S_lam_tilde_bot_mat = generate_membership_vector(gamma, S_lam_tilde_bot_set)
            elif lam(u) in { Measurement.X, Measurement.Z, Measurement.XZ }:
                # Top of the matrix is:
                # (NGamma(u) & P_a_u(u)) | {u}
                # Bottom of the matrix is:
                # (NGamma(u) & Y_a_u(u)
                S_lam_tilde_top_set = (set(gamma.neighbors(u)) & P_a_u(u)) | set(u)
                S_lam_tilde_top_mat = generate_membership_vector(gamma, S_lam_tilde_top_set)
                S_lam_tilde_bot_set = (set(gamma.neighbors(u)) & Y_a_u(u))
                S_lam_tilde_bot_mat = generate_membership_vector(gamma, S_lam_tilde_bot_set)
            elif lam(u) in { Measurement.Y, Measurement.Z, Measurement.YZ }:
                # Top of the matrix is:
                # (NGamma(u) & P_a_u(u))
                # Bottom of the matrix is:
                # (NGamma(u) & Y_a_u(u)
                S_lam_tilde_top_set = (set(gamma.neighbors(u)) & P_a_u(A, u))
                S_lam_tilde_top_mat = generate_membership_vector(gamma, S_lam_tilde_top_set)
                S_lam_tilde_bot_set = (set(gamma.neighbors(u)) & Y_a_u(A, u))
                S_lam_tilde_bot_mat = generate_membership_vector(gamma, S_lam_tilde_bot_set)

            # 3. Unknown matrix X_K
            # Solve the top system
            X_K_top, residuals_top, rank_top, s_top = np.linalg.lstsq(M_a_u_top_mat, S_lam_tilde_top_mat, rcond=None)

            # Solve the bottom system
            X_K_bot, residuals_bot, rank_bot, s_bot = np.linalg.lstsq(M_a_u_bot_mat, S_lam_tilde_bot_mat, rcond=None)

            # To ensure the solutions concur, multiply them together. As the solutions vectors represent the presence of an element in a set,
            # the matrix product should eliminate any 1s which represent presence in one set with 0s that represent absence in another set.
            
            # The product that satisfies this definition is the Hadamard product.
            # Since the numpy product is polymorphic it should work like this automatically.
            X_K = (X_K_top * X_K_bot).flatten()

            # The result should be a 1 x N vector
            assert X_K.ndim == 1, "Vector is not 1-dimensional"
            assert X_K.shape[0] > 0, "N must be greater than 0"
            assert np.all(np.isin(X_K, [0, 1])), "Vector contains values other than 0 and 1"

            # if a solution K_0 is found for any of K_XY, K_XZ, K_YZ, then do this
            # the system has solutions if X_K has any candidate members, as encoded as a binary vector
            is_not_zero_vector = np.any(X_K)
            has_solutions = is_not_zero_vector
            if has_solutions:
                # convert the vector back into the set of nodes that it represents
                K_0 = {idx_to_node(V, i) for i, val in enumerate(X_K) if val != 0}
                C.update(K_0)
                p[u] = K_0  # accumulate solutions for each node
                d[u] = k    # accumulate the depth for each node
        # if there are no for the given node solutions but the resource is explored beyond the input depth
        if not C and k > 0:
            if B == V:
                return (True, p, d)
            else:
                return (False, dict(), dict())
        else:
            B_prime = B | C
            return PauliFlowAux(V, gamma, I, lam, B_prime, B_prime, k + 1)

    for v in V:
        if v in output_nodes:
            d[v] = 0
        if lam(v) == Measurement.X:
            L_x.update({v})
        elif lam(v) == Measurement.Y:
            L_y.update({v})
        elif lam(v) == Measurement.Z:
            L_z.update({v})
            
    return PauliFlowAux(V, gamma, input_nodes, lam, set(), output_nodes, 0)

nodes = { Q.A, Q.B, Q.C }
input_nodes = { Q.A }
output_nodes = { Q.C }
edges = { ( Q.A, Q.B ), ( Q.B, Q.C ) }

G = nx.Graph()
G.add_edges_from(edges)

print(PauliFlow(nodes, G, input_nodes, output_nodes))

# Line of 3 qubits
# Measure the first one in X, the second at any angle
# { (A, B) (B, C) }
# universe = { A, B, C }
# input_nodes = {A}
# output_nodes = {C}
# I^C = { A, B, C } - { A } = { B, C }
# gamma =  [[0, 1, 0], 
#           [1, 0, 1], 
#           [0, 1, 0]]
# identiy = [[1,0,0]
#            [0,1,0]
#            [0,0,1]]
# gamma + identity
# intersection operator ~~~~ submatrix [rows given by K, columns given by P]
# => K, P of variable size, here gamma and identity are 3x3
# set of witnesses K will be the solution to a matrix

# /\^p_u = { v \el O^c | v != u ^ lam(v) = P }
# p = measurement basis
#
# -> {}
# -> {}
# -> {}
# /\^X_u = { }
# -> u = A => { }
# -> u = B => { A }
# -> u = C => { A }
# /\^Y_u = { }
# -> u = A => { }
# -> u = B => { }
# -> u = C => { }
# /\^Z_u = { }
# -> u = A => { }
# -> u = B => { }
# -> u = C => { }

# K_a_u = {a U /\^X U /\^Y} & {B, C}
#       = {a U {A} } & { B, C }
#       = {} ?

# P_a_u = {A, B, C} - {a U /\^Y U /\^Z}
#       = {A, B, C} - {} ?
#       = {A, B, C}

# Y_a_u = { } - a 

# Gamma

# \lam = XY:
# [{ u } | 0 ]
# ?
# u = A => [[1, 0, 0], 
#           [0,0,0], 
#           [0,0,0]]
# u = B => [[0, 0, 0], 
#           [0,1,0], 
#           [0,0,0]]
# u = C => [[0, 0, 0], 
#           [0, 0, 0], 
#           [0, 0, 1]]

# NGamma(u)
# u = A => { B } 
# u = B => { A, C }
# u = C => { B }

# NGamma(u) /\ P_a_u

# M_a_u = two block matricies, solved separately
# -> Gamma & ((K_a_u) X (P_a_u)) => find the adjacencies satisfying both constraints
# -> (Gamma + Id) & ((K_a_u) X (Y_a_u)) => find the adjacencies satisfying both constraints


# X_K ~~~ size N qubits
# M_a_u ~~~~ N x N
# => not full rank ~ sparsity
# S_lam ~~~ size N qubits
# keep zeroes wherever nodes are unconsidered
#
# whenever the set has an element, put 1, else put 0
# some rows will be all zero
# solve for top block and bottom block separately
# once solving for X_K, X_K' ~ element-wise product will find set intersection between both sets
#
# Sanity check:
# - once the system is solved ~~ K should be a subset of some other predicate
# - ensure the 1s/0s are not in the subsets
#
# index of vector = index of node in the subset

# Two approaches:
# - constraint before => 1 wherever there is a tuple
# - constraint after => find the submatrix rows of K and P
# If a node is not in K -> row is all zeroes M x N


# 1)
# [[0, 1] [1, 0]]       <= constrain
# -> getting the subsets with ~K guarantees the size
# 2)
# [[010], [100], [000]] <= padding
# -> solve the N x N

# 1 is more sparse