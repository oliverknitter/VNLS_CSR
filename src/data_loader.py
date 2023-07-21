import os
import copy
import numpy as np
from scipy.sparse import csr_matrix

def create_random_graph_adjacency_mtx(num_sites):
    adjacency = np.random.randint(2, size=[num_sites, num_sites])
    adjacency = (adjacency + adjacency.transpose())//2
    np.fill_diagonal(adjacency, 0)
    return adjacency

def generate_full_matrix(num_sites, observable): # Given a dictionary-represented observable, returns the full matrix as a numpy array.
    matrix = np.zeros((2**num_sites, 2**num_sites))
    for item in observable:
        if 1 in item:
            if item[1] == 'X':
                term = np.array([[0.0, 1.0],[1.0, 0.0]])
            elif item[1] == 'Z':
                term = np.array([[1.0, 0.0],[0.0, -1.0]])
            elif item[1] == 'XZ':
                term = np.array([[0.0, -1.0],[1.0, 0.0]])
            elif item[1] == 'ZX':
                term = np.array([[0.0, 1.0],[-1.0, 0.0]])
            elif item[1] == '0':
                term = np.array([[1.0, 0.0],[0.0, 0.0]])
            elif item[1] == '+':
                term = np.array([[0.0, 1.0],[0.0, 0.0]])
            elif item[1] == '-':
                term = np.array([[0.0, 0.0],[1.0, 0.0]])
            elif item[1] == '1':
                term = np.array([[0.0, 0.0],[0.0, 1.0]])
        else:
            term = np.identity(2)
        term *= item['constant']
        for i in range(2, num_sites + 1):
            if i in item:
                if item[i] == 'X':
                    term = np.kron(term, np.array([[0.0, 1.0],[1.0, 0.0]]))
                elif item[i] == 'Z':
                    term = np.kron(term, np.array([[1.0, 0.0],[0.0, -1.0]]))
                elif item[i] == 'XZ':
                    term = np.kron(term, np.array([[0.0, -1.0],[1.0, 0.0]]))
                elif item[i] == 'ZX':
                    term = np.kron(term, np.array([[0.0, 1.0],[-1.0, 0.0]]))
                elif item[i] == '0':
                    term = np.kron(term, np.array([[1.0, 0.0],[0.0, 0.0]]))
                elif item[i] == '+':
                    term = np.kron(term, np.array([[0.0, 1.0],[0.0, 0.0]]))
                elif item[i] == '-':
                    term = np.kron(term, np.array([[0.0, 0.0],[1.0, 0.0]]))
                elif item[i] == '1':
                    term = np.kron(term, np.array([[0.0, 0.0],[0.0, 1.0]]))
            else:
                term = np.kron(term, np.identity(2))
        matrix += term
    return matrix

def create_VQLS_problem(num_sites, vector_choice, kappa = 10.0, sparsity_coeff = 3, J = 0.1):
    qubits = num_sites
    A = []
    b = {}

    # Generate Ising-inspired Hermitian matrix A
    for i in range(qubits - 1):
        operator = {'constant': J}
        operator[i + 1] = 'Z'
        operator[i + 2] = 'Z'
        A.append(operator)
    
    for i in range(qubits):
        operator = {'constant': 1.0}
        operator[i + 1] = 'X'
        A.append(operator)

    # Compute zeta and eta coefficents and return result
    lambda_max = float(num_sites)
    lambda_min = float(-num_sites)
    eta = (lambda_max - kappa*lambda_min)/(kappa - 1)
    zeta = (lambda_max + eta)

    A.append({'constant': eta})
    for operator in A:
        operator['constant'] *= 1.0/zeta

    # Generate vector b depending on vector_choice
    if vector_choice == 'alternation': # Entries of b alternate 2--1, starting with 2
        for i in range(2**num_sites):
            configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
            b[tuple(configuration)] = 1.0
            if i%2 == 0:
                b[tuple(configuration)] = 2.0
    
    elif vector_choice == 'constant':
        for i in range(2**num_sites):
            configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
            b[tuple(configuration)] = 1.0

    elif vector_choice == 'one_sparse': # The number of nonzero entries is bounded above by some fixed power of qubit size. Cannot be represented by a real RBM in general
        if sparsity_coeff == 0:
            sparsity_coeff = 3
        spins = np.array([1.0, -1.0])
        configurations = np.random.choice(spins, (qubits**sparsity_coeff, qubits))
        for item in configurations:
            b[tuple(item)] = 1.0
    
    elif vector_choice == 'cosine': # Deterministically generates a problem whose solution is feasible for a real-valued RBM to learn.
        x = np.zeros(2**num_sites)
        k = 1.0
        for i in range(2**num_sites):
            x[i] = (np.cos(k*i*2*np.pi/(2**num_sites - 1)))**2

        A_matrix = generate_full_matrix(num_sites, A)
        b_vector = np.matmul(A_matrix, x)
        for i in range(2**num_sites):
            configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
            b[tuple(configuration)] = b_vector[i]
    
    elif vector_choice == 'negative':
        for i in range(2**num_sites):
            configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
            b[tuple(configuration)] = (-1.0)**i

    elif vector_choice == 'poisson_init':
        matrix = generate_full_matrix(num_sites, A)
        target = np.ones(2**num_sites)
        solution = np.linalg.solve(matrix, target)
        for i in range(2**num_sites):
            configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
            b[tuple(configuration)] = solution[i]
        
        A = [{'constant':1.0}]
    
    return A, b

def vqls_direct_loader(num_sites, datum, Matrix=False, tol=1e-15): # Loads input matrix/vector entry-by-entry into VQLS format. Highly inefficient for dense systems
    if Matrix:
        for i in range(2**num_sites):
            for j in range(2**num_sites):
                if abs(datum[i,j])<tol:
                    datum[i,j] = 0.0
        output = csr_matrix(datum)
    else:
        output = {}
        for i in range(2**num_sites):
            if abs(datum[i])>tol:
                configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
                output[tuple(configuration)] = datum[i].item()
    return output

def create_data(pb_type, num_sites, vector_choice):
    if pb_type in ['maxcut']:
        adjacency = create_random_graph_adjacency_mtx(num_sites)
        data = {'adjacency': adjacency}
    elif pb_type in ['vqls']:
        A, b = create_VQLS_problem(num_sites, vector_choice)
        data = {'A': A, 'b': b}
    return data

def perturb_data(data, perburb_fac): # Currently only perturbs maxcut problems!
    data_copy = copy.deepcopy(data)
    for key in data_copy.keys():
        if key in ['adjacency']:
            adjacency = data_copy[key]
            noise = np.random.normal(0, perburb_fac, size=adjacency.shape)
            noise = (noise + noise.transpose())/2
            adjacency = (adjacency + noise).clip(0,1).round().astype(np.int)
            np.fill_diagonal(adjacency, 0)
            data_copy[key] = adjacency
    return data_copy

def load_data(cfg, num_tasks=0, perburb_fac=0.0):
    pb_type = cfg.DATA.PROBLEM_TYPE
    num_sites = cfg.DATA.NUM_SITES
    vector_choice = cfg.DATA.VECTOR_CHOICE
    data_path = './datasets/{}'.format(pb_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if pb_type == 'vqls':
        data_path = os.path.join(data_path, "data-{}-{}.npy".format(num_sites, vector_choice))
    else:
        data_path = os.path.join(data_path, "data-{}.npy".format(num_sites))
    # create new data or load existing data if exists
    if not os.path.exists(data_path):
        data = create_data(pb_type, num_sites, vector_choice)
        np.save(data_path, data)
    else:
        data = np.load(data_path, allow_pickle=True).item()
        if pb_type == 'vqls_direct':
            if isinstance(data['A'], np.ndarray):
                data['A'] = vqls_direct_loader(num_sites, data['A'], True)
            if isinstance(data['b'], np.ndarray):
                data['b'] = vqls_direct_loader(num_sites, data['b'], False)
    # perturb data (draw from metadata distribution) if asked
    if num_tasks > 0:
        lst = []
        for _ in range(num_tasks):
            lst.append(perturb_data(data, perburb_fac))
        return lst
    else:
        return data

