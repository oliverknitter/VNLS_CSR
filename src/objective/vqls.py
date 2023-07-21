import torch
import torch.nn as nn
import numpy as np

from .hamiltonian import Hamiltonian

class VQLS(Hamiltonian):
    def __init__(self, A, b):
        super(VQLS, self).__init__()
        self.A = A
        self.b = b
        self.A_squared = self.square_A()
        self.b_dist, self.b_total, self.b_indexes = self.square_b()
        self.n, self.U, self.K = self.alias_method()

    # Takes A as a list of dictionaries of local Pauli operators and returns its square in the same form.
    # Function presumes no ordering of entries in dictionary or of dictionaries in list.
    def square_A(self):
        square = []
        for item1 in self.A:
            for item2 in self.A:
                zero_flag = True
                local = dict(item1)
                for key in item2.keys():
                    if key in local:
                        if key == 'constant':
                            local[key] *= item2[key]
                        elif local[key] == item2[key]:
                            if local[key] in ['X', 'Y', 'Z']: # Operating the same Pauli operator twice on the same qubit is the identity; entry is deleted from dictionary
                                del(local[key])
                            elif local[key] in ['+', '-']: # '+' and '-' operators are nilpotents, term becomes zero
                                zero_flag = False
                            # '0' and '1' operators are projections, remain the same
                        elif (local[key] in ['X', 'Z'] and item2[key] in ['X', 'Z']) or (local[key] == 'Y') or (item2[key] == 'Y'):
                            local[key] += item2[key]
                        else:
                            if (local[key] == 'X' and item2[key] == '-') or (local[key]=='+' and item2[key] == 'X') or (local[key]=='Z' and item2[key] == '0') or (local[key]=='0' and item2[key] == 'Z')  or (local[key]=='+' and item2[key] == '-'):
                                local[key] = '0'
                            elif (local[key] == 'X' and item2[key] == '1') or (local[key]=='0' and item2[key] == 'X') or (local[key]=='Z' and item2[key] == '+') or (local[key]=='+' and item2[key] == 'Z') or (local[key]=='0' and item2[key] == '+')  or (local[key]=='+' and item2[key] == '1'):
                                if (local[key]=='+' and item2[key] == 'Z'):
                                    local['constant'] *= (-1.0)
                                local[key] = '+'
                            elif (local[key] == 'X' and item2[key] == '0') or (local[key]=='1' and item2[key] == 'X') or (local[key]=='Z' and item2[key] == '-') or (local[key]=='-' and item2[key] == 'Z') or (local[key]=='-' and item2[key] == '0') or (local[key]=='1' and item2[key] == '-'):
                                if (local[key]=='Z' and item2[key] == '-'):
                                    local['constant'] *= (-1.0)
                                local[key] = '-'
                            elif (local[key] == 'X' and item2[key] == '+') or (local[key]=='-' and item2[key] == 'X') or (local[key]=='Z' and item2[key] == '1') or (local[key]=='1' and item2[key] == 'Z') or (local[key]=='-' and item2[key] == '+'):
                                if (local[key]=='Z' and item2[key] == '1') or (local[key]=='1' and item2[key] == 'Z'):
                                    local['constant'] *= (-1.0)
                                local[key] = '1'
                            else:
                                zero_flag = False # All other combinations are zero-divisors
                        #elif (local[key] in ['0', '1', '+', '-']) or (item2[key] in ['0', '1', '+', '-']):
                            
                    else:
                        local[key] = item2[key]
            
                if zero_flag:
                    square.append(local)
        return square
    
    def square_b(self): # Returns the unnormalized probability distribution of b, the normalization factor, and indexes for the keys of b
        indexes = {}
        b_dist = {}
        counter = 1
        for key in self.b.keys():
            indexes[counter] = key
            b_dist[key] = self.b[key]**2
            counter += 1 
        return b_dist, sum(b_dist.values()), indexes

    def alias_method(self): # Prepares and returns lookup tables for Alias method sampling: https://en.wikipedia.org/wiki/Alias_method
        n = len(self.b_dist)
        U = np.zeros(n)
        K = np.zeros(n)

        overfull = []
        underfull = []
        exact = []

        for i in range(n):
            U[i] = n*self.b_dist[self.b_indexes[i+1]]/self.b_total
            if U[i] > 1:
                overfull.append(i+1)
            elif U[i] < 1:
                underfull.append(i+1)
            else:
                exact.append(i+1)
        for i in exact:
            K[i-1] = i

        while overfull and underfull:
            over = overfull.pop(0)
            under = underfull.pop(0)
            K[under - 1] = over
            U[over - 1] += U[under - 1] - 1.0
            exact.append(under)
            if U[over - 1] > 1:
                overfull.append(over)
            elif U[over - 1] < 1:
                underfull.append(over)
            else:
                exact.append(over)

        while overfull:
            over = overfull.pop(0)
            U[over - 1] = 1.0

        while underfull:
            under = underfull.pop(0)
            U[under - 1] = 1.0
        
        return n, U, K

    def b_sampler(self, batch_size): # Samples spin configurations from |b|^2 distribution using lookup tables generated by self.alias_method()
        random_samples = np.random.rand(batch_size)
        i = np.floor(self.n*random_samples) + 1
        i = i.astype(int)
        y = self.n*random_samples + 1 - i

        for j in range(batch_size):
            if y[j] >= self.U[i[j]-1]:
                i[j] = self.K[i[j]-1]
        
        sample_configs = []
        for j in i:
            sample_configs.append(self.b_indexes[j])
        return torch.Tensor(sample_configs)

    def b_eval(self, samples): # Returns corresponding values of b for a given batch of spin configurations
        values = []
        for item in samples.numpy():
            key = tuple(item)
            if key in self.b:
                values.append(self.b[key])
            else:
                values.append(0.0)
        return torch.Tensor(values)

    def row_entry(self, samples, observable): # Computes the column and value of the nonzero entry of a product of local Pauli operators at a given row.
        samples = samples.detach().clone()
        values = torch.ones(samples.shape[0])
        for key in observable: # Iterates through local Pauli operators and computes necessary values
            if key == 'constant':
                values *= observable[key]
            elif observable[key] == 'Z':
                values *= samples[:, key-1]
            elif observable[key] == 'X':
                samples[:, key-1] *= -1.0
            elif observable[key] == 'Y':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                samples[:, key-1] *= -1.0
                values *= samples[:, key-1]*1.0j

            # Include standard basis matrices
            elif observable[key] == '0':
                values *= torch.sign(samples[:, key-1] + 1)
            elif observable[key] == '1':
                values *= torch.sign(1 - samples[:, key-1])
            elif observable[key] == '+':
                values *= torch.sign(samples[:, key-1] + 1)
                samples[:, key-1] *= -1.0
            elif observable[key] == '-':
                values *= torch.sign(1 - samples[:, key-1])
                samples[:, key-1] *= -1.0
        
            # Need to include cases for pairwise products of Pauli matrices and products of Y and basis matrices- not mathematically necessary, but prevents need to recast to complex tensors for purely real problems.
            elif observable[key] == 'XZ':
                samples[:, key-1] *= -1.0
                values *= samples[:, key-1]
            elif observable[key] == 'ZX':
                values *= samples[:, key-1]
                samples[:, key-1] *= -1.0
            elif observable[key] == 'XY':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= samples[:, key-1] * 1.0j
            elif observable[key] == 'YX':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= samples[:, key-1] * -1.0j
            elif observable[key] == 'YZ':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                samples[:, key-1] *= -1.0
                values *= 1.0j
            elif observable[key] == 'ZY':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                samples[:, key-1] *= -1.0
                values *= -1.0j

            # Basis--Y products
            elif observable[key] == 'Y0':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(1 - samples[:, key-1])*1.0j
                samples[:, key-1] *= -1.0

            elif observable[key] == '0Y':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(samples[:, key-1] + 1)*(-1.0j)
                samples[:, key-1] *= -1.0

            elif observable[key] == 'Y+':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(1 - samples[:, key-1])*1.0j

            elif observable[key] == '+Y':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(samples[:, key-1] + 1)*1.0j

            elif observable[key] == 'Y-':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(samples[:, key-1] + 1)*(-1.0j)

            elif observable[key] == '-Y':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(1 - samples[:, key-1])*(-1.0j)
            
            elif observable[key] == 'Y1':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(samples[:, key-1] + 1)*(-1.0j)
                samples[:, key-1] *= -1.0

            elif observable[key] == '1Y':
                if values.type() != 'torch.ComplexFloatTensor': # Must recast to complex tensor if necessary.
                    values = values.type(torch.complex64)
                values *= torch.sign(1 - samples[:, key-1])*1.0j
                samples[:, key-1] *= -1.0
            
            else:
                print('Unidentified local operator: ' + observable[key])

        return samples, values

    def row_product(self, samples, observable, model): # For each sample in batch, returns product of the associated row of observable and the model.
        values = torch.zeros(samples.shape[0]) # Create output vector for batch
        values = torch.complex(values,values)
        for item in observable:
            columns, entries = self.row_entry(samples, item)
            values += entries*model(columns).exp()
        return values

    def b_row_product(self, samples, observable): # For each sample in batch, returns product of the associated row of observable and b.
        values = torch.zeros(samples.shape[0])
        for item in observable:
            columns, entries = self.row_entry(samples, item)
            values += entries*self.b_eval(columns)
        return values
    
    def b_mean(self, model, batch_size = 128): # Samples a batch of confirguations from |b|^2 and averages the corresponding local energies of A.
        b_samples = self.b_sampler(batch_size)
        b_vals = self.b_eval(b_samples)
        energies = self.row_product(b_samples, self.A, model)/b_vals
        mean = torch.mean(energies)
        return mean

    def compute_local_energy(self, samples, model): # Computes the local energy of the VQLS Hamiltonian.
        with torch.no_grad():
            mean = self.b_mean(model, samples.shape[0]) # Currently set to compute local energy for b_configurations in sample batches of size num_sites.
            local_energy = self.row_product(samples, self.A_squared, model) - self.b_row_product(samples, self.A)*mean
        log_psi = model(samples)
        return local_energy/log_psi.exp(), log_psi

    def exact_test(self, num_sites, model):
        # Prints the fidelity between the RBM's learned state and the true solution. Not particularly efficient. Not yet built to handle matrices/vectors A and b with complex entries.
        # Only use for small qubit size!
        all_configurations = []
        b_vector = []

        truth = getattr(self, 'truth', None) # Retrieves stored true solution, if it has been previously computed.
        if truth is not None:
            for i in range(2**num_sites):
                configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
                all_configurations.append(configuration)

            learned_vector = model(torch.Tensor(all_configurations)).exp() # Unnormalized learned solution vector
            true_vector = truth
        
        else:
            for i in range(2**num_sites):
                configuration = -2.0*(np.fromiter(np.binary_repr(i, width = num_sites), dtype=float))+1
                all_configurations.append(configuration)

                key = tuple(configuration)
                if key in self.b:
                    b_vector.append(self.b[key])
                else:
                    b_vector.append(0.0)
            learned_vector = model(torch.Tensor(all_configurations)).exp() # Unnormalized learned solution vector
            b_vector = torch.Tensor(b_vector) # Vector b constructed in full for exact solver

            A_matrix = torch.zeros(2**num_sites, 2**num_sites) # Construct A matrix for exact solver
            for item in self.A:
                if 1 in item:
                    if item[1] == 'X':
                        term = torch.Tensor([[0.0, 1.0],[1.0, 0.0]])
                    elif item[1] == 'Z':
                        term = torch.Tensor([[1.0, 0.0],[0.0, -1.0]])
                    elif item[1] == 'XZ':
                        term = torch.Tensor([[0.0, -1.0],[1.0, 0.0]])
                    elif item[1] == 'ZX':
                        term = torch.Tensor([[0.0, 1.0],[-1.0, 0.0]])
                    elif item[1] == '0':
                        term = torch.Tensor([[1.0, 0.0],[0.0, 0.0]])
                    elif item[1] == '+':
                        term = torch.Tensor([[0.0, 1.0],[0.0, 0.0]])
                    elif item[1] == '-':
                        term = torch.Tensor([[0.0, 0.0],[1.0, 0.0]])
                    elif item[1] == '1':
                        term = torch.Tensor([[0.0, 0.0],[0.0, 1.0]])
                else:
                    term = torch.eye(2)
                term *= item['constant']
                for i in range(2, num_sites + 1):
                    if i in item:
                        if item[i] == 'X':
                            term = torch.kron(term, torch.Tensor([[0.0, 1.0],[1.0, 0.0]]))
                        elif item[i] == 'Z':
                            term = torch.kron(term, torch.Tensor([[1.0, 0.0],[0.0, -1.0]]))
                        elif item[i] == 'XZ':
                            term = torch.kron(term, torch.Tensor([[0.0, -1.0],[1.0, 0.0]]))
                        elif item[i] == 'ZX':
                            term = torch.kron(term, torch.Tensor([[0.0, 1.0],[-1.0, 0.0]]))
                        elif item[i] == '0':
                            term = torch.kron(term, torch.Tensor([[1.0, 0.0],[0.0, 0.0]]))
                        elif item[i] == '+':
                            term = torch.kron(term, torch.Tensor([[0.0, 1.0],[0.0, 0.0]]))
                        elif item[i] == '-':
                            term = torch.kron(term, torch.Tensor([[0.0, 0.0],[1.0, 0.0]]))
                        elif item[i] == '1':
                            term = torch.kron(term, torch.Tensor([[0.0, 0.0],[0.0, 1.0]]))
                    else:
                        term = torch.kron(term, torch.eye(2))
                A_matrix += term

            true_vector, LU = torch.solve(torch.unsqueeze(b_vector, -1), A_matrix)
            true_vector = torch.squeeze(true_vector)
            true_vector /= torch.linalg.norm(true_vector)
            self.truth = true_vector

        learned_vector /= torch.linalg.norm(learned_vector)

        fidelity = torch.linalg.norm(torch.sum(learned_vector*true_vector))**2
        return learned_vector.detach(), true_vector.detach(), fidelity.detach()