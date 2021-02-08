"""
Implementation of the mean-field Ising Influence Maximisation algorithm.
"""

#Import modules:
import numpy as np
import math
import scipy
from scipy import sparse
import networkx as nx

#Modules in this dir:
from . import projection_simplex as proj


def project_block_to_graph(block_sizes, block_level_vals):
    """
    Projects a set of values at the level of
    blocks to the nodes in the graph.
    """
    full_graph_values = []
    for k, n in enumerate(block_sizes):
        current_block = []
        for q in range(n):
            current_block.append(block_level_vals[k])
        full_graph_values = np.concatenate((full_graph_values, current_block))
    return full_graph_values

def get_ascending_pairs(values) :
    edges = np.concatenate(([0], np.cumsum(values)))
    pairs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return pairs

def block_level_average(block_sizes,node_values) :

        """
        Average a quantity defined at the level of nodes
        to the level of blocks.

        Parameters
        -------------

        node_values : list

        Values of a list at the level of nodes


        """

        block_level_average=[]
        size_ascending_pairs = get_ascending_pairs(block_sizes)
        for index_pairs in size_ascending_pairs :
            block_level_average.append(np.mean(node_values[index_pairs[0]:index_pairs[1]]))
        return block_level_average

class mean_field_ising_system :

    """
    Mean field Ising system on a specific graph
    """

    def __init__(self,graph,background_field,block_sizes=None,notebook=False) :

        """

        Initialises the mean-field Ising system class
        on a graph

        Parameters
        ----------
        graph : networkx graph

        Networkx graph encoding the interactions between spins

        background_field : numpy array

        Background fields for nodes - this array must be the same size as the
        network. Set this to zero if no external fields are required.

        block_sizes : list

        List of block sizes. Assume that graph nodes are
        ordered according to the respective blocks.

        """

        self.graph=graph
        self.background_field=background_field
        self.block_sizes=block_sizes

        #FP iteration parameters
        """
        Gamma = Damping parameter for fixed point iteration. For standard mean-field
        theory the dynamics will converge even with gamma=1.0.
        (See: Aspelmeier, Timo, et al. "Free-energy landscapes, dynamics, and the edge of chaos in mean-field models of spin glasses." Physical Review B 74.18 (2006): 184411.  )
        Note in the paper above the damping parameter is alpha and gamma is used differently.
        Damping can be introduced to provide a 'smoother' convergence.
        """
        self.gamma=1.0
        """
        tol : float
        Tolerance parameter. The iterations are terminated when the difference
        between successive magnetisation values goes below this tolerance.
        """
        self.tol=1E-5
        self.max_mf_fp_iterations=10000 #Sets a maximum to the number of iterations.
        self.mf_fp_init_state='aligned' #This can also be set as an array.
        self.mf_fp_noisy=True

        #IIM parameters
        self.max_mf_iim_iterations=1000
        self.mf_iim_step_size=1.0
        self.mf_iim_tolerance=1E-6
        self.mf_iim_init_control='uniform'
        self.mf_iim_noisy=True #If ture then we print the number of iterations and current magnetisation at each step.


        self.notebook = notebook



    def mf_magnetization(self,h,beta,return_sequence=False) :

        """

        Implements damped fixed point iteration
        for solving the mean-field self consistency
        equations on a weighted graph with general
        external field.

        Parameters
        --------------

        graph : networkx graph

        Weighted networkx graph where the edge weights represent
        the coupling strengths between nodes.

        h : numpy array

        control field acting on the N nodes

        beta : float

        Inverse temperature of the Ising system

        return_sequence : bool (opt)

        If true then we return the sequence of iterations of the
        algorithm. This output can be used to visualize the extent
        of convergence.

        Returns
        -----------

        m : numpy array

        Magnetizations for each of the nodes

        """
        if self.notebook == True :
            from tqdm import tqdm_notebook as tqdm
        else :
            from tqdm import tqdm

        N = len(self.graph)

        if self.mf_fp_init_state == 'aligned':
            m = np.ones(N)  #Initialize at the maximum magnetization.
        else:
            m = np.copy(self.mf_fp_init_state)

        m_sequence = []
        m_sequence.append(np.mean(m))
        for t in (tqdm(range(self.max_mf_fp_iterations)) if self.mf_fp_noisy else range(self.max_mf_fp_iterations)):  # tqdm progress bar used if 'noisy' is set.

            old_m = np.mean(m)

            for i in range(N):

                neighbors_mags = [m[p] for p in list(dict(self.graph[i]).keys())]

                #Networkx graphs may not contain edge weights. Have to check whether they exist.
                edges_to_current_node = [[i, p] for p in list(dict(self.graph[i]).keys())]
                if len(edges_to_current_node) > 0:
                    weights_to_current_node = [len(list(self.graph.get_edge_data(i, j).values())) for i, j in
                                               zip(np.transpose(edges_to_current_node)[0],
                                                   np.transpose(edges_to_current_node)[1])]
                else:
                    weights_to_current_node = [0.0]

                if np.sum(weights_to_current_node) == 0.0:  # Graph with no edge weight data.
                    neighbor_weights = np.asarray([1.0 for p in list(dict(self.graph[i]).keys())])
                else:  #Use edge weights if they exist
                    neighbor_weights = [list(self.graph.get_edge_data(i, j).values())[0] for j in self.graph[i]]

                m[i] = (1.0 - self.gamma) * m[i] + self.gamma * math.tanh(
                    beta * (np.dot(neighbors_mags, neighbor_weights) + h[i]))

            m_sequence.append(np.mean(m))
            difference = abs(np.mean(old_m) - np.mean(m))
            if difference < self.tol:
                if self.mf_fp_noisy == True:
                    print("MF completed with {} iterations and diff ={} m = {}".format(t, difference, np.mean(m)))
                break

        if return_sequence == True:
            return m, m_sequence
        else:
            return m

    def mf_magnetization_gradient(self,m, beta):

        """

        Compute the normalised gradient in the mean-field magnetisation for an
        Ising system with connectivity matrix A and current magnetisation m.

        The stability matrix is defined eq. (18) in:

        Tanaka, Toshiyuki. "Mean-field theory of Boltzmann machine learning." Physical Review E 58.2 (1998): 2302.

        This matrix is the inverse of the susceptibility matrix.

        Parameters
        ----------

        m : numpy array

        Mean-field magnetisation

        beta : float

        Inverse temperature

        Returns
        --------

        gradient : numpy array

        Gradient in the mean-field magnetisation.

        """

        A = nx.to_numpy_matrix(self.graph)

        m_square_inv = [1.0 / (1 - i ** 2) for i in m]  # Fails if entries of m are equal to 1.0
        Stability_Matrix = (1.0/beta) * (np.diag(m_square_inv) - beta * A)

        #The linear solver fails if all elements of m are equal to 1.0 or close to one:
        if all([ math.isclose(i,j,abs_tol=1E-4)  for i,j in zip(m,np.ones(len(m))) ] ) :
            gradient = np.zeros(len(m))
        else :
            gradient = scipy.linalg.solve(Stability_Matrix, np.ones(len(m)))
            gradient = (1.0 / np.linalg.norm(gradient))*gradient #Normalise

        return gradient

    def mf_sparse_magnetisation_gradient(self,m, beta):

        A_sparse = nx.to_scipy_sparse_matrix(self.graph)
        # Infinite entries occur if m=1 for any node.
        # Take inf to be 10**20 to prevent this (may introduce minor error).
        m_square_inv = [ min( 1.0 / (1 - i ** 2) , float(10**20) ) for i in m]
        D_M = sparse.diags(m_square_inv, offsets=0, shape=None, format=None, dtype=None)
        Stability_sparse = (1.0 / beta) * (D_M - beta * A_sparse)

        gradient_sparse = sparse.linalg.spsolve(Stability_sparse, np.ones(len(m)))
        gradient_sparse = (1.0 / np.linalg.norm(gradient_sparse)) * gradient_sparse

        return gradient_sparse

    def mf_IIM(self,beta,Field_Budget,full_record=False,block_sizes=None,sparse=False) :

        """

        Identifies the optimal control field for an Ising system
        under the mean-field approximation using a projected gradient
        ascent algorithm

        Assumes that the p=1 for the p-norm in the budget constraint

        Parameters
        ---------------

        graph : networkx graph

        beta : float

        Field_Budget : float

        Field budget.

        Max_Iterations : int

        Maximum number of iterations before terminating.

        step_size : float

        tolerance : float

        The algorithm terminates if the different between two
        successive magnetization values goes below the tolerance
        value.

        Noise : bool

        If true we print out information on each iteration.


        Returns
        --------------

        control_field :  numpy array

        Proposed control field

        magnetization_values : list

        list of magnetization values as a function of
        the iteration.


        """

        if self.notebook == True :
            from tqdm import tqdm_notebook as tqdm
        else :
            from tqdm import tqdm


        magnetization_values = []

        if self.background_field is None:
            self.background_field = np.zeros(len(self.graph))

        # Store the zero field magnetisation:
        m = self.mf_magnetization( self.background_field, beta)
        if block_sizes is not None :
            weighted_mags = [ (i*j)/np.sum(block_sizes) for i,j in zip(block_sizes,m)]
            magnetization_values.append(np.sum(weighted_mags))
        else :
            magnetization_values.append(np.mean(m))

        if self.mf_iim_init_control == 'uniform':
            if block_sizes is not None :
                #control_field = np.asarray( [Field_Budget/(len(block_sizes)*i) for i in block_sizes] )
                control_field = np.asarray([ Field_Budget/(np.sum(block_sizes)), Field_Budget/(np.sum(block_sizes)) ])
            else :
                control_field = (Field_Budget / len(self.graph)) * np.ones(len(self.graph))
        else:
            control_field = self.mf_iim_init_control

        if full_record == True:
            all_control_fields = []
            all_gradients = []
            all_magnetization_vals=[]

        for t in (tqdm(range(self.max_mf_iim_iterations)) if self.mf_iim_noisy else range(self.max_mf_iim_iterations)):  # tqdm progress bar used if 'noisy' is set.

            m = self.mf_magnetization(control_field + self.background_field, beta)

            if sparse == True :
                gradient = self.mf_sparse_magnetisation_gradient(m,beta)
            else :
                gradient = self.mf_magnetization_gradient(m, beta)

            if block_sizes is not None :
                gradient = np.asarray([(i*j)/np.sum(block_sizes) for i,j in zip(block_sizes,gradient)])

            # print(f"gradient = {gradient}")
            control_field = control_field + self.mf_iim_step_size*gradient
            if block_sizes is not None :
                #block_spread_control = np.asarray([i*j for i,j in zip(block_sizes,control_field)])
                block_spread_control = project_block_to_graph(block_sizes, control_field)
                projected = proj.projection_simplex_sort(block_spread_control, z=Field_Budget)
                control_field= np.asarray( block_level_average(block_sizes, projected) )
            else :
                control_field = proj.projection_simplex_sort(control_field, z=Field_Budget)

            if block_sizes is not None:
                weighted_mags = [ (i*j)/np.sum(block_sizes) for i, j in zip(block_sizes, m)]
                magnetization_values.append(np.sum(weighted_mags))
            else:
                magnetization_values.append(np.mean(m))

            if full_record == True:
                all_control_fields.append(control_field)
                all_gradients.append(gradient)
                all_magnetization_vals.append(m)

            if len(magnetization_values) > 2:
                mag_difference = magnetization_values[-1] - magnetization_values[-2]
                if abs(mag_difference) < self.mf_iim_tolerance:
                    break
                elif mag_difference < 0.0 :
                    print("Failed to converge smoothly - difference in magnetisation -VE. Suggest choosing a different step size.\n{} Iterations completed so far.".format(len(magnetization_values)-1))
                    break

                if self.mf_iim_noisy == True:
                    print("Iteration {} , m_difference = {}".format(t, mag_difference))

        if t == self.max_mf_iim_iterations:
            print("IIM not converged after {} iterations".format(self.max_mf_iim_iterations))

        if full_record == True:
            return control_field, magnetization_values, all_control_fields, all_gradients ,all_magnetization_vals
        else:
            return control_field, magnetization_values