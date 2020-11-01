"""

Code used in the analysis of the IIM problem
at the level of blocks.


Given the parameters for an SBM with K blocks
this code can be used to:
i) Sample SBMs from the ensemble

ii) Compute mean-field controls at the level of the
full graph and at the level of blocks.

iii) Evaluate different influence strategies on the network
using Monte Carlo simulations

Created on: 09/12/19

"""

import networkx as nx
import numpy as np
import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import math
import pdb

#My own modules:
from . import mean_field_IIM
from . import  projection_simplex as proj
from spatial_spin_monte_carlo.spatial_spin_monte_carlo import spins
from spatial_spin_monte_carlo import spatial_spin_monte_carlo as Spins


def split_over_blocks(control, sizes):

    """

    Splits control at the level of blocks over
    the entire graph given knowledge of the block
    sizes.

    Parameters
    -------------

    control : numpy array

    fraction of the total control to apply to
    each block

    sizes : numpy array

    sizes of the blocks in the block structure

    Field_Budget : float

    Total field budget to allocate across the
    blocks.

    """

    full_control = []
    for k in range(len(control)):
        full_control = np.concatenate( (full_control, (control[k]/(sizes[k]) )*np.ones(sizes[k]))  )
    return full_control

def get_ascending_pairs(values) :
    edges = np.concatenate(([0], np.cumsum(values)))
    pairs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return pairs

class block_generator :

    def __init__(self,coupling_matrix,block_sizes) :
        self.coupling_matrix = coupling_matrix
        self.coupling_graph = nx.from_numpy_matrix(coupling_matrix)
        self.block_sizes = block_sizes
        self.N=np.sum(block_sizes)
        self.num_blocks=len(block_sizes)
        self.prob_mat = self.prob_matrix_from_coupling()

    def prob_matrix_from_coupling(self) :
        prob_mat = np.zeros( ( len(self.block_sizes),len(self.block_sizes) ))
        for i in range(self.num_blocks) :
            for j in range(self.num_blocks) :
                if i == j :
                    prob_mat[i][j] = self.coupling_matrix[i][j]/self.block_sizes[i]
                else :
                    prob_mat[i][j] = (self.coupling_matrix[i][j]*self.N)/(2.0*self.block_sizes[i]*self.block_sizes[j])
        return prob_mat

    def make_sbm(self) :
        sbm_graph = nx.stochastic_block_model(self.block_sizes, self.prob_mat)
        return sbm_graph

class block_mf_ising_system(block_generator,mean_field_IIM.mean_field_ising_system) :
    def __init__(self,coupling_matrix,block_sizes,block_background_field,notebook=False) :
        block_generator.__init__(self,coupling_matrix,block_sizes)
        mean_field_IIM.mean_field_ising_system.__init__(self,nx.from_numpy_matrix(coupling_matrix),block_background_field,block_sizes=block_sizes,notebook=notebook)


def project_block_to_graph_for_names(block_sizes, block_level_vals):
    """
    Projects a set of values at the level of
    blocks to the nodes in the graph.
    """
    full_graph_values = []
    for k, n in enumerate(block_sizes):
        current_block = []
        for q in range(n):
            current_block.append(block_level_vals[k] + "_{}".format(q))
        full_graph_values = np.concatenate((full_graph_values, current_block))

    return full_graph_values

class ising_analysis(block_mf_ising_system) :

    """

    Class to explore the properties of
    and ising system on both a full graph
    and block level system.

    """

    def __init__(self,graph,coupling_matrix,block_sizes,block_background_field,block_labels=None,notebook=False) :
        """

        Parameters
        -------------

        graph : networkx graph

        Graph data asssociated with the block structure.

        Note that this might not necessarily be sampled from the ensemble.

        Assume that the node labels are ordered according to the blocks.
        (having helper functions to show that this is expecitly being done could
        be useful).


        """
        block_mf_ising_system.__init__(self,coupling_matrix,block_sizes,block_background_field,notebook=notebook)
        #self.full_graph= nx.convert_node_labels_to_integers(graph) #Relabelling to get spin code to work.
        #self.full_graph = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(graph)) #Relabelling required for spin code to work

        self.full_graph=graph

        if block_labels == None :
            #Defualt to integar labels:
            self.block_labels = ["Block {}".format(p) for p in range(len(block_sizes))]
        else :
            self.block_labels=block_labels

        print("Setting critical temperature... (sparse method)")
        self.beta_c = Spins.crit_beta_sparse(graph)

        full_graph_fields = self.project_block_to_graph( block_background_field)
        full_graph_names = project_block_to_graph_for_names(block_sizes, self.block_labels)

        """
        The set of steps below to make sure field ordering reflects graph ordering.
        If the current node names are not aligned with the block names we will have to make it so.
        If node names match full graph names then this should not have any effect.
        """

        node_name_map={}
        current_labels=list(self.full_graph.nodes())
        for current,new in zip(current_labels,full_graph_names) :
            node_name_map[current]=new
        self.full_graph=nx.relabel_nodes(self.full_graph,node_name_map)


        self.field_val_dict = {}
        for i, j in zip(full_graph_names, full_graph_fields):
            self.field_val_dict[i] = j

        graph_ordered_node_list = list(self.full_graph.nodes())
        re_ordered_field_vals = []
        for node in graph_ordered_node_list :
            try :
                re_ordered_field_vals.append(self.field_val_dict[node])
            except :
                pdb.set_trace()
        self.full_graph_field = np.asarray(re_ordered_field_vals)

        #Now the above is done we can switch back to int labels:
        self.full_graph=nx.convert_node_labels_to_integers(self.full_graph)
        self.block_membership = [ int(p) for p in self.project_block_to_graph(np.arange(0,len(block_sizes),1)) ]

        #Setup a second Ising system class for the full graph:
        self.full_mf_system = mean_field_IIM.mean_field_ising_system(self.full_graph,self.full_graph_field,notebook=notebook)

        #MC magnetization parameters:
        self.eval_initial_state = np.ones(len(self.full_graph))

        #Parameters for MC simulations:
        self.T=0
        self.T_Burn=0
        self.MC_Runs=0

        self.H_sweep_data_fname="H_sweep_data"
        self.H_sweep_diagnostics_fname="H_sweep_diagnostics"

        self.controls_to_get = {'no control':True,
                                'uniform control':True,
                                'NC control':True,
                                'SV control':True,
                                'Block control':True,
                                'Full control':True,
                                'AOB control':True,
                                'Degree control':False}

        #Control parameters:
        self.sv_percentile_threshold = 50


    def set_block_labels(self,block_labels) :
        self.block_labels=block_labels


    def project_block_to_graph(self,block_level_vals) :
        """
        Projects a set of values at the level of
        blocks to the nodes in the graph.
        """
        full_graph_values=[]
        for k,n in enumerate(self.block_sizes):
            current_block=[]
            for q in range(n) :
                current_block.append(block_level_vals[k])
            full_graph_values = np.concatenate((full_graph_values, current_block))
        return full_graph_values


    def block_membership_matrix(self) :
        block_membership_matrix = np.zeros((self.num_blocks, self.N))
        for k in range(self.num_blocks):
            for i in range(self.N):
                if self.block_membership[i] == k:
                    block_membership_matrix[k][i] = 1.0
        return block_membership_matrix


    def block_level_average(self,node_values) :

        """
        Average a quantity defined at the level of nodes
        to the level of blocks.

        Parameters
        -------------

        node_values : list

        Values of a list at the level of nodes


        """

        block_level_average=[]
        size_ascending_pairs = get_ascending_pairs(self.block_sizes)
        for index_pairs in size_ascending_pairs :
            block_level_average.append(np.mean(node_values[index_pairs[0]:index_pairs[1]]))
        return block_level_average

    def split_by_block(self,node_values,labels=None) :

        """
        Splits an array up by block.
        """

        block_arrays = { }
        size_ascending_pairs = get_ascending_pairs(self.block_sizes)
        if labels is None :
            labels = np.arange(0,len(self.block_sizes),1)

        for lab,index_pairs in zip(labels,size_ascending_pairs) :
            block_arrays[lab]= node_values[index_pairs[0]:index_pairs[1]]
        return block_arrays

    def block_level_series_average(self,node_series) :
        """
        Average time series defined at the level of nodes
        stored in an n-d array.
        """

        block_level_series=[]
        size_ascending_pairs = get_ascending_pairs(self.block_sizes)
        for indices in size_ascending_pairs:
            block_level_series.append( [np.mean(k) for k in np.transpose(np.transpose(node_series)[indices[0]:indices[1]])] )
        return block_level_series

    def mc_block_level_magnetizations(self,beta_factor,T,T_Burn,initial_state=None,addition_control=None) :

        """

        Use Monte Carlo simulations to simulate the
        magnetization at the level of blocks.

        """

        beta = beta_factor*self.beta_c

        if addition_control is not None :
            control = self.full_graph_field + addition_control
        else :
            control = self.full_graph_field

        spin_series = Spins.Run_MonteCarlo(self.full_graph, T, beta, T_Burn=T_Burn, positions=None,
                                           Initial_State=initial_state,
                                           control_field=control)

        block_level_mag_series= self.block_level_series_average(spin_series)

        return block_level_mag_series

    def average_magnetization(self,beta_factor,T,T_Burn,MC_Runs,initial_state=None,addition_control=None) :

        mean_mags=[]
        for k in range(MC_Runs) :
            #block_level_mag_series = self.mc_block_level_magnetizations(beta_factor,T,T_Burn,initial_state=initial_state,addition_control=addition_control)
            block_level_mag_series = self.Run_MonteCarlo_Block(T, beta_factor, T_Burn=T_Burn,addition_control=addition_control,sampling_method="Metropolis")
            net_series = [np.mean(k) for k in np.transpose(block_level_mag_series)]
            mean_mags.append(np.mean(net_series))
        return np.mean(mean_mags), stats.sem(mean_mags)

    def multiple_mags(self,beta_factor,T,T_Burn,MC_Runs,initial_state=None,addition_control=None) :

        # Dict to store the mags at the level of blocks:
        block_average_mags = {}
        for index, label in enumerate(self.block_labels):
            block_average_mags[label] = []

        mean_mags = []
        for k in tqdm.tqdm( range(MC_Runs) ) :
            block_level_mag_series = self.Run_MonteCarlo_Block(T, beta_factor, T_Burn=T_Burn,
                                                               addition_control=addition_control,
                                                               sampling_method="Metropolis")
            net_series = [np.mean(k) for k in np.transpose(block_level_mag_series)]
            mean_mags.append(np.mean(net_series))

            for index, label in enumerate(self.block_labels):
                block_average_mags[label] = np.concatenate((block_average_mags[label], [np.mean(block_level_mag_series[index])]))

        block_level_overall_means = {}
        for index, label in enumerate(self.block_labels):
            block_level_overall_means[label] = np.mean(block_average_mags[label])

        return mean_mags , block_level_overall_means


    def make_negative_cancel_control(self,Field_Budget):

        """

        Generates a control field which cancels
        out negative values whilst satisfying the budget
        constraint.

        Assumes that the negative's in the background field
        are greater than the field budget so that it is not
        possible to fully cancel it out.

        If this is not the case then the remaining budget can
        also be spent.

        """

        negative_cancelling_field = []
        for field in self.full_graph_field :
            if field < 0.0:
                negative_cancelling_field.append(-1.0*field)
            else:
                negative_cancelling_field.append(0.0)

        negative_cancelling_field = np.asarray(negative_cancelling_field)
        negative_cancelling_field = (Field_Budget/np.sum(negative_cancelling_field))*np.asarray(negative_cancelling_field)

        return negative_cancelling_field

    def make_swing_voter_control(self,Field_Budget,threshold=0.1):
        """
        This control spreads the budget equally among
        individuals who have a background field less
        than some set threshold.
        """
        swing_voter_field = []

        num_svs = np.sum( [ abs(field) < threshold for field in self.full_graph_field ]  )
        if num_svs == 0 :
            pdb.set_trace()

        for field in self.full_graph_field :
            if threshold > abs(field)  :
                swing_voter_field.append(Field_Budget/float(num_svs))
            else:
                swing_voter_field.append(0.0)

        swing_voter_field = np.asarray(swing_voter_field)

        if np.sum(swing_voter_field) == 0.0 :
            pdb.set_trace()

        return swing_voter_field


    def derive_controls(self,beta_factor,Field_Budget):

        beta = beta_factor*self.beta_c
        derived_controls = { }

        if self.controls_to_get['no control'] == True :
            no_control = np.zeros(len(self.full_graph))
            derived_controls['no control']=no_control
        if self.controls_to_get['uniform control'] == True :
            uniform_control = (Field_Budget / len(self.full_graph)) * np.ones(len(self.full_graph))
            derived_controls['uniform'] = uniform_control
        if self.controls_to_get['NC control'] == True :
            neg_cancel_control = self.make_negative_cancel_control(Field_Budget)
            derived_controls['NC'] = neg_cancel_control
        if self.controls_to_get['SV control'] == True :
            #Set SV threshold based on percentiles:
            sv_percentile = self.sv_percentile_threshold  # Target individuals below this threshold.
            self.sv_absolute_threshold  = np.percentile([abs(p) for p in self.full_graph_field], sv_percentile)
            print("SV threshold = {}".format( self.sv_absolute_threshold) )
            swing_voter_control = self.make_swing_voter_control(Field_Budget, threshold=self.sv_absolute_threshold)
            print("SV control magnitude = {}".format(np.sum(swing_voter_control)))
            derived_controls['SV']=swing_voter_control
        if self.controls_to_get['Block control'] == True :
            #These won't be saved the same as original params:
            #self.mf_iim_init_control=self.block_level_average(neg_cancel_control)
            self.mf_iim_init_control=np.zeros(len(self.block_sizes))

            #block_control , block_optimize_mvals = self.mf_IIM(beta,Field_Budget)
            block_control , block_optimize_mvals = self.mf_IIM(beta,Field_Budget,block_sizes=self.block_sizes)

            block_control_full = split_over_blocks(block_control,self.block_sizes)
            derived_controls['block'] = block_control_full

        if self.controls_to_get['AOB control'] == True :
            #Magnetization based control:
            simulated_block_mags = self.Run_MonteCarlo_Block( self.T, beta_factor, T_Burn=self.T_Burn, addition_control=None, sampling_method="Metropolis")
            simulated_1ms2 = Field_Budget*np.asarray([ (1.0 - np.mean(k)**2) for k in np.transpose(simulated_block_mags)])
            ms2_full_len = split_over_blocks(simulated_1ms2,self.block_sizes)
            ms2_control = proj.projection_simplex_sort(ms2_full_len, z=Field_Budget)
            derived_controls['AOB']=ms2_control

            if abs( np.sum(ms2_control) - np.sum(uniform_control) ) > (10**(-3)) :
                pdb.set_trace()

        if self.controls_to_get['Degree control']==True:
            degrees = list(dict(self.full_graph.degree()).values())
            low_temp_sus = Field_Budget*np.asarray([math.exp(-4.0 * beta_factor *self.beta_c* k) for k in degrees ] )
            degree_based_control = proj.projection_simplex_sort(low_temp_sus, z=Field_Budget)
            derived_controls['Degree control']=degree_based_control

        control_diagnostics = { 'block optimize': block_optimize_mvals}

        if len(self.full_graph) < 3000 and self.controls_to_get['Full control'] == True:
            self.full_mf_system.mf_iim_init_control=uniform_control
            full_control , full_optimize_mvals = self.full_mf_system.mf_IIM(beta,Field_Budget)
            derived_controls['full']=full_control
            control_diagnostics['full optimize']=full_optimize_mvals



        return derived_controls , control_diagnostics


    def evaluate_controls(self, beta_factor, control_dict, T, T_Burn, MC_Runs):

        """
        Evaluate the performance of the different controls
        specified in 'control_dict' on the graph for a given
        value of the beta factor.
        Controls are evaluated by running Monte-Carlo simulations.
        Parameters
        -------------
        graph : networkx graph
        Graph to run MC simulations on
        beta_factor : float
        Multiplier of the critical temperature
        control_dict : dict
        Dictionary with names of controls as keys and
        arrays of the same lenght as the graph as values.
        T : int
        Number of steps to run MC simulations for
        T_Burn : int
        Burn in time for the Monte-Carlo simulations
        MC_Runs : int
        Number of MC runs to average over when computing the magnetisation.
        Returns
        -----------
        control_eval_data : pandas dataframe
        Dataframe containing the controls,their respective magnetisations
        and relevant system parameters.
        """

        control_eval_data = {"T": [T], "T_Burn": [T_Burn], "MC_Runs": [MC_Runs], "beta_factor": [beta_factor]}
        beta = beta_factor*self.beta_c
        for key, control in control_dict.items():
            print("Evaluating {}".format(key))

            mags , block_level_overall_means = self.multiple_mags(beta_factor,T,T_Burn,MC_Runs,initial_state=self.eval_initial_state,addition_control=control)

            Mean=np.mean(mags)
            SE=stats.sem(mags)

            print("\n{} control = {} +/- {}".format(key, Mean, SE))
            print("block level mag means")
            print( block_level_overall_means)
            control_eval_data['M values {}'.format(key)]=[mags]
            control_eval_data['M({})'.format(key)] = Mean
            control_eval_data['M({})_SE'.format(key)] = SE
            control_eval_data['{} control'.format(key)] = [ ' '.join([str(p) for p in control])]

            control_eval_data['{} block level control'.format(key)]=[ ' '.join([str(p) for p in self.block_level_average(control)])]

            for ward in list(block_level_overall_means.keys()) :
                control_eval_data['{} control {} mag'.format(key,ward)]=block_level_overall_means[ward]

        control_eval_data['H'] = round(np.sum(control_dict['uniform']), 2)

        return control_eval_data


    def make_h_sweep_data(self,beta_factor,H_vals) :

        all_control_eval_data = pd.DataFrame()
        all_control_diagnostics = pd.DataFrame()

        for Field_Budget in tqdm.tqdm(H_vals):
            control_dict, control_diagnostics = self.derive_controls(beta_factor, Field_Budget)
            control_eval_data = self.evaluate_controls(beta_factor, control_dict, self.T, self.T_Burn, self.MC_Runs)

            try :
                all_control_eval_data = all_control_eval_data.append(pd.DataFrame(control_eval_data))
            except :
                pdb.set_trace()

            # Make compatible with being a df.
            for k in list(control_diagnostics.keys()):
                control_diagnostics[k] = [[control_diagnostics[k]]]
            control_diagnostics['H'] = Field_Budget
            control_diagnostics['beta_factor']=beta_factor
            all_control_diagnostics = all_control_diagnostics.append(pd.DataFrame(control_diagnostics))

            #Save at each iteration:
            all_control_eval_data.to_csv(self.H_sweep_data_fname+".csv")
            all_control_diagnostics.to_csv(self.H_sweep_diagnostics_fname+".csv")

    def make_h_beta_sweep_data(self,beta_vals,H_vals,spin_alignment) :

        all_control_eval_data = pd.DataFrame()
        all_control_diagnostics = pd.DataFrame()
        for beta_factor in beta_vals :
            for Field_Budget in tqdm.tqdm(H_vals):
                control_dict, control_diagnostics = self.derive_controls(beta_factor, Field_Budget)
                control_eval_data = self.evaluate_controls(beta_factor, control_dict, self.T, self.T_Burn, self.MC_Runs)
                control_eval_data['spin_alignment']=[spin_alignment]
                try:
                    all_control_eval_data = all_control_eval_data.append(pd.DataFrame(control_eval_data))
                except:
                    pdb.set_trace()

                # Make compatible with being a df.
                for k in list(control_diagnostics.keys()):
                    control_diagnostics[k] = [[control_diagnostics[k]]]
                control_diagnostics['H'] = Field_Budget
                control_diagnostics['beta_factor'] = beta_factor
                all_control_diagnostics = all_control_diagnostics.append(pd.DataFrame(control_diagnostics))

                # Save at each iteration:
                all_control_eval_data.to_csv(self.H_sweep_data_fname + ".csv")
                all_control_diagnostics.to_csv(self.H_sweep_diagnostics_fname + ".csv")

    def sample_block_level_phase_diagram(self,beta_vals_block,initial_conditions) :

        """

        Compute the magnetisation using the block level
        mean field approximation for a range of
        different beta values and initial conditions
        in order to contain a phase diagram for the system.

        Parameters
        -------------

        beta_vals_block : numpy array

        list or array of values of the multiplier
        of the critical temperature.

        This is multiplied by the critical temp
        of the full graph, rather than the critical
        temperature associated with the block coupling matrix.

        initial_conditions : list

        list of arrays specifiying initial conditions for the
        iterative procedure.

        Returns
        ------------

        mag_data_block : pandas dataframe

        Data about the magnetisations for each beta
        and each initial condition.

        """
        mag_data_block = pd.DataFrame()

        for beta_f in tqdm.tqdm(beta_vals_block,position=0,leave=True):
            beta = beta_f*self.beta_c

            for index, initial_state in enumerate(initial_conditions):

                data_for_this_init={}

                self.mf_fp_init_state=initial_state
                m_block=self.mf_magnetization(self.background_field,beta)

                for m,lab in zip(m_block,self.block_labels) :
                    data_for_this_init['{}'.format(lab)]=[m]

                data_for_this_init['beta_factor']=[beta_f]
                data_for_this_init['Mean_mag_block']=[np.mean(m_block)]
                data_for_this_init['init_state']=[initial_state]

                df_to_append = pd.DataFrame(data_for_this_init)
                mag_data_block = mag_data_block.append(df_to_append)

        return mag_data_block


    def prob_mat_spring_positions(self,a=0):
        """

        Computes the positions of a spring layout
        using the NxN connection probability matrix
        used to generate the full SBM.

        Useful for generating plots with the blocks
        still together even when many nodes are disconnected

        This function also has the option to mix
        in a certain amount of the full adjacency
        matrix in order to base plot somewhat on actual
        connections.

        """
        block_membership_matrix = self.block_membership_matrix()

        P_tot = np.dot( np.transpose(block_membership_matrix),np.dot(self.prob_mat, block_membership_matrix))
        A = nx.to_numpy_matrix(self.full_graph)

        W = P_tot + a*A

        full_weight_graph = nx.from_numpy_matrix(W)
        print("Computing spring layout...")
        weighted_graph_positions = nx.spring_layout(full_weight_graph)

        return weighted_graph_positions

    def get_block_median_positions(self, position_dict):

        """
        Given a dictionary of positions for nodes in
        the full graph we return the average positions
        of nodes in each of the blocks.
        """

        mean_xs = []
        mean_ys = []
        size_ascending_pairs = get_ascending_pairs(self.block_sizes)

        for indices in size_ascending_pairs :
            these_positions = [ position_dict[list(self.full_graph.nodes())[i]] for i in range(indices[0], indices[1])]
            these_x = np.transpose(these_positions)[0]
            these_y = np.transpose(these_positions)[1]
            mean_xs.append(np.median(these_x))
            mean_ys.append(np.median(these_y))
        return mean_xs, mean_ys

    def plot_config(self) :
        self.node_colors = self.block_membership
        self.node_colors_cmap='Paired'

        #Only compute spring layout for smaller graphs:
        if len(self.full_graph) < 1500 :
            self.weighted_graph_positions = self.prob_mat_spring_positions(a=0.5)

    def add_colorbar(self,node_color) :
        vmin = min(node_color)
        vmax = max(node_color)

        vmin=-1.0*max(abs(vmin),abs(vmax))
        vmax = 1.0 * max(abs(vmin), abs(vmax))
        sm = plt.cm.ScalarMappable(cmap=self.node_colors_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = plt.colorbar(sm, pad=0.075)
        cbar.ax.set_ylabel('External Field', rotation=270,labelpad=12.0,fontsize=18)

        #Set colourbar labels:
        largest_mag=round( max(abs(vmin),abs(vmax)) , 1  )
        cbar.set_ticks([-largest_mag,0.0,largest_mag])
        cbar.set_ticklabels(['{}'.format(-largest_mag), '0.0', '{}'.format(largest_mag)])
        cbar.ax.tick_params(labelsize=16)

    def plot_full_graph(self,block_labels=None,node_colors=None,fname='graph_plot'):

        if node_colors is None :
            node_cols=self.node_colors
        else :
            node_cols=node_colors

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(self.full_graph, self.weighted_graph_positions, node_size=20, node_color=node_cols,cmap=self.node_colors_cmap, alpha=0.9)
        nx.draw_networkx_edges(self.full_graph, self.weighted_graph_positions, width=1.0, alpha=0.5) #Previous 0.5.

        if node_colors is not None :
            self.add_colorbar(node_colors)

        if block_labels is not None :
            x_ward, y_ward = self.get_block_median_positions(self.weighted_graph_positions)
            for i, txt in enumerate(block_labels):
                ax.annotate(txt, (x_ward[i], y_ward[i]), fontsize=15, fontweight='bold')



        plt.axis('off')
        plt.savefig(fname,dpi=100)

    def Run_MonteCarlo_Block(self, T, beta_factor, T_Burn=0,addition_control=None,sampling_method="Metropolis"):

        """

        Samples a sequence of spin states on an Ising system
        for the graph supplied.

        Parameters
        -------------

        graph : networkx graph

        network structure

        T : int

        number of time steps to run the simulation for

        beta : float

        Inverse temperature

        T_Burn : int (opt)

        Burn in time. We run the dynamics for T+T_Burn timesteps
        and only record samples after T_Burn.

        positions : numpy ndarray (optional)

        Positions of nodes. Needs to be in the same order as the
        node list. If this is specified we use the spatial spins
        class.

        Returns
        ------------

        Spin_Series : ndarray (N_Block x T)

        Array continaing time series of spin values for each
        of the blocks.

        """


        spin_system = spins(self.full_graph)

        # Set params:
        spin_system.sampling_method = sampling_method
        spin_system.Beta = beta_factor*self.beta_c

        # Set the initial state:
        spin_system.node_states = np.copy(self.eval_initial_state)


        #Set the field:
        spin_system.field = spin_system.applied_field_to_each_node

        if addition_control is not None :
            spin_system.scalar_field = self.full_graph_field + addition_control
        else :
            spin_system.scalar_field = self.full_graph_field

        current_block_level_mags = []
        #for p in tqdm.tqdm( range(T_Burn + T) , miniters=int(float(T)/10) ) :
        for p in range(T_Burn + T) :
            spin_system.do_spin_flip()
            # Copy the array so it is not a pointer:
            current_state = np.copy(spin_system.node_states)
            if p > T_Burn - 1:
                current_block_level_mags.append( self.block_level_average(current_state) )

        block_mag_series = np.asarray(current_block_level_mags)

        return block_mag_series


    def save_iim_eval_parameters(self,fname)  :

        parameters={}

        #Block level FP iteration and IIM:
        parameters["$\gamma_{\mathrm{block}}$"]=self.gamma
        parameters["$\\tau_{\mathrm{block}}$"] = self.tol
        parameters['$\epsilon_{\mathrm{block}}$'] = self.mf_iim_step_size
        parameters['$a_{\mathrm{block}}$'] = self.mf_iim_tolerance

        #Full graph FP iteration and IIM:
        parameters["$\gamma_{\mathrm{full}}$"] = self.full_mf_system.gamma
        parameters["$\\tau_{\mathrm{full}}$"] = self.full_mf_system.tol
        parameters['$\epsilon_{\mathrm{full}}$'] = self.full_mf_system.mf_iim_step_size
        parameters['$a_{\mathrm{full}}$'] = self.full_mf_system.mf_iim_tolerance

        #MC parameters:
        parameters["$T$"]=int(self.T)
        parameters["$T_{\mathrm{Burn}}$"]=int(self.T_Burn)
        parameters["MC Runs"]=int(self.MC_Runs)

        #Swing voter parameters:
        if self.controls_to_get['SV control'] == True:
            parameters['SV Percentile']=self.sv_percentile_threshold
            parameters['$\sigma$']=self.sv_absolute_threshold

        param_df = pd.DataFrame({'Parameter' : list(parameters.keys()), 'Value' : list(parameters.values())} )

        param_df.to_csv(fname,index=False)




