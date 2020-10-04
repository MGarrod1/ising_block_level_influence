"""

Code used for studying the control of Ising systems
on a two block SBM. Used to:
i) Derive controls and the level of blocks and the
full graph.
ii) Compare the different controls.


Created on: 17/08/19

"""

#Python functions:
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

#My own functions:
from . import mean_field_IIM
from spatial_spin_monte_carlo import spatial_spin_monte_carlo as Spins


def split_over_blocks(control, sizes, Field_Budget):

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

    if Field_Budget > 0.0:
        full_control = []
        for k in range(len(control)):
            full_control = np.concatenate( (full_control, control[k]*np.ones(sizes[k])) )

    else:
        full_control = np.zeros(np.sum(sizes))
    return full_control


class two_block_sbm_analysis :

    """

    """

    def __init__(self,nodes_per_block=50):

        self.N1 = nodes_per_block
        self.N2 = nodes_per_block

        self.sizes = [ self.N1,self.N2 ]
        self.N = np.sum(self.sizes)
        self.N_blocks = len(self.sizes)


        #If block sizes are not equal we need constraints on the coupling matrix?
        self.coupling_matrix = np.asarray( [ [10.0,2.5] , [2.5,2.5]])

        self.coupling_graph = nx.from_numpy_matrix(self.coupling_matrix ) # Create a weighted graph.

        self.block_graph_system = mean_field_IIM.mean_field_ising_system(self.coupling_graph,np.zeros(len(self.sizes)))

    def prob_matrix_from_coupling(self) :

        prob_mat = np.zeros( ( len(self.sizes),len(self.sizes) ))
        for i in range(self.N_blocks) :
            for j in range(self.N_blocks) :

                if i == j :
                    prob_mat[i][j] = self.coupling_matrix[i][j]/self.sizes[i]
                else :
                    prob_mat[i][j] = (self.coupling_matrix[i][j]*self.N)/(2.0*self.sizes[i]*self.sizes[j])
        return prob_mat

    def make_sbm(self) :
        prob_mat = self.prob_matrix_from_coupling()
        sbm_graph = nx.stochastic_block_model(self.sizes, prob_mat)

        return sbm_graph


    def get_critical_temp(self,sbm_graph):
        self.beta_c = Spins.crit_beta(sbm_graph)
        print("beta c = {}".format(self.beta_c))


    def plot_equilibration(self,sbm_graph,T,beta_factor,file_path="two_block/Equilibration") :

        """

        Samples two Metropolis chains and plots
        the magnetization as a function of time.

        Parameters
        ------------

        sbm_graph : networkx graph

        T : int

        Number of timesteps to run the simulations for.

        beta_factor : float

        Multiplier of the critical temeprature.

        file_path : str

        Location to save the file in


        """

        print("Running Metropolis Chains for Equilibration time")
        beta = beta_factor*self.beta_c
        plt.rcParams['figure.figsize'] = [8, 8]

        mag_series = Spins.sample_magnetization_series(sbm_graph, T, beta, positions=None, T_Burn=0, Initial_State=None,
                                                       control_field=None, sampling_method="Metropolis")
        mag_series_2 = Spins.sample_magnetization_series(sbm_graph, T, beta, positions=None, T_Burn=0,
                                                         Initial_State=None, control_field=None,
                                                         sampling_method="Metropolis")

        plt.figure(1)
        plt.clf()
        plt.plot(mag_series, 'b', alpha=0.7)
        plt.plot(mag_series_2, 'r', alpha=0.7)

        plt.xlabel("Timestep", fontsize=20)
        plt.ylabel("Magnetization", fontsize=20)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(file_path)


    def plot_phase_diagram(self,sbm_graph,T,T_Burn,Glauber_Runs,file_path="two_block/phase_diagram",full_initial_state='aligned',block_initial_state='aligned') :

        """

        At a later date may need to
        seperate out data generation
        and plotting. Especially if we
        need to make these scripts HPC
        compatiable for larger versions.
        """

        h_block = np.zeros(self.coupling_matrix.shape[0])
        h_full = np.zeros(len(sbm_graph))

        self.full_graph_system = mean_field_IIM.mean_field_ising_system(sbm_graph,np.zeros(len(sbm_graph)))

        beta_vals = np.arange(0.1, 2.5, 0.1)
        m_block_sweep = []
        m_full_sweep = []
        m_monte_carlo = []
        for beta_f in tqdm(beta_vals):
            beta = beta_f*self.beta_c

            #m_block = mean_field_IIM.mean_field_magnetization_weighted_graph(self.coupling_graph,h_block,beta,initial_state=block_initial_state)
            #m_full = mean_field_IIM.mean_field_magnetization_weighted_graph(sbm_graph,h_full,beta,initial_state=full_initial_state)

            m_block = self.block_graph_system.mf_magnetization(h_block,beta)
            m_full = self.full_graph_system.mf_magnetization(h_full,beta)

            m_block_sweep.append(np.mean(m_block))
            m_full_sweep.append(np.mean(m_full))

            M_mean, M_sem = Spins.sample_magnetization_average(sbm_graph, T, Glauber_Runs, beta, positions=None,
                                                               T_Burn=T_Burn, Initial_State=None, control_field=None)

            m_monte_carlo.append(M_mean)

        plt.figure(1)
        plt.clf()

        plt.plot(beta_vals, m_full_sweep, 'r--', label='Mean field (graph)', lw=2.0)
        plt.plot(beta_vals, m_block_sweep, 'b', label='Mean field (block level)', lw=2.0)
        plt.plot(beta_vals, m_monte_carlo, 'g-.', label="Monte Carlo simulation", lw=2.0)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("$\\frac{\\beta}{\\beta_c}$", fontsize=24)
        plt.ylabel("Magnetization", fontsize=20)
        plt.legend()

        plt.savefig(file_path)


    def full_graph_control(self,sbm_graph,beta_factor ,Field_Budget,Max_Iterations=10000,step_size=50.0) :

        Field_Budget_Full = self.N1*Field_Budget #Valid for equal block sizes.
        beta = beta_factor*self.beta_c

        self.full_graph_system = mean_field_IIM.mean_field_ising_system(sbm_graph, np.zeros(len(sbm_graph)))

        #Do this to make the graph weighted:
        A = nx.to_numpy_matrix(sbm_graph)
        graph_again = nx.from_numpy_matrix(A)
        #full_control, mag_vals = mean_field_IIM.mean_field_IIM(graph_again, beta, Field_Budget_Full, step_size=step_size,Max_Iterations=Max_Iterations)
        full_control, mag_vals =self.full_graph_system.mf_IIM(beta,Field_Budget_Full,full_record=False)

        return full_control , mag_vals


    def block_level_control(self,beta_factor,Field_Budget,step_size=50.0) :

        """

        Performs optimization in order to estimate the optimal
        control under the block level approximation.

        Parameters
        -------------

        beta_factor : float

        factor to multiply the critical temperature by

        Field_Budget : float

        Field budget to use for deriving the control.

        Returns
        ------------

        block_control :  numpy array

        fraction of the control to apply to
        each of the blocks

        mag_vals : list

        series of magnetization values for each
        step in the optimization procedure.

        """

        beta = beta_factor*self.beta_c
        #block_control, mag_vals = mean_field_IIM.mean_field_IIM(self.coupling_graph, beta, Field_Budget, step_size=step_size)
        block_control ,mag_vals = self.block_graph_system.mf_IIM(beta,Field_Budget,full_record=False,block_sizes=None)

        return block_control , mag_vals


    def plot_full_control_on_network(self,sbm_graph,full_control,file_path="two_block/full_control_on_graph",label=None) :



        pos_block_1 = [[np.random.uniform(0, 1), np.random.uniform(0, 0.25)] for k in range(self.N1)]
        pos_block_2 = [[np.random.uniform(1.5, 2.5), np.random.uniform(0, 0.25)] for k in range(self.N2)]
        pos = np.concatenate((pos_block_1, pos_block_2))

        plt.rcParams['figure.figsize'] = [16, 6]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(16, 6)
        nx.draw_networkx_nodes(sbm_graph, pos=pos, node_size=160, alpha=0.95, node_color=full_control, cmap='coolwarm')
        nx.draw_networkx_edges(sbm_graph, pos=pos, alpha=0.2)
        plt.axis('off')

        vmin = min(full_control)
        vmax = max(full_control)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = plt.colorbar(sm)

        if label is not None :
            plt.text(-0.20,0.25, label, fontsize=25)

        cbar.ax.set_ylabel('Control', rotation=270,fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        plt.savefig(file_path,bbox_inches='tight')

    def plot_control_vs_degrees(self,sbm_graph,full_control,block_control,file_path="two_block/control_vs_degrees",label=None) :

        node_degs = list(dict(sbm_graph.degree()).values())


        plt.figure(1,figsize=(6,6))
        plt.clf()
        plt.plot(node_degs, full_control, 'bo',label="Full Graph IIM")
        plt.plot([0.0, max(node_degs)], [block_control[0], block_control[0]], 'g', label="Block 1 control", lw=4.0,
                 alpha=0.6)
        plt.plot([0.0, max(node_degs)], [block_control[1], block_control[1]], 'r', label="Block 2 control", lw=4.0,
                 alpha=0.6)

        plt.plot([12.5, 12.5], [0.0, max(full_control)], 'g--', label="Mean degree B1",lw=3.0,alpha=0.6)
        plt.plot([5.0, 5.0], [0.0, max(full_control)], 'r--', label="Mean degree B2",lw=3.0,alpha=0.6)

        plt.xlabel("Degree", fontsize=16)
        plt.ylabel("Control", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=15,loc='upper right')
        plt.savefig(file_path,bbox_inches='tight')

    def plot_control_histogram(self,full_control,block_control,file_path='two_block/control_hist',label=None) :

        plt.figure(figsize=(6,6))
        n1, bins, hist = plt.hist(full_control[0:self.N1], label='Block 1 full', alpha=0.6, color='g')
        n2, bins, hist = plt.hist(full_control[self.N1:self.N1 + self.N2], label='Block 2 full', alpha=0.5, color='r')

        top_of_plot = max(max(n1), max(n2))

        #Lines for the block controls:
        plt.plot([block_control[0], block_control[0]], [0.0, top_of_plot], 'k', label='Block 1 block', lw=4.0)
        plt.plot([block_control[1], block_control[1]], [0.0, top_of_plot], 'k--', label='Block 2 block', lw=4.0)

        plt.plot([np.mean(full_control[0:self.N1]), np.mean(full_control[0:self.N1])], [0.0, top_of_plot], 'g', lw=2.0)
        plt.plot([np.mean(full_control[self.N1:self.N1 + self.N2]), np.mean(full_control[self.N1:self.N1 + self.N2])], [0.0, top_of_plot], 'r', lw=2.0)

        plt.xlabel("Control", fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14,loc='upper right')
        if label is not None :
            plt.text(-2.1, 105, label, fontsize=25)
        plt.savefig(file_path,bbox_inches='tight')


    def plot_control_MC_comparison(self,beta,sbm_graph,full_control,block_control,uniform_control,file_path='two_block/control_MC_comparison') :


        """

        Plots the results of a single chain of Monte-Carlo
        simulations for each of the controls.

        This plot is useful for diagnostic purposes / determining
        how much noise there is / how much the control has an effect.


        """

        T_Burn = 0
        T = 20000

        block_control_split = split_over_blocks(block_control, [self.N1, self.N2], np.sum(block_control) )

        print("simulating no control...")
        mags_no_control = Spins.sample_magnetization_series(sbm_graph, T, beta, positions=None, T_Burn=T_Burn,
                                                            Initial_State=None, control_field=None,
                                                            sampling_method="Metropolis")

        print("simulating full control...")
        mags_full_control = Spins.sample_magnetization_series(sbm_graph, T, beta, positions=None, T_Burn=T_Burn,
                                                              Initial_State=None, control_field=full_control,
                                                              sampling_method="Metropolis")

        print("simulation block control...")
        mags_block_control = Spins.sample_magnetization_series(sbm_graph, T, beta, positions=None, T_Burn=T_Burn,
                                                               Initial_State=None, control_field=block_control_split,
                                                               sampling_method="Metropolis")

        print("simulating uniform control...")
        mags_uniform_control = Spins.sample_magnetization_series(sbm_graph, T, beta, positions=None, T_Burn=T_Burn,
                                                                 Initial_State=None, control_field=uniform_control,
                                                                 sampling_method="Metropolis")

        plt.rcParams['figure.figsize'] = [8, 8]

        plt.figure(1)
        plt.clf()
        plt.plot(mags_no_control, 'b', label="No control")
        plt.plot(mags_full_control, 'r', label="Mean field control")
        plt.plot(mags_block_control, 'm', label="Blockwise MF")
        plt.plot(mags_uniform_control, 'g', label="Uniform control")
        plt.legend()
        plt.savefig(file_path)


    def make_control_comparison_data(self,beta_factor,sbm_graph,uniform_control,full_control,block_control,add_mean_field_comp=False) :

        T_Burn = 20000  # gets it close to the stationary state.
        T = 10000
        MC_Runs = 25
        H = np.sum(block_control) #field budget is that used for the block control (not quite the same as the full)

        beta = beta_factor*self.beta_c

        block_control_split = split_over_blocks(block_control, [self.N1, self.N2], np.sum(block_control))

        Nocon_Mean, Nocon_SE = Spins.sample_magnetization_average(sbm_graph, T, MC_Runs, beta, positions=None,
                                                                  T_Burn=T_Burn, Initial_State=None, control_field=None)

        print("No control = {} +/- {}".format(Nocon_Mean, Nocon_SE))

        Uniform_Mean, Uniform_SE = Spins.sample_magnetization_average(sbm_graph, T, MC_Runs, beta, positions=None,
                                                                      T_Burn=T_Burn, Initial_State=None,
                                                                      control_field=uniform_control)

        print("Uniform control = {} +/- {}".format(Uniform_Mean, Uniform_SE))

        MF_Mean, MF_SE = Spins.sample_magnetization_average(sbm_graph, T, MC_Runs, beta, positions=None,
                                                            T_Burn=T_Burn, Initial_State=None,
                                                            control_field=full_control)

        print("MF control = {} +/- {}".format(MF_Mean, MF_SE))

        MF_Block_Mean, MF_Block_SE = Spins.sample_magnetization_average(sbm_graph, T, MC_Runs, beta,
                                                                        positions=None, T_Burn=T_Burn,
                                                                        Initial_State=None,
                                                                        control_field=block_control_split)

        print("MF blockwise control = {} +/- {}".format(MF_Block_Mean, MF_Block_SE))



        #Magnetizations under the mean-field control.
        #if add_mean_field_comp == True :




        the_time = datetime.datetime.now()
        #
        data = pd.DataFrame({"Time" : [the_time] , "T" : [T], "T_Burn" : [T_Burn] , "MC_Runs" : [ MC_Runs ] , "M(0)" : [Nocon_Mean] ,
                                "M(0)_SE" : [ Nocon_SE] , "M(unif)" : [Uniform_Mean] , "M(unif)_SE" : [Uniform_SE] ,
                             "M(full)" : [MF_Mean], "M(full)_SE" : [MF_SE] , "M(block)" : [MF_Block_Mean] , "M(block)_SE" : [MF_Block_SE] ,
                             "beta" : [beta],"beta_factor" : [ beta_factor], "H_block" : [H] , "full_control" : [full_control] ,
                             "block_Control" : [block_control]} )


        return data


    def plot_mag_comparison_single_beta_H(self,data,file_path="two_block/single_param_mags_comp") :

        """

        Given a dataframe for a single parameter set
        plot the magnetizations of the different controls

        """

        plt.figure(1)
        plt.clf()

        Nocon_Mean = data[ "M(0)"]
        Nocon_SE = data["M(0)_SE"]

        Uniform_Mean = data[ "M(unif)"]
        Uniform_SE = data["M(full)_SE"]

        MF_Mean = data["M(full)"]
        MF_SE = data["M(full)_SE" ]

        MF_Block_Mean = data["M(block)"]
        MF_Block_SE = data["M(block)_SE"]

        H = list(data["H_block"])[0]
        beta_factor = list(data["beta_factor"])[0]

        all_ms = [Nocon_Mean, Uniform_Mean, MF_Mean, MF_Block_Mean]
        all_ses = [Nocon_SE, Uniform_SE, MF_SE, MF_Block_SE]
        x = [1, 2, 3, 4]
        my_xticks = ['Nocon', 'Unif', 'MF-IIM', 'MF-Block']
        plt.rcParams['figure.figsize'] = [8, 8]
        plt.ylabel("Average magnetization", fontsize=20)
        plt.errorbar(x, all_ms, all_ses, fmt='bo', markersize=10)
        plt.xticks(x, my_xticks, fontsize=18)
        plt.title("H = {} , beta_f = {}".format(H,beta_factor))

        plt.ylim(0.95 * min([float(k) for k in [Uniform_Mean, MF_Mean, MF_Block_Mean]]),
                 1.05 * max([float(k) for k in [Uniform_Mean, MF_Mean, MF_Block_Mean]]))

        plt.savefig(file_path)


    def Field_Budget_Sweep(self,sbm_graph,beta_factor,H_Vals,Max_Iterations=10000) :

        """
        Compute the magnetization markups for a range
        of field budget values.

        Parameters
        --------------

        beta_factor : float

        multiplier of the inverse temperature

        H_Vals : numpy array

        List of values of the field budget to sample for.

        Max_Iterations : int

        Maximum number of iterations to run the mean-field IIM
        using the full graph for.


        Returns
        --------------

        as_h_vals_data : pandas dataframe

        Pandas dataframe containing magnetization values

        """


        as_h_vals_data = pd.DataFrame()

        for Field_Budget in tqdm(H_Vals):
            block_control, mag_vals = self.block_level_control(beta_factor, Field_Budget)
            full_control, mag_vals = self.full_graph_control(sbm_graph, beta_factor, Field_Budget , Max_Iterations=Max_Iterations)
            uniform_control = (self.N1*Field_Budget / len(sbm_graph)) * np.ones(len(sbm_graph))

            data = self.make_control_comparison_data(beta_factor, sbm_graph, uniform_control, full_control,block_control)
            as_h_vals_data = as_h_vals_data.append(data)

            #ADD in mean-field markups for full graph and blockwise control.


        #Now compute the markups:
        full_graph_markup = [(i - j) for j, i in zip(list(as_h_vals_data['M(unif)']), list(as_h_vals_data['M(full)']))]
        block_level_markup = [(i - j) for j, i in
                              zip(list(as_h_vals_data['M(unif)']), list(as_h_vals_data['M(block)']))]

        as_h_vals_data['Detla_M_full'] = full_graph_markup
        as_h_vals_data['Delta_M_block'] = block_level_markup

        return as_h_vals_data


    def plot_markup_as_H(self,as_h_vals_data,file_path="two_block/markup_as_H") :

        """

        Plot the control markup as a function of the field budget
        given the dataframe.

        Parameters
        -----------

        as_h_vals_data : pandas dataframe

        Pandas dataframe containing the

        file_path : str

        path to save the figure to.


        Returns
        ------------

        """

        sns.lineplot(x='H_block', y='Detla_M_full', data=as_h_vals_data, label='full graph')
        sns.lineplot(x='H_block', y='Delta_M_block', data=as_h_vals_data, label='block level')

        sns.scatterplot(x='H_block', y='Detla_M_full', data=as_h_vals_data)
        sns.scatterplot(x='H_block', y='Delta_M_block', data=as_h_vals_data)
        plt.legend()

        plt.savefig(file_path)


if __name__=="__main__" :

    from tqdm import tqdm

    two_block_class = two_block_sbm_analysis()
    sbm_graph = two_block_class.make_sbm()
    two_block_class.get_critical_temp(sbm_graph)

    #Equilbration plot:
    two_block_class.plot_equilibration(sbm_graph,10000,1.5)

    #Hacky change and change back:
    A_SBM = nx.to_numpy_matrix(sbm_graph)
    sbm_graph = nx.from_numpy_matrix(A_SBM)

    #Phase diagram:
    T=10000
    T_Burn=10000
    Glauber_Runs=5
    two_block_class.plot_phase_diagram(sbm_graph,T,T_Burn,Glauber_Runs)


