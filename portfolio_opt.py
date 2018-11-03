
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import random
import copy
from timeit import default_timer as timer
import re
import pathlib
import time



class DataSet:
    def __init__(self, file, min_invest = 0.01, max_invest = 1):
        """Creates a random instance. Replace this with an actual instance read in from a text file."""
        #initializations
        self.file = file
        self.epsilon = min_invest
        self.delta = max_invest

        # load and pre-process the datafile
        ds_raw = self.load_dataset(self.file)
        self.N, self.df_asset_details, self.df_correlations, self.df_rho = self.data_preprocessing(ds_raw)
        self.mu =  np.array(self.df_asset_details.ExpReturn.tolist())

        # convert dataframes to numpy arrays
        std = np.array(self.df_asset_details.StDev.tolist())
        correlation = np.array(self.df_rho)
        self.sigma =  np.asarray(correlation * std * std.reshape((std.shape[0],1)))

        if 10 * self.epsilon > 1.0:
            print("Epsilon is too large")
            raise ValueError
        if 10 * self.delta < 1.0:
            print("Delta is too small")
            raise ValueError
        self.F = 1.0 - 10 * self.epsilon


    def load_dataset(self, file):
        """ Loads the raw-unstructured data from the text file"""
        try:
            ds_raw = pd.read_csv(file, header=None)
            return ds_raw
        except:
            print("Oops! Error on file import. Make sure you have a folder named 'datasets' with the assets inside and try again...")

    def data_preprocessing(self, dataset):
        """ Converts the unstructured data the are loaded from a file into structured data"""
        ### number of available assets (N)
        assets_num = int(dataset[0][0])

        ### dataset with details of each available asset (expected return, standard deviation)
        ds_asset_details = dataset.iloc[1:assets_num + 1, 0]
        ds_asset_details = ds_asset_details.str.split(' ', expand=True, n=2)
        ds_asset_details.drop(ds_asset_details.columns[[0]], axis=1, inplace=True)
        ds_asset_details.columns = ['ExpReturn', 'StDev']
        # Convert both columns from string to float
        ds_asset_details['ExpReturn'] = ds_asset_details['ExpReturn'].astype(float)
        ds_asset_details['StDev'] = ds_asset_details['StDev'].astype(float)
        #print('*' * 40)
        #ds_asset_details.info()


        ### dataset with the correlations of each available asset
        ds_correlations = dataset.iloc[assets_num + 1: , 0]
        ds_correlations = ds_correlations.str.split(' ', expand=True, n=3)
        ds_correlations.drop(ds_correlations.columns[[0]], axis=1, inplace=True)
        ds_correlations.columns = ['Asset1', 'Asset2', 'Correlation']
        # Convert both columns from string to int/float
        ds_correlations['Asset1'] = ds_correlations['Asset1'].astype(int)
        ds_correlations['Asset2'] = ds_correlations['Asset2'].astype(int)
        ds_correlations['Correlation'] = ds_correlations['Correlation'].astype(float)

        # convert to correlation matrix (N x N)
        ds_rho = pd.DataFrame(index=range(1, assets_num + 1), columns=range(1, assets_num + 1))
        for i in range(len(ds_correlations)):
            ds_rho.iloc[ds_correlations.iloc[i,0] - 1, ds_correlations.iloc[i,1] - 1] = ds_correlations.iloc[i,2]
            ds_rho.iloc[ds_correlations.iloc[i,1] - 1, ds_correlations.iloc[i,0] - 1] = ds_correlations.iloc[i,2]

        # convert correlation matrix to a numpy array for performace!
        rho = np.array(ds_rho.iloc[0].tolist())
        for i in range(1, len(ds_rho)):
            rho = np.append(rho, ds_rho.iloc[i].tolist(), axis=0)
        rho = rho.reshape((assets_num, assets_num))

        return assets_num, ds_asset_details, ds_correlations, rho



class Candidate:
    def __init__(self, N, K):
        """Creates a random solution"""
        # produce K permutations of numbers between 0 and N
        self.Q = np.random.permutation(N)[:K]
        # produce K random numbers from a uniform distribution over [0, 1)
        self.s = np.random.rand(K)
        self.w = np.zeros(N)
        self.CoVar = np.nan
        self.R = np.nan



class investments():
    """
    This class implements all the functions required for the QUESTION 1
    including:
        - 'Random Search' algorithm
        - Evaluation algorithm (based on the accompanying paper)
        - ..

    Parameters
    ----------
    assets_num : int
        The number of the available assets
    max_invest : float
        The maximum transaction level (max-buy)
    min_invest : float
        The minimum transaction level (min-buy)
    df_asset_details : dataframe
        A dataframe with details of each available asset
        (expected return, standard deviation)
    df_correlations : dataframe
        A dataframe with the correlations of each available asset
    max_evaluations_num : int
        The maximum solution evaluations number for the Random Search algorithm
    E : int
        Number of lamda values

    """

    def __init__(self,
                 dataset,
                 max_invest,
                 min_invest,
                 max_evaluations_num,
                 E
                 ):

        # input parameters
        self.dataset = dataset
        self.max_invest = max_invest
        self.min_invest = min_invest
        self.max_evaluations_num = max_evaluations_num
        self.E = E

        # initializations
        self.df_investments = pd.DataFrame()
        self.best_f = 0
        self.best_investments = pd.DataFrame()
        self.seed = 0
        self.random_seeds = []
        self.H = []
        self.V = []

        #! Produce the investments
        self.invest()



    def evaluate(self, solution, lamda, V, H):
        """solution is a Solution, lamda is an integer index into Lambda and Best_value_found,
        dataset is a DataSet, best_solutions is a list of improved Solution(s)"""

        improved = False
        epsilon = self.dataset.epsilon
        delta = self.dataset.delta

        w = solution.w
        L = solution.s.sum()
        w_temp = epsilon + solution.s * self.dataset.F / L
        is_too_large = (solution.s > delta)
        while is_too_large.sum() > 0:
            R = solution.Q[is_too_large]
            is_not_too_large = np.logical_not(is_too_large)
            L = solution.s[is_not_too_large].sum()
            F_temp = 1.0 - (epsilon * is_not_too_large.sum() + delta * is_too_large.sum())
            w_temp = epsilon + solution.s * F_temp / L
            w_temp[R] = delta
        # Re-init the w values to zero
        w[:] = 0
        # Assign the new values
        w[solution.Q] = w_temp
        solution.s = w_temp - dataset.epsilon

        if np.any(w < 0.0) or not np.isclose(w.sum(), 1) or np.sum(w > 0.0) != 10:
            if np.any(w < 0.0):
                print("There's a negative proportion in solution: " + str(w))
            elif not np.isclose(w.sum(), 1):
                print("Proportions don't sum up to 1 (" + str(w.sum()) + ") in solution: " + str(w))
            else:
                print("More than " + str(10) + " assets selected (" + str(np.sum(w > 0.0)) + ") in solution: " + str(w))
            raise ValueError

        # CoVar = sum of (w * transpose of w * sigma)
        solution.CoVar = np.sum((w * w.reshape((w.shape[0], 1))) * self.dataset.sigma)
        # Even shorter:
        # solution.obj1 = np.sum(np.outer(w, w) * self.dataset.sigma)
        # Return = sum of (w * mu)
        solution.R = np.sum(w * self.dataset.mu)
        f = lamda * solution.CoVar - (1 - lamda) * solution.R
#        solution.s = w[solution.Q] - epsilon
#        w = w[solution.Q]

        if f[0] < V[lamda[0]]:
            improved = True
            V[lamda[0]] = f[0]
            H.append(solution)


        return  solution, solution.R, solution.CoVar, f, improved



    def random_search(self, E):
        """
        This function implements the 'Random Search' algorithm

        This algorithm generates and evaluates  a number of random solutions
        (1000 * assets_num) and returns the best one (lowest f).

        Returns:
        ----------
            best_R : the calculated R value of the best investments

            best_Covar : the calculated CoVar value of the best investments

            best_f : the calculated f value of the best investments

            best_investments : the best investments (s value,weights,amount)
        """

        # initialize with the largest possible number
        # ... (as we need to minimize our objective function f(s))
        best_f = float("inf")

        self.H = []
        self.V = {}
        for e in range(1, E + 1):
            if E == 1:
                lamda = np.array([0.5])
            else:
                lamda = np.array([(e-1)/(E -1)])
                print('--RS_lamda: ', e, '/', E)


            self.V[lamda[0]] = float('inf')
            # counter for the maximum number of evaluations
            counter = 0
            # loop until the maximum solution evaluations number have been reached
            while self.max_evaluations_num > counter:
                # generate a candidate solution/candidate S
                S = Candidate(self.dataset.N, 10)

                # Evaluate Algorithm
                df_investments, R, CoVar, f, improved = self.evaluate(S, lamda, self.V, self.H)

                # keep the investments with the best f of all lamdas
                if f < best_f:
                    best_R = copy.deepcopy(R)
                    best_Covar = copy.deepcopy(CoVar)
                    best_f = copy.deepcopy(f)
                    best_investments = copy.deepcopy(df_investments)


                # increase the counter after each evaluation
                counter += 1
            #print(lamda, " " ,best_R)
        return best_R, best_Covar, best_f, best_investments

    def invest(self):
        """
        This function calls the "Random Search" function with 30 different seeds
        and stores the results (R, Covar, f) produced by each seedv number into
        a dataframe.

        Returns:
        ----------
            df_R_CoVar_results : a dataframe with the R, CoVar and f results of all
                                 30 runs produced by the 30 different seeds.

        """
        # Dataframe that we keep the value of R and CoVar of the best solution returned by each run
        self.df_R_CoVar_results = pd.DataFrame(columns=['R', 'CoVar', 'f'])

        # produce 30 permutations of numbers between 0 and 1000
        self.random_seeds = np.random.permutation(1000)[:30]

        print('--Random Search')
        count_seed = 0
        # Repeat each run with a different initial random seed 30 times
        for seed in self.random_seeds:
            print('Random seed: ', count_seed + 1, '/', len(self.random_seeds))
            #seed = 40   # TO BE DELETED.
            random.seed(seed)
            R, Covar, f, df_investments = self.random_search(self.E)

            # add to the df the R and CoVar of the best solution returned by this run
            self.df_R_CoVar_results.loc[len(self.df_R_CoVar_results)] = [R, Covar, f[0]]
            count_seed += 1

        return self.df_R_CoVar_results


    def report_results_R_CoVar(self, file):
        """
        This function is used to report the results of Question 1, part d)
        """
        print(self.df_R_CoVar_results.describe())
        # Save into a file the results (min,max,mean,etc..) of Random Search
        with open(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q1.txt', 'w') as f:
            print(self.df_R_CoVar_results.describe(), file=f)

    def report_TS_results_R_CoVar(self, file):
        """
        This function is used to report the results of Question 2), part d)
        """
        print(self.df_R_CoVar_results_TS.describe())
        # Save into a file the results (min,max,mean,etc..) of Tabu Search
        with open(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q2_d.txt', 'w') as f:
            print(self.df_R_CoVar_results_TS.describe(), file=f)


class investments_TabuSearch(investments):
    """ This class inherits the Investments class. Implements additional functions so to perform
        the Tabu Search """
    def __init__(self,
                 dataset,
                 max_invest,
                 min_invest,
                 max_evaluations_num,
                 L_star,
                 E,
                 sets_of_assets=[],
                 values_list=[]):

        # input parameters
        self.dataset = dataset
        self.max_invest = max_invest
        self.min_invest = min_invest
        self.max_evaluations_num = max_evaluations_num
        self.L_star = L_star
        self.sets_of_assets=sets_of_assets
        self.values_list = values_list
        self.E = E

        # initializations
        self.df_R_CoVar_results_TS = pd.DataFrame()

        # ! Produce the investments with Tabu Search!
        self.invest_TS()


    def invest_TS(self):

        # Dataframe that we keep the value of R ,CoVar and f of the best solution returned by each run
        self.df_R_CoVar_results_TS = pd.DataFrame(columns=['L', 'R', 'CoVar', 'f'])

        for L in self.L_star:
            print('L = ', L)

            # produce 30 permutations of numbers between 0 and 1000
            self.random_seeds = np.random.permutation(1000)[:30]

            count_seed = 0
            # Repeat each run with a different initial random seed 30 times
            for seed in self.random_seeds:
                print('Random seed: ', count_seed + 1, '/', len(self.random_seeds))

                random.seed(seed)

                start = timer()
                # run the Tabu Search for current random seed
                R, Covar, f, df_investments = self.tabu_search(L, self.E)
                print(timer() - start)


                # add to the df the R and CoVar of the best solution returned by this run
                self.df_R_CoVar_results_TS.loc[len(self.df_R_CoVar_results_TS)] = [L, R, Covar, f[0]]

                #increase the seed counter by one
                count_seed += 1

        return self.df_R_CoVar_results_TS


    def tabu_search(self, L, E):
        """ Tabu Search Algorithm """

        epsilon=self.min_invest
        self.H = []
        self.V = {}

        for e in range(1, E + 1):
            if E == 1:
                # if E=1 we want to check only for lamda = 0.5
                lamda = np.array([0.5])
            else:
                # if E<>1 we want to check for multiple lamdas
                lamda = np.array([(e-1)/(E -1)])
                print('--TS_lamda: ', e, '/', E)

            self.V[lamda[0]] = float('inf')
            for i in range(0, 1000):
                # produce a random solution with 10 assets
                candidate = Candidate(self.dataset.N, 10)
                # evaluate the random solution
                S, R, CoVar, f, improved = self.evaluate(candidate, lamda, self.V, self.H)
                if improved:
                    # keep the best solution found after 1000 iteration into the S_Star
                    S_star = copy.deepcopy(S)

            # initialize (with zero values) tabu values (table Q x m)
            L_im = np.zeros((len(S_star.Q), 2), dtype=np.int)

            # based on the paper, to perform 1000xN evaluations T_Star should be : 500 * N / K
            T_star = int(500 * self.dataset.N / 10)

            for z in range(1, T_star):
                V_dstar=float('inf')  # initialise best neighbour value H
                for i in range(len(S_star.Q)):
                    for m in range(1,3):
                        C=S_star
                        if m==1:
                            C.s[i]= 0.9 * (epsilon + S_star.s[i]) - epsilon
                        else:
                            C.s[i] = 1.1 * (epsilon + S_star.s[i]) - epsilon
                        if C.s[i] < 0:
                            # randomly select an asset j not-in R
                            j = random.choice(list(set(range(0, self.dataset.N))-set(C.Q)))
                            # In the C.Q list (= R), replace the element(asset) in index i with the value j
                            np.put(C.Q, [i], [j])
                            # s[i] same with c[i]  # Substitute the negative element with zero.
                            C.s[i] = 0

                        df_investments, R, CoVar, f, improved = self.evaluate(S_star, lamda, self.V, self.H)
                        if improved:
                            L_im[i][m-1] = 0
                        if L_im[i][m-1] == 0 and f < V_dstar:
                            V_dstar = copy.deepcopy(f)
                            S_dstar = copy.deepcopy(C)
                            k = copy.deepcopy(i)
                            n = copy.deepcopy(m)



                if V_dstar == float('inf'):
                    break  # go to the next lamda
                else:
                    S_star = copy.deepcopy(S_dstar)
                    L_im = L_im - 1  # reduce all tenures by one.
                    L_im[L_im < 0] = 0  # replace negative values with zero
                    if n == 1:
                        opp_n = 2
                    else:
                        opp_n = 1
                    L_im[k][opp_n-1] = L


        # we re-call the evaluate to re-obtain the results of the best solution S*
        best_investments, best_R, best_CoVar, best_f, improved = self.evaluate(S_star, lamda, self.V, self.H)

        return best_R, best_CoVar, best_f, best_investments








def results_comparison(results1, results2, file, results_dir):
    """ Reports the results of the comparison of the Random Search with the Tabu Search """

    results_comp = pd.concat([results1.iloc[:,[0,1]], results2.iloc[:,[0,1]]], axis=1, ignore_index=True)
    results_comp.columns = ['RandomSearch_R', 'RandomSearch_CoVar',
                              'TabuSearch_R',   'TabuSearch_CoVar']
    #print(results_comp, '\n')
    print(results_comp.describe())

    with open(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q2_e.txt', 'w') as f:
        print(results_comp.describe(), file=f)

    return results_comp


def millions(x, pos):
    """ Formats the Revenue values to millions of pounds """
    return '£%1.1fM' % (x*1e-6)


def plot_boxplot_R(random_search_investments, tabu_search_investments, dataset, results_dir):
    """ Boxplot based on the Revenue-Return """

    # take the R value from each run
    ds_randomsearch = random_search_investments.iloc[:, 0] * total_investment
    ds_TabuSearch_L_1 = tabu_search_investments[tabu_search_investments.L == 1].iloc[:,1] * total_investment
    ds_TabuSearch_L_2 = tabu_search_investments[tabu_search_investments.L == 2].iloc[:,1] * total_investment
    ds_TabuSearch_L_5 = tabu_search_investments[tabu_search_investments.L == 5].iloc[:,1] * total_investment
    ds_TabuSearch_L_7 = tabu_search_investments[tabu_search_investments.L == 7].iloc[:,1] * total_investment
    ds_TabuSearch_L_10 = tabu_search_investments[tabu_search_investments.L == 10].iloc[:,1] * total_investment
    ds_TabuSearch_L_15 = tabu_search_investments[tabu_search_investments.L == 15].iloc[:,1] * total_investment

    algorithms = ['Random Search'
                  , 'Tabu Search (L* = 1)'
                  , 'Tabu Search (L* = 2)'
                  , 'Tabu Search (L* = 5)'
                  , 'Tabu Search (L* = 7)'
                  , 'Tabu Search (L* = 10)'
                  , 'Tabu Search (L* = 15)']
    data = [ds_randomsearch
            , ds_TabuSearch_L_1
            , ds_TabuSearch_L_2
            , ds_TabuSearch_L_5
            , ds_TabuSearch_L_7
            , ds_TabuSearch_L_10
            , ds_TabuSearch_L_15]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.set_axisbelow(True)
    ax1.set_title(' Boxplot for the Revenue Values for: ' + str(dataset), fontsize=14)
    ax1.set_ylabel('Revenue Values', fontsize=12)


    # Set the axes ranges and axes labels
    #ax1.set_xlim(0.5, 12 + 0.5)
    #top = 40
    #bottom = 0
    #ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(algorithms, rotation=55, fontsize=12)
    formatter = FuncFormatter(millions)
    ax1.yaxis.set_major_formatter(formatter)
    #plt.show()

    # save the plot
    filename = results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q3_R' + '.png'
    plt.savefig(filename)

    return True

def plot_boxplot_f(random_search_investments, tabu_search_investments, dataset, results_dir):
    """ Boxplot based on the f value"""

    # take the f value from each run
    ds_randomsearch = random_search_investments.iloc[:, 2]
    ds_TabuSearch_L_1 = tabu_search_investments[tabu_search_investments.L == 1].iloc[:,3]
    ds_TabuSearch_L_2 = tabu_search_investments[tabu_search_investments.L == 2].iloc[:,3]
    ds_TabuSearch_L_5 = tabu_search_investments[tabu_search_investments.L == 5].iloc[:,3]
    ds_TabuSearch_L_7 = tabu_search_investments[tabu_search_investments.L == 7].iloc[:,3]
    ds_TabuSearch_L_10 = tabu_search_investments[tabu_search_investments.L == 10].iloc[:,3]
    ds_TabuSearch_L_15 = tabu_search_investments[tabu_search_investments.L == 15].iloc[:,3]

    algorithms = ['Random Search'
                  , 'Tabu Search (L* = 1)'
                  , 'Tabu Search (L* = 2)'
                  , 'Tabu Search (L* = 5)'
                  , 'Tabu Search (L* = 7)'
                  , 'Tabu Search (L* = 10)'
                  , 'Tabu Search (L* = 15)']
    data = [ds_randomsearch
            , ds_TabuSearch_L_1
            , ds_TabuSearch_L_2
            , ds_TabuSearch_L_5
            , ds_TabuSearch_L_7
            , ds_TabuSearch_L_10
            , ds_TabuSearch_L_15]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.set_axisbelow(True)
    ax1.set_title(' Boxplot for the f Values for: ' + str(dataset), fontsize=14)
    ax1.set_ylabel('f Values', fontsize=12)
    ax1.set_xticklabels(algorithms, rotation=55, fontsize=12)
    #plt.show()

    # save the plot
    filename = results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q3_f' + '.png'
    plt.savefig(filename)

    return True


def efficientfrontier(L_star, E, cls_RandomSearch, cls_TabuSearch):
    """ Produces the efficient frontiers of both Random Search and Tabu Search results """

    ############ Random Search ############
    cls_RandomSearch.H = [] # re-initialize lists
    # Run RandomSearch for E = 50
    R, Covar, f, df_investments = cls_RandomSearch.random_search(E)

    # Find the dominated efficient solutions
    dominated_points_RS, dominating_RS = dominated(cls_RandomSearch.H, 'RS')
    # dominating points
    x_dominating_CoVar = dominating_RS[:,1]
    y_dominating_Return = dominating_RS[:,0]
    # dominated points
    x_dominated_CoVar = dominated_points_RS[:,1]
    y_dominated_Return = dominated_points_RS[:,0]

    # plot Tabu Search efficient frontier
    fig = plt.figure(figsize=(8,6))
    plt.plot(x_dominated_CoVar, y_dominated_Return, 'o', markersize=5, label='Available Portfolio')
    plt.plot(x_dominating_CoVar, y_dominating_Return, 'y-o', color='orange', markersize=8, label='Efficient Frontier')
    plt.xlabel('Risk - Variance', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.title(' Efficient Frontier (Random Search)', fontsize=14)
    plt.legend(loc='best')
    plt.show()

    # save the plot
    filename = results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q4_' + '_Frontier_RS'  + '.png'
    fig.savefig(filename)

    ############ Tabu Search ############
    cls_TabuSearch.H = [] # re-initialize lists
    # Run RandomSearch for E = 50
    R, Covar, f, df_investments = cls_TabuSearch.tabu_search(L_star, E)

    # Find the dominated efficient solutions
    dominated_points_TS, dominating_TS = dominated(cls_TabuSearch.H, 'TS')
    # dominating points
    x_dominating_CoVar = dominating_TS[:,1]
    y_dominating_Return = dominating_TS[:,0]
    # dominated points
    x_dominated_CoVar = dominated_points_TS[:,1]
    y_dominated_Return = dominated_points_TS[:,0]

    # plot Tabu Search efficient frontier
    fig = plt.figure(figsize=(8,6))
    plt.plot(x_dominated_CoVar, y_dominated_Return, 'o', markersize=0.7, label='Available Portfolio')
    plt.plot(x_dominating_CoVar, y_dominating_Return, 'y-o', color='orange', markersize=3, label='Efficient Frontier')
    plt.xlabel('Risk - Variance', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.title(' Efficient Frontier (Tabu Search)', fontsize=14)
    plt.legend(loc='best')
    plt.show()

    # save the plot
    filename = results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q4_' + '_Frontier_TS' + '.png'
    fig.savefig(filename)

    return True


def dominated(H, alg):
    """ Filters the non-dominated solutions from all the improved solutions("H") """

    # add into a numpy array ("points") the unique points of the H results
    # ... and into another numpy array ("assets") the set of assets that we can obtain this Revenue-Return
    points = np.array([]).reshape((0, 2))
    assets = np.array([]).reshape((0, 10))
    weights= np.array([]).reshape((0, 10))
    for solution in H:
        R = solution.R
        CoVar = solution.CoVar
        Q = np.sort(solution.Q)
        if [R, CoVar] not in points:
            # if row doesn't already exist, then append
            points = np.append(points, [[R, CoVar]], axis = 0)
            assets = np.append(assets, [Q], axis = 0)
            weights = np.append(weights, [solution.w[np.nonzero(solution.w)]], axis = 0)

    # sort assets, points and weights based on the Return-Revenue
    assets = assets[points[:,0].argsort()]
    points = points[points[:,0].argsort()]
    weights=weights[points[:,0].argsort()]

    # keep in a list the indexes with the DOMINATED points
    index_to_delete = []
    for i in range(len(points)):
        for j in range(len(points)):
            if points[j, 0] != points[i, 0] and points[j, 1] != points[i, 1]: # i != j (not same line!)
                if points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1]: # if Rc > Rd and COVARc < COVARd
                    index_to_delete.append(i)
                    break
    dominated_points = points[index_to_delete]  # points that are about to be deleted are the dominated points
    dominating_points = np.delete(points, index_to_delete, axis=0)  # delete the dominated points
    assets = np.delete(assets, index_to_delete, axis=0)  # delete the set of assets of the the dominated points
    weights = np.delete(weights, index_to_delete, axis=0)  # delete the set of weights of the the dominated points

    # Export the H (dominated_points) results to an Excel File
    pd.DataFrame(dominated_points).to_excel(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q4_' + 'H_' + alg +'.xlsx', index=False)
    # Export the H (dominating_points) results to an Excel File
    pd.DataFrame(dominating_points).to_excel(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q4_' + 'H_' + alg + '_filtered.xlsx', index=False)
    # Export the assets to invest (assets), which are sorted based on the Revenue, to an Excel File
    pd.DataFrame(assets).to_excel(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q4_' + 'AssetsToInvest_' + alg + '.xlsx', index=False)
    # Export the weights to invest to each asset, which are sorted based on the Revenue, to an Excel File
    pd.DataFrame(weights).to_excel(results_dir + re.sub('[^a-zA-Z0-9 \n\.]', '_', file[:-4]) + '_Q4_' + 'WeightToInvest_' + alg + '.xlsx', index=False)

    return dominated_points, dominating_points


def market_comparison_plot():
    """ Plots all 5 efficient frontiers of the Tabu Search algorithm in one graphical presentation after importing
        the results from Excel of each market  """
    df_assets1 = pd.read_excel('markets/assets1.xlsx')
    df_assets2 = pd.read_excel('markets/assets2.xlsx')
    df_assets3 = pd.read_excel('markets/assets3.xlsx')
    df_assets4 = pd.read_excel('markets/assets4.xlsx')
    df_assets5 = pd.read_excel('markets/assets5.xlsx')

    x_asset1_CoVar = df_assets1.iloc[:,1].tolist()
    y_asset1_Return = df_assets1.iloc[:,0].tolist()

    x_asset2_CoVar = df_assets2.iloc[:,1].tolist()
    y_asset2_Return = df_assets2.iloc[:,0].tolist()

    x_asset3_CoVar = df_assets3.iloc[:,1].tolist()
    y_asset3_Return = df_assets3.iloc[:,0].tolist()

    x_asset4_CoVar = df_assets4.iloc[:,1].tolist()
    y_asset4_Return = df_assets4.iloc[:,0].tolist()

    x_asset5_CoVar = df_assets5.iloc[:,1].tolist()
    y_asset5_Return = df_assets5.iloc[:,0].tolist()

    fig = plt.figure(figsize=(8,8))
    plt.plot(x_asset1_CoVar, y_asset1_Return, color='orange', markersize=0.7, label='(1) Hang Seng ')
    plt.plot(x_asset2_CoVar, y_asset2_Return, color='g', markersize=0.7, label='(2) DAX ')
    plt.plot(x_asset3_CoVar, y_asset3_Return, color='k', markersize=0.7, label='(3) FTSE')
    plt.plot(x_asset4_CoVar, y_asset4_Return, color='c', markersize=0.7, label='(4) S&P')
    plt.plot(x_asset5_CoVar, y_asset5_Return, 'y-o', color='C3', markersize=0.7, label='(5) Nikkei')
    plt.xlabel('Risk - Variance', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.title(' Efficient Frontier per Market', fontsize=14)
    plt.legend(loc='best')
    plt.show()

##############################################################################
total_investment = 1000000000  # 1 Billion
min_invest = 0.01              # min-buy (epsilon)
max_invest = 1                 # max-buy (delta)


# create a list with all five datasets
ls_files = [ 'datasets/assets1.txt'
            ,'datasets/assets2.txt'
            ,'datasets/assets3.txt'
            ,'datasets/assets4.txt'
            #,'datasets/assets5.txt'
            ]

# create a folder in the current directory to store the image results
results_dir = 'results_' + time.strftime("%Y_%m_%d-%H%M") + '\\'
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)


# for all 5 datasets (asset 1-5)
for file in ls_files:


    ####################### DATA LOADING  #######################
    print('*' * 50, '\n \t File: ', file, '\n')
    dataset = DataSet(file, min_invest, max_invest)



    ##################### Initializations ######################
    max_evals = 1000 * dataset.N
    E = 1


    #######################  Question 1  #######################
    # Question 1, part a) b) c)
    investments_results = investments(dataset,
                                      max_invest,
                                      min_invest,
                                      max_evals,
                                      E)

    # Question 1, part d)+
    investments_results.report_results_R_CoVar(file)




    #######################  Question 2  #######################
    # Question 2, part a) b) c)
    L_star=[7]
    investments_TabuSearch_results = investments_TabuSearch(dataset,
                                                           max_invest,
                                                           min_invest,
                                                           max_evals,
                                                           L_star,
                                                           E)
    # Question 2, part d)
    investments_TabuSearch_results.report_TS_results_R_CoVar(file)

    # Question 2, part e)
    results_comparison(investments_results.df_R_CoVar_results,
                       investments_TabuSearch_results.df_R_CoVar_results_TS[investments_TabuSearch_results.df_R_CoVar_results_TS.L == 7].iloc[:,1:3],
                       file, results_dir)




    #######################  Question 3  #######################
    # Question 3, part a)
    L_star=[1,2,5,7,10,15]
    investments_TabuSearch_results = investments_TabuSearch(dataset,
                                                           max_invest,
                                                           min_invest,
                                                           max_evals,
                                                           L_star,
                                                           E)

    #Question 3, part b)
    plot_boxplot_R(investments_results.df_R_CoVar_results,
                 investments_TabuSearch_results.df_R_CoVar_results_TS,
                 file, results_dir)

    plot_boxplot_f(investments_results.df_R_CoVar_results,
                 investments_TabuSearch_results.df_R_CoVar_results_TS,
                 file, results_dir)


    #######################  Question 4  #######################
    # Question 4, part a)
    E = 50

    # for each dataset we give the BEST L_star value
    if file == 'datasets/assets1.txt':
        L_star = 5
    elif file == 'datasets/assets2.txt':
        L_star = 7
    elif file == 'datasets/assets3.txt':
        L_star = 7
    elif file == 'datasets/assets4.txt':
        L_star = 7
    elif file == 'datasets/assets5.txt':
        L_star = 10

    efficientfrontier(L_star, E, investments_results,
                      investments_TabuSearch_results)



# create a plot that contains all 5 efficient frontiers into one chart
market_comparison_plot()





