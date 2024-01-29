"""This script contains all of the math, ideas, and plotting, behind the principle heatmaps,
mortgage efficiency heatmap, and ownership curves. Running the script as is will generate
a handful of figures as shown in the associated article"""

__author__ = 'MW-OS'
__version__ = 0.1
__date__ = '2024.01'

# standard
import itertools
# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# ---- layout ----
# 1 Principle heatmap
# 2 Efficiency of mortgage terms
# 3 Ownership curves
# 4 Generate all figures


# 1
class PrincipleHeatmap:
    """PrincipleHeatmap Class has all of the tools required to make a heatmap of principles"""

    def __init__(self):
        """create new instance, no action taken"""
        pass

    def __call__(self, *args, **kwargs):
        """the call method passes variables off directly to the wrapper method: heatmap_plot_wrapper"""
        self.heatmap_plot_wrapper(*args, **kwargs)

    @staticmethod
    def principle_calculator(payment, rate, duration):
        """ calculator to back out principle.'payment' is target monthly payment, 'rate' is annual rate (in
        percent), 'duration' is length (in years), all inputs are floats, return is float"""
        return payment * (1 - (1 + (rate / 1.2e3)) ** (-duration * 12)) / (rate / 1.2e3)

    def generate_heatmap_arrays(self, payment_steps, rate_steps, duration):
        """ generate the 3d arrays that are required to plot a heatmap of principles based on rates and payments.
        'rate_steps' is a list or array of each rate, 'payment_steps' is a list or array of each monthly payment,
        'duration is the length of the loan (in years)"""
        # generate the mesh arrays of the rates and monthly payments
        rates_array, payments_array = np.meshgrid(rate_steps, payment_steps)

        # calculate all of the principles into a list
        principle_list = list(map(lambda x: self.principle_calculator(*x),
                                  itertools.product(payment_steps, rate_steps, [duration])))

        # unpack the principle list into an array and return their transposes
        principle_array = np.reshape(principle_list, [len(payment_steps), len(rate_steps)])
        return [principle_array.T, rates_array.T, payments_array.T]

    def plot_heatmap(self, zz, yy, xx, duration, labeled_points=None):
        """takes in the generate_curves values and generates heatmaps
        zz is the principle_array, yy is the rates_array, xx is the payments_array, duration is years of mortgage """

        # make the values for the colorbar
        colobarticks_values = np.arange(1e5, np.round(np.max(zz)/1e6, 2)*1e6, 5e4).tolist()
        # make the string labels for the colorbar
        colobarticks_lables = ['$' + str(int(i / 1e3)) + 'k' if len(str(i)) < 9
                               else '$' + str(np.round(i / 1e6, 2)) + 'm' for
                               i in colobarticks_values]

        # clear existing plots and make new plot
        plt.clf()
        plt.contourf(xx, yy, zz, levels=len(colobarticks_values), cmap='gist_ncar')

        #  get and set axes object properties
        ax_1 = plt.gca()
        # payments ticks on horizontal
        payment_ticks = np.arange(np.min(xx), np.max(xx), 5e2)
        ax_1.set_xticks(payment_ticks)
        ax_1.set_xticklabels(['$' + str(np.round(i/1e3, 1)) + 'k' for i in payment_ticks])
        # rates ticks on vertical
        rate_ticks = np.arange(np.min(yy), np.max(yy), 1).astype(int)
        ax_1.set_yticks(rate_ticks)
        ax_1.set_yticklabels([str(i) + '%' for i in rate_ticks])

        #  get and set plot object properties
        # setting grid
        plt.minorticks_on()
        plt.grid(axis='both', which='major',  color='0.75', linewidth=2)
        plt.grid(axis='both', which='minor',  color='0.45', linewidth=1)
        # colorbar values and labels
        cbar = plt.colorbar(ticks=colobarticks_values, label="Loan amount ($)")
        cbar.ax.set_yticklabels(colobarticks_lables)
        # titles
        plt.title(f"Loan amount as a function of monthly payments and mortgage rates on {duration}-yr fixed mortgages")
        plt.xlabel("Monthly payments ($)")
        plt.ylabel("Mortgage interest rate (%)")

        # if there are labeled points then plot them here
        if labeled_points:
            for i in labeled_points:
                ax_1.text(i[0], i[1], '$' + str(int(self.principle_calculator(i[0], i[1], duration)/1e3)) + 'k',
                          bbox=dict(fc=(1, 1, 1, 0.66), ec="k"))
                plt.plot(i[0], i[1], 'ro')

        # show the plot
        ax_1.text(sum(ax_1.get_xlim())/2, sum(ax_1.get_ylim())/2, 'git:' + __author__,
                  bbox=dict(fc=(1, 0.8, 1, 0.5), ec=(1, 0.8, 1, 1)))
        ax_1.text(sum(ax_1.get_xlim())/4, sum(ax_1.get_ylim())/4, 'git:' + __author__, c='k', alpha=0.11,
                  bbox=dict(fc=(1, 1, 1, 0.11), ec=(1, 1, 1, 0.11)))
        plt.show()

    def heatmap_plot_wrapper(self, max_payment, max_rate, duration, labeled_points=None):
        """the make_heatmap_wrapper function takes in the largest value of monthly payments in max_payment as a float,
        the largest value of mortgage rate in max_rate as a float, and the length (in years) in duration as a float,
        and plots a heatmap of the total principle available for a given rate and monthly payment,
        labeled points is a list of lists of floats for points that have their principle values highlighted"""
        # generate the ranges for all payments and rates
        payment_steps = np.arange(1e3, max_payment + 1e2, 1e1)
        rate_steps = np.arange(0.01, max_rate + 1e-1, 1e-2)
        # calculate the arrays of payments, rates, and resulting principles
        arrays_out = self.generate_heatmap_arrays(payment_steps, rate_steps, duration)
        # plot these arrays in a matplotlib heatmap
        self.plot_heatmap(*arrays_out, duration, labeled_points)


# 2
class MortgageEfficiency:
    """MortgageEfficiency Class has all of the tools required to make a heatmap of mortgage efficiencies"""

    def __init__(self):
        """create new instance, no action taken"""
        pass

    def __call__(self, *args, **kwargs):
        """the call method passes variables off directly to the wrapper method: heatmap_plot_wrapper
        and plots example curves"""
        self.heatmap_plot_wrapper(*args, **kwargs)
        self.plot_mortgage_return_curves(2.5, np.arange(0, 30 + 1, 1/12))
        self.plot_mortgage_return_curves(8, np.arange(0, 30 + 1, 1/12))

    @staticmethod
    def principle_calculator(payment, rate, duration):
        """ calculator to back out principle. 'payment' is target monthly payment, 'rate' is annual rate (in
        percent), 'duration' is length (in years), all inputs are floats, return is float"""
        return payment * (1 - (1 + (rate / 1.2e3)) ** (-duration * 12)) / (rate / 1.2e3)

    def optimal_single_rate_efficiency(self, rate, duration_steps):
        """the single_rate_efficiency method calculates the optimal mortgage efficiency for a specific rate over
        each of the provided durations. rate is a float of the specific rate in question, duration steps
        is a list of the durations (in years) to calculate the mortgage efficiency on. returns the best duration
        as a float (in years)"""
        # calculate the mortgage efficiency for a given rate and return the duration of the best one
        single_mortgage_efficiency = self.single_rate_efficiency(rate, duration_steps)
        optimal_duration = duration_steps[np.argmax(single_mortgage_efficiency)]
        return optimal_duration

    def single_rate_efficiency(self, rate, duration_steps):
        """the single_rate_efficiency method calculates the mortgage efficiency for a specific rate over each of the
        provided durations. rate is a float of the specific rate in question, duration steps is a list of the
        durations (in years) to calculate the mortgage efficiency on. returns a vector of floats from 0 to 1
        that is the length of the duration_steps"""
        # calculate the amount of principle that can be attained for a monthly payment of $1, a given 'rate' and
        # multiple durations in the 'duration_steps' list
        principle_return = self.principle_calculator(1, rate, duration_steps)
        # calculate the efficiency, which shows the trade off of attained principle with the drop off in principle
        efficiency_vector = principle_return * np.r_[0, np.diff(principle_return)]
        # normalize this value by the maximum value to get a scale from 0 to 1
        normalized_efficiency_vector = efficiency_vector/np.max(efficiency_vector)
        return normalized_efficiency_vector

    def plot_heatmap(self, rate_steps, duration_steps):
        """the plot_heatmap method takes in a list of rates and a list of durations and plots the corresponding
        heatmap for the mortgage efficiencies in the specified lists """
        # get the arrays for plotting
        duration_array, rate_array = np.meshgrid(duration_steps, rate_steps)
        efficiency_array = np.array([self.single_rate_efficiency(i, duration_steps) for i in rate_steps])

        # establish tick of z axis
        cb_tick_steps = np.arange(np.round(np.min(efficiency_array), 2), np.round(np.max(efficiency_array), 2), 1e-2)
        # clear any current plots and plot the heatmap
        plt.clf()
        plt.contourf(rate_array, duration_array, 1e2*efficiency_array,
                     levels=int(len(cb_tick_steps)/2), cmap='gist_ncar')
        # set plot object attributes
        plt.title("Mortgage efficiency as a function of interest rate and duration")
        plt.ylabel("Duration (years)")
        plt.xlabel("Mortgage interest rate (%)")
        plt.colorbar(label="Loan versus interest efficiency percentage (%)")
        plt.minorticks_on()
        plt.grid(axis='both', which='major', color='0.75', linewidth=2)
        plt.grid(axis='both', which='minor', color='0.85', linewidth=1)

        # generate and plot the optimal curve
        best_durations = [self.optimal_single_rate_efficiency(i, duration_steps) for i in rate_steps]
        plt.plot(rate_steps, best_durations, color='k', linewidth=2)

        # best duration curve fit function --> # 0.86*ln(x)^4-7.9*ln(x)^3+31.5*ln(x)^2-68.2*ln(x)+69.2
        # best_dur_func = lambda x: 0.86*(np.log(x)**4)-7.9*(np.log(x)**3)+31.5*(np.log(x)**2)-68.2*np.log(x)+69.2
        # plt.plot(rate_steps, [best_dur_func(i) for i in rate_steps], 'r--')

        # show the plot
        ax_2 = plt.gca()
        ax_2.text(sum(ax_2.get_xlim())/2, sum(ax_2.get_ylim())/2, 'git:' + __author__,
                  bbox=dict(fc=(1, 0.8, 1, 0.5), ec=(1, 0.8, 1, 1)))
        ax_2.text(sum(ax_2.get_xlim())/4, sum(ax_2.get_ylim())/4, 'git:' + __author__, c='k', alpha=0.07,
                  bbox=dict(fc=(1, 1, 1, 0.11), ec=(1, 1, 1, 0.11)))
        plt.show()

    def heatmap_plot_wrapper(self, max_rate, max_duration):
        """the make_heatmap_wrapper function takes in the largest value of rate (in percent) in max_rate as a float,
        the largest value of duration (in years) in max_rate as a float, and makes a heatmap of the Mortgage
        Efficiency"""
        # generate the steps
        rate_steps = np.arange(1.0, max_rate + 1e-1, 1e-2)
        duration_steps = np.arange(0, max_duration + 1, 1/12)
        # make heatmap
        self.plot_heatmap(rate_steps, duration_steps)

    def plot_mortgage_return_curves(self, rate, duration_steps):
        """plots examples of the mortgage principle vs its duration derivative for an example 'rate' (float)
        over all of the provided duration floats in the duration_steps list"""
        # get the principle and its derivative for each duration of a given rate
        principles = self.principle_calculator(1, rate, duration_steps)
        principles_diff = np.r_[np.diff(principles)[0], np.diff(principles)]

        # generate subplot objects
        fig, ax_3 = plt.subplots()
        # plot the principle on the left axis
        left_color = 'tab:red'
        plt.title(f"Principle vs Principle derivative for $1 monthly payments for a rate of {rate}%")
        ax_3.set_xlabel('Duration (years)')
        ax_3.set_ylabel('Principle ($)', color=left_color)
        ax_3.plot(duration_steps, principles, color=left_color)
        ax_3.tick_params(axis='y', labelcolor=left_color)

        # plot the mortgage efficiency curve itself, with peak highlighted
        ax_3.plot(duration_steps, principles*principles_diff, 'g--')
        ax_3.plot(duration_steps[np.argmax(principles*principles_diff)],
                  max(principles*principles_diff), 'ko')
        ax_3.legend(['Principle', 'Principle * Principle derivative'])

        # generate a second axis object and plot the principle derivative on the right axis
        ax_4 = ax_3.twinx()
        right_color = 'tab:blue'
        ax_4.set_ylabel('Principle derivative ($)', color=right_color)
        ax_4.plot(duration_steps, principles_diff, color=right_color)
        ax_4.tick_params(axis='y', labelcolor=right_color)
        ax_4.legend(['Principle derivative'])

        # show the plot
        fig.tight_layout()
        ax_3.text(sum(ax_3.get_xlim())/2, sum(ax_3.get_ylim())/2, 'git:' + __author__,
                  bbox=dict(fc=(1, 0.8, 1, 0.5), ec=(1, 0.8, 1, 1)))
        ax_3.text(sum(ax_3.get_xlim())/4, sum(ax_3.get_ylim())/4, 'git:' + __author__, c='k', alpha=0.07,
                  bbox=dict(fc=(1, 1, 1, 0.11), ec=(1, 1, 1, 0.11)))
        plt.show()


# 3
class OwnershipCurves:
    """OwnershipCurves Class has all of the tools required to make plots of all ownership"""

    def __init__(self):
        """create new instance, no action taken"""
        pass

    def __call__(self, *args, **kwargs):
        """the call method passes variables off directly to the plotting method"""
        self.plot_ownership_curves(*args)

    @staticmethod
    def principle_calculator(payment, rate, duration):
        """ calculator to back out principle.
        'payment' is target monthly payment, 'rate' is annual rate (in percent), 'duration' is length (in years),
        all inputs are floats, return is float"""
        return payment * (1 - (1 + (rate / 1.2e3)) ** (-duration * 12)) / (rate / 1.2e3)

    def amortization_calc(self, *args):
        """ calculator to back out interest vs balance paid on loan. args 0 is monthly payments ($) as float,
        args 1 is annual rate (%) in percentage as float, args 2 is years duration in years as float.
        returns the percentage of ownership over time"""
        # initialize the list of cumulative data
        list_out = []
        # generate the remaining principle balance, loop through months of ownership
        balance = self.principle_calculator(args[0], args[1], args[2])
        for i in range(int(args[2]*12)):
            # calculate the results of each monthly step
            interest = balance * args[1]/1.2e3
            principal = args[0] - interest
            balance -= principal
            # append monthly results into the list
            list_out += [[interest, principal, balance]]
        # return percentage of ownership over time
        percent_ownership = 1e2 * np.cumsum(np.array(list_out)[:, 1]) / self.principle_calculator(*args)
        return percent_ownership

    def plot_ownership_curves(self, rate_steps, duration_steps):
        """the plot_ownership_curves method generates the ownership curves for up to three given durations provided
        in a list of floats (in years) and four rates in a list of floats (in percent), and plots them together"""
        # generate all of the ownership percentage curves for each of the rates and durations
        ownership_vectors = [[self.amortization_calc(1, j, i) for j in rate_steps] for i in duration_steps]
        # generate duration vector to plot against on the horizontal axis
        duration_vector = np.linspace(0, max(duration_steps), len(ownership_vectors[-1][0]))

        # loop through all of these and plot them, supports 4 rates and 3 durations
        duration_symbols = ['', ':', '--']
        duration_names = [str(i) for i in duration_steps]
        for i in ownership_vectors:
            rate_colors = ['k', 'r', 'g', 'b']
            rate_names = [str(i) for i in rate_steps]
            tmp_duration_symbol = duration_symbols.pop(0)
            tmp_duration_name = duration_names.pop(0)
            for j in i:
                plt.plot(duration_vector[:len(j)], j, rate_colors.pop(0)+tmp_duration_symbol,
                         label=tmp_duration_name + 'yr at ' + rate_names.pop(0) + '%')
        # add lines, grid, and info
        plt.hlines(25, min(duration_vector), max(duration_vector), 'k')
        plt.hlines(50, min(duration_vector), max(duration_vector), 'k')
        plt.hlines(75, min(duration_vector), max(duration_vector), 'k')
        plt.minorticks_on()
        plt.grid(axis='both', which='major', color='0.75', linewidth=2)
        plt.grid(axis='both', which='minor', color='0.85', linewidth=1)
        plt.legend()
        plt.title(f"Percentage of home owned (accumulated principle) as a function of term duration and rate")
        plt.xlabel("Time since first payment (yr)")
        plt.ylabel("Percentage owned (%)")
        # show the plot
        ax_5 = plt.gca()
        ax_5.text(sum(ax_5.get_xlim())/2, sum(ax_5.get_ylim())/2, 'git:' + __author__,
                  bbox=dict(fc=(1, 0.8, 1, 0.5), ec=(1, 0.8, 1, 1)))
        ax_5.text(sum(ax_5.get_xlim())/4, sum(ax_5.get_ylim())/4, 'git:' + __author__, c='k', alpha=0.11,
                  bbox=dict(fc=(1, 1, 1, 0.11), ec=(1, 1, 1, 0.11)))
        plt.show()


def get_all_article_figures():
    """The get_all_article_figures generates all the figures for the article, takes no arguments and only
    initiates plots, does not return anything. Other method inputs hard coded below"""
    # principle heat maps
    phm = PrincipleHeatmap()
    phm(7e3, 1e1, 30, [[3e3, 8], [3e3, 2], [3e3, 0.1]])
    phm(7e3, 1e1, 20, [[3e3, 8], [3e3, 2], [3e3, 0.1]])
    phm(7e3, 1e1, 15, [[3e3, 8], [3e3, 2], [3e3, 0.1]])

    # mortgage efficiency heatmap and plot
    MortgageEfficiency()(11, 70)

    # plotting of ownership curves
    OwnershipCurves()([2, 4, 6, 8], [15, 20, 30])


if __name__ == '__main__':
    print('running ' + __file__)
    get_all_article_figures()

# eof
