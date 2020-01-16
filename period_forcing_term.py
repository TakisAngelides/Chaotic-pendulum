import numpy as np
from scipy.integrate import odeint
from numpy import pi
from scipy.constants import g
import matplotlib.pyplot as plt
import pandas as pd

OMEGA = 2.0/3.0 # Driving angular frequency
l = g
omega_0 = 0.0
theta_0 = 0.01
total_time = 1000.0
num_of_time_points = 200000
q = 0.5

def derivatives(y, t, F, q):
    """
    :param y: (list) A 2-tuple list containing theta, omega
    :param t: (float) Time point to evaluate the derivatives
    :param F: (float) Forcing term
    :param q: (float) Damping term
    :return: (list) omega, d(omega)/dt
    """
    return [y[1], -((g/l)*np.sin(y[0])) - q*y[1] + F*np.sin(OMEGA*t)]

# Initial conditions
y_0 = [theta_0, omega_0]
# Generate points in time to evaluate the integration
t = np.linspace(0.0, total_time, num_of_time_points)

# Has length num_of_theta_0_points and contains the period for each theta_0 sample
period = []

def get_y_data(F_value):
    """
    :return: (2-tuple) theta, omega solution from odeint
    """
    y_data = odeint(derivatives, y_0, t, args = (F_value, q))
    theta_list = y_data[:, 0]
    omega_list = y_data[:, 1]

    return theta_list, omega_list

def get_zero_crossing_index_list(theta_local):
    """
    The logic of this function is that each zero crossing gives us information about the period. We run through the
    theta list and whenever theta changes sign we store that index. Then we will use these indexes to create a list
    of the actual times that the crossings occur. A formula is then used on the latter list to find a value for the
    period for each crossing time. Theoretical each crossing should give the same period but this process is discrete,
    inaccurate and that is why we take the average value of the period from all these calculated periods.
    :param theta_local: (list) theta solution from odeint
    :return: (list) The index of the zero crossings in the time list t
    """
    bool_flag = theta_local[0] >= 0
    theta_was_positive = bool_flag
    zero_crossing_index_list_local = []

    for j in range(100000):
        # If theta is negative and was previously positive, then theta changed sign so store it
        if theta_local[j] < 0 and theta_was_positive:
            zero_crossing_index_list_local.append(j)
            theta_was_positive = False
        # If theta is positive and was previously negative, then theta changed sign so store it
        elif theta_local[j] > 0 and not theta_was_positive:
            zero_crossing_index_list_local.append(j)
            theta_was_positive = True
        # If theta is positive/negative and was previously positive/negative, then continue to the next theta value
        else:
            continue

    return zero_crossing_index_list_local

def get_period_average(number_of_zero_crossings_local, time_of_zero_crossing_local):
    """
    Each crossing holds information for the period of the specific theta_0 being tested. The formula used here is
    straight forward and emanates from the accumulation of T/4 durations, where T is the period.
    :param time_of_zero_crossing_local: (list) The times theta crosses 0
    :param number_of_zero_crossings_local: (int) The number of zero crossing index
    :return: (float) Average period from all the periods calculated for each crossing
    """
    period_list = []
    for k in range(number_of_zero_crossings_local):
        odd = 2*k + 1
        period_list.append(time_of_zero_crossing_local[k]*(4/odd))

    return np.average(period_list)

# F terms to plot with period
F_list = [0.5, 0.7, 1.2, 1.38, 1.44, 1.465, 1.61, 1.72, 1.84, 1.91]
# Initialize dataframes to store the theta and omega values for each F
df_theta = pd.DataFrame()
df_omega = pd.DataFrame()

# For each F value to be tested we generate the solution, extract the period and put the solutions in the dataframes
for F in F_list:
    theta, omega = get_y_data(F)
    df_theta[f'F: {F:.3f}'] = theta
    df_omega[f'F: {F:.3f}'] = omega
    get_zero_crossing_index_list(theta)
    zero_crossing_index_list = get_zero_crossing_index_list(theta)
    time_of_zero_crossing = [t[index] for index in zero_crossing_index_list]
    number_of_zero_crossings = len(time_of_zero_crossing)
    period_average = get_period_average(number_of_zero_crossings, time_of_zero_crossing)
    period.append(period_average)

def plot_period_vs_F():
    # Plot period vs F
    plt.scatter(F_list, period, marker = '+', color = 'k')
    plt.plot(F_list, period, color = 'r', linestyle = '--')
    plt.grid(True)
    plt.xlabel('F (Newtons)')
    plt.ylabel('Period (seconds)')
    plt.title(f'Period vs F driving force\n(q = {q}, $\u03B8_0$ = {y_0[0]}, $\u03C9_0$ = {y_0[1]})')
    plt.savefig('period_vs_F.pdf', bbox = 'tight')
    plt.show()

def plot_theta_first_5_F_values():
    # Plot theta vs time for the first 5 F values
    plt.plot(t, df_theta.iloc[:, 0:5])
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([-30, 5])
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    labels = df_theta.columns
    labels_list = [F_val for F_val in labels[0:5]]
    labels_list_to_str = ", ".join(labels_list)
    plt.legend(labels_list, loc = 'lower left')
    plt.title(f'$\u03B8$ vs time\n(' + labels_list_to_str + ')')
    plt.savefig('theta_vs_time_first_5_F.pdf', bbox = 'tight')
    plt.show()

def plot_theta_last_5_F_values():
    # Plot theta vs time for the last 5 F values
    plt.plot(t, df_theta.iloc[:, 5:])
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([-30, 25])
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    labels = df_theta.columns
    labels_list_last = [F_val for F_val in labels[5:]]
    labels_list_last_to_str = ", ".join(labels_list_last)
    plt.title(f'$\u03B8$ vs time\n(' + labels_list_last_to_str + ')')
    plt.legend(labels_list_last, loc='lower left')
    plt.savefig(f'theta_vs_time_last_5_F.pdf', bbox = 'tight')
    plt.show()

def plot_omega_first_5_F_values():
    # Plot omega vs time for the first 5 F values
    plt.plot(t, df_omega.iloc[:, 0:5])
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([-3, 3])
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03C9$ (rad $s^{-1}$)')
    labels = df_omega.columns
    labels_list = [F_val for F_val in labels[0:5]]
    labels_list_to_str = ", ".join(labels_list)
    plt.legend(labels_list, loc = 'lower left')
    plt.title(f'$\u03C9$ vs time\n(' + labels_list_to_str + ')')
    plt.savefig('omega_vs_time_first_5_F.pdf', bbox = 'tight')
    plt.show()

def plot_omega_last_5_F_values():
    # Plot omega vs time for the last 5 F values
    plt.plot(t, df_omega.iloc[:, 5:])
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([-4, 4])
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03C9$ (rad $s^{-1}$)')
    labels = df_omega.columns
    labels_list_last = [F_val for F_val in labels[5:]]
    labels_list_last_to_str = ", ".join(labels_list_last)
    plt.title(f'$\u03C9$ vs time\n(' + labels_list_last_to_str + ')')
    plt.legend(labels_list_last, loc = 'lower left')
    plt.savefig(f'omega_vs_time_last_5_F.pdf', bbox = 'tight')
    plt.show()

# Create a dataframe for F and period
F_df = pd.DataFrame(F_list, columns = ['F (Newtons)'])
period_df = pd.DataFrame(period, columns = ['Period (seconds)'])
df = pd.concat([F_df, period_df], axis = 1)

def generate_period_vs_F_df():
    # Generates a dataframe in html format
    df.to_html('period_vs_F.html')

generate_period_vs_F_df()
plot_period_vs_F()
plot_theta_first_5_F_values()
plot_theta_last_5_F_values()
plot_omega_first_5_F_values()
plot_omega_last_5_F_values()
