import numpy as np
from scipy.integrate import odeint
from numpy import pi
from scipy.constants import g
import matplotlib.pyplot as plt

OMEGA = 2.0/3.0 # Driving angular frequency
l = g
num_of_theta_0_points = 200
omega_0 = 0.0
total_time = 1000.0
num_of_time_points = 200000
F = 0.0
q = 0.0

def derivatives(y, t, F, q):
    """
    :param y: (list) A 2-tuple list containing theta, omega
    :param t: (float) Time point to evaluate the derivatives
    :param F: (float) Forcing term
    :param q: (float) Damping term
    :return: (list) omega, d(omega)/dt
    """
    return [y[1], -((g/l)*np.sin(y[0])) - q*y[1] + F*np.sin(OMEGA*t)]

# Generate a list of initial displacements to evaluate the period for
theta_0_list = np.linspace(0, pi, num_of_theta_0_points)
# Generate points in time to evaluate the integration
t = np.linspace(0.0, total_time, num_of_time_points)

# Has length num_of_theta_0_points and contains the period for each theta_0 sample
period = []

def get_y_data(theta_0_value):
    """
    :param theta_0_value: (float) Initial displacement in radians
    :return: (2-tuple) theta, omega solution for each time point in the t list
    """
    y_0 = [theta_0_value, omega_0]
    y_data = odeint(derivatives, y_0, t, args = (F, q))
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
    :param number_of_zero_crossings_local: (int) The number of zero crossing index
    :return: (float) Average period from all the periods calculated for each crossing
    """
    period_list = []
    for k in range(number_of_zero_crossings_local):
        odd = 2*k + 1
        period_list.append(time_of_zero_crossing_local[k]*(4/odd))

    return np.average(period_list)

# For each theta_0 sample generate the solution and extract its period
for theta_0 in theta_0_list:

    theta, omega = get_y_data(theta_0)
    zero_crossing_index_list = get_zero_crossing_index_list(theta)
    time_of_zero_crossing = [t[index] for index in zero_crossing_index_list]
    number_of_zero_crossings = len(time_of_zero_crossing)
    period_average = get_period_average(number_of_zero_crossings, time_of_zero_crossing)
    period.append(period_average)

def plot_period_vs_theta_0():
    # Plot period vs theta_0
    plt.plot(theta_0_list, period, color='k')
    plt.xlabel('$\u03B8_0$ (radians)')
    plt.ylabel('Period (seconds)')
    plt.grid(True)
    plt.title(f'Period vs $\u03B8_0$\n(q = {q:.3f}, F = {F:.3f})')
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'period_vs_theta_0_F{F}_q{q}.pdf', bbox='tight')
    plt.show()

plot_period_vs_theta_0()