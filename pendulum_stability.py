import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import g
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi
from scipy.integrate import odeint

OMEGA = 2.0/3.0 # Angular frequency of driving sinusoid
omega_0 = 0.0 # Initial angular velocity
l = g # Lenght of pendulum with g = 9.80665
total_time = 10000 # Total time to evolve the system
num_of_time_points = 200000 # Number of points to evaluate the solution using Runge-Kutta

def derivatives(y, t_val, F, q):
    """
    :param t_val: (float) Time index
    :param q: (float) Damping term
    :param F: (float) Forcing term
    :param y: (list) A 2-tuple of theta, omega
    :return: (list) Omega, d(omega)/dt
    """
    return [y[1], -((g/l)*np.sin(y[0])) - q*y[1] + F*np.sin(OMEGA*t_val)]

# Generate points in time to evaluate the integration
t = np.linspace(0.0, total_time, num_of_time_points)

def get_y_data(theta_0_value, F, q):
    """
    :param F: (float) Forcing term
    :param q: (float) Damping term
    :param theta_0_value: (float) Initial displacement in radians
    :return: (2-tuple) theta, omega solution for each time point in the t list
    """
    y_0 = [theta_0_value, omega_0]
    y_data = odeint(derivatives, y_0, t, args = (F, q))
    theta_list = y_data[:, 0]
    omega_list = y_data[:, 1]

    return theta_list, omega_list

# Gather data for the two initial conditions varying theta_0 vary slightly
theta, omega = get_y_data(theta_0_value = 0.2, F = 5.0, q = 0.0)
theta_per, omega_per = get_y_data(theta_0_value = 0.20001, F = 5.0, q = 0.0)

# Correct the range of theta if needed for plots
# theta = ((theta + pi) % (2*pi)) - pi
# theta_per = ((theta_per + pi) % (2*pi)) - pi

# Create pandas dataframe to contain the solutions and the difference between the two
time_df = pd.DataFrame(t, columns = ['Time (seconds)'])
theta_df = pd.DataFrame(theta, columns = ['Theta (radians)'])
omega_df = pd.DataFrame(omega, columns = ['Omega (rad s^-1)'])
theta_per_df = pd.DataFrame(theta_per, columns = ['Theta perturbed (radians)'])
omega_per_df = pd.DataFrame(omega_per, columns = ['Omega perturbed (rad s^-1)'])
df = pd.concat([time_df, theta_df, theta_per_df, omega_df, omega_per_df], axis = 1)
df['Theta percentage difference (%)'] = (abs((df['Theta (radians)'] - df['Theta perturbed (radians)']))*100)/abs(df['Theta (radians)'])
df['Omega percentage difference (%)'] = (abs((df['Omega (rad s^-1)'] - df['Omega perturbed (rad s^-1)']))*100)/abs(df['Omega (rad s^-1)'])

def plot_theta_vs_time_start():
    # Plot theta vs time for the first 100 seconds
    q, F = 0.5, 1.2
    plt.plot(t, theta, color = 'k')
    plt.plot(t, theta_per, color = 'r')
    plt.title(f'$\u03B8$ vs time for the first 200 seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03C9_0$ = {omega_0:.3f})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    axes = plt.gca()
    axes.set_ylim([-23,23])
    axes.set_xlim([0, 200])
    plt.legend(['$\u03B8_0$ = 0.2', '$\u03B8_0$ = 0.20001'], loc = 'best')
    plt.grid(True)
    # plt.subplots_adjust(left = 0.12)
    plt.savefig(f'theta_vs_time_per_beg_clr.pdf', bbox = 'tight')
    plt.show()

def plot_theta_vs_time_finish():
    # Plot theta vs time for the last seconds
    plt.plot(t[-1500:], theta[-1500:], color = 'k')
    plt.plot(t[-1500:], theta_per[-1500:], color='r')
    q, F = 0.5, 1.2
    plt.title(f'$\u03B8$ vs time for the last seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03C9_0$ = {omega_0:.3f})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    axes = plt.gca()
    plt.legend(['$\u03B8_0$ = 0.2', '$\u03B8_0$ = 0.20001'], loc = 'best')
    plt.grid(True)
    # plt.subplots_adjust(left = 0.19)
    plt.savefig(f'theta_vs_time_per_fin_clr.pdf', bbox = 'tight')
    plt.show()

def plot_omega_vs_theta():
    # Plot omega vs theta for the last seconds
    lim = 800
    plt.scatter(theta[:lim], omega[:lim], color = 'k', s = 0.8, alpha = 1)
    plt.scatter(theta_per[:lim], omega_per[:lim], color='r', s = 0.8, alpha = 1)
    q, F = 0.0, 5.0
    plt.title(f'$\u03C9$ vs $\u03B8$ for the first seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03C9_0$ = {omega_0:.3f})')
    plt.xlabel('$\u03B8$ (radians)')
    plt.ylabel('$\u03C9$ (rad $s^{-1}$)')
    axes = plt.gca()
    plt.legend(['$\u03B8_0$ = 0.2', '$\u03B8_0$ = 0.20001'], loc = 'best')
    plt.grid(True)
    # plt.subplots_adjust(left = 0.19)
    plt.savefig(f'phase_space_omega_vs_theta.pdf', bbox = 'tight')
    plt.show()

def generate_html_dataframe():
    # Generates a dataframe in html format
    df.loc[:20000, :].to_html(f'dataframe_data_pertubation.html')

plot_theta_vs_time_start()
plot_theta_vs_time_finish()
plot_omega_vs_theta()
generate_html_dataframe()