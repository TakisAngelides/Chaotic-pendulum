import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import g
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi

alpha = 0.0 # Damping coefficient
m = 1.0 # Mass
l = g # Lenght of pendulum with g = 9.80665
q = alpha/(m*l)
G = 0.0 # Coefficient of driving term
F = G/(m*l**2)
OMEGA = 2.0/3.0 # Angular frequency of driving sinusoid
theta_0 = 0.01 # Initial displacement
omega_0 = 0.0 # Initial angular velocity
small_angle_undriven_undamped_period = 2.0*pi
num_of_oscillations = 1000.0 # Number of undamped undriven oscillations
total_time = num_of_oscillations * small_angle_undriven_undamped_period # Total time to evolve the system
num_of_time_points = 200000 # Number of points to evaluate the solution using Runge-Kutta
q = 0.0
F = 0.0

# Override constants if user wishes
while True:
   try:
       q = float(input(f'Provide a value for q which controls the damping coefficient\n(default is set to {q}): '))
   except ValueError:
       print('Input was not a number, please provide a number for q in the range 0.0 to 10.0')
   else:
       if 0.0 <= q <= 10.0:
           break
       else:
           print('Please enter a value for q in the range 0.0 to 10.0')

while True:
   try:
       F = float(input(f'Provide a value for F which controls the forcing term\n(default is set to {F}): '))
   except ValueError:
       print('Input was not a number, please provide a number for F in the range 0.0 to 10.0')
   else:
       if 0.0 <= F <= 10.0:
           break
       else:
           print('Please enter a value for F in the range 0.0 to 10.0')

while True:
   try:
       theta_0 = float(input(f'Provide a value for the initial displacement theta_0\n(default is set to {theta_0}): '))
   except ValueError:
       print('Input was not a number, please provide a number for theta_0 in the range -10.0 to 10.0')
   else:
       if -10.0 <= theta_0 <= 10.0:
           break
       else:
           print('Please enter a value for theta_0 in the range -10.0 to 10.0')

while True:
   try:
       omega_0 = float(input(f'Provide a value for the initial angular velocity omega_0\n(default is set to {omega_0}): '))
   except ValueError:
       print('Input was not a number, please provide a number for omega_0 in the range -10.0 to 10.0')
   else:
       if -10.0 <= omega_0 <= 10.0:
           break
       else:
           print('Please enter a value for omega_0 in the range -10.0 to 10.0')

def derivatives(t, y):
    """
    :param t: (float) Time index
    :param y: (list) A 2-tuple of theta, omega
    :return: (list) Omega, d(omega)/dt
    """
    # y[0] is theta and y[1] is d(theta)/dt in this vectorised implementation
    return [y[1], -((g/l)*np.sin(y[0])) - q*y[1] + F*np.sin(OMEGA*t)]

def analytical_undamped_undriven(t_list, initial_conditions):
    """
    :param t_list: (list) The time index values to evaluate the analytical function
    :param initial_conditions: (list) Initial displacement theta_0 and angular velocity omega_0
    :return: (list) Theta evaluated at the times in the t_list for the analytical solution
    """
    initial_displacement = initial_conditions[0]
    initial_angular_velocity = initial_conditions[1]
    B = initial_displacement
    A = initial_angular_velocity*np.sqrt(l/g)
    f = A*np.sin(np.sqrt(g/l)*t_list) + B*np.cos(np.sqrt(g/l)*t_list)
    return f

# Initial conditions
y_0 = [theta_0, omega_0]
# Generate points in time to evaluate the integration
t = np.linspace(0.0, total_time, num_of_time_points)

# Analytical solution for the undamped, undriven case with dataframe created as well
analytical_undamped_undriven_theta = analytical_undamped_undriven(t, y_0)
analytical_undamped_undriven_theta_df = pd.DataFrame(analytical_undamped_undriven_theta, columns = ['Analytical theta\n(radians)'])

# Integrate the coupled 1st order ODE to get theta and d(theta)/dt with their time index using Runge-Kutta
data = solve_ivp(derivatives, t_span = (t[0], t[-1]), method = 'RK45', y0 = y_0, t_eval = t, vectorized = True, max_step = (total_time/num_of_time_points))

# Time, Theta, Omega, Energy - (undriven, undamped) arrays
t_array = np.array(data.t)
theta_array = np.array(data.y[0])
omega_array = np.array(data.y[1])
energy_array = m*g*l*(1-np.cos(theta_array)) + 0.5*m*(l**2)*omega_array**2

# Time, Theta, Omega, Energy dataframes
t_array_df = pd.DataFrame(t_array, columns = ['Time (seconds)'])
theta_array_df = pd.DataFrame(theta_array, columns = ['Theta (radians)'])
omega_array_df = pd.DataFrame(omega_array, columns = ['Omega (rad s^(-1))'])
energy_array_df = pd.DataFrame(energy_array, columns = ['Energy (Joules)'])
df = pd.concat([t_array_df, theta_array_df, analytical_undamped_undriven_theta_df, omega_array_df, energy_array_df], axis = 1)
pd.set_option('display.max_rows', len(df))

def generate_html_dataframe():
    # Generates a dataframe in html format
    df.to_html(f'dataframe_data_osc_{num_of_oscillations}_F{F}_theta0_{theta_0}.html')

def plot_omega_vs_time():
    # Plot omega vs time for the first 100 seconds
    plt.plot(t_array, omega_array, color = 'k')
    plt.title(f'$\u03C9$ vs time for the first 100 seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03B8_0$ = {theta_0:.3f}, $\u03C9_0$ = {omega_0:.3f})\nNumber of oscillations = {num_of_oscillations}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03C9$ (rad $s^{-1})$')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    plt.subplots_adjust(left = 0.2, top = 0.85)
    plt.legend(['$\u03C9$ from Runge-Kutta'], loc='best')
    plt.grid(True)
    plt.savefig(f'omega_vs_time_beg_osc_{num_of_oscillations}_F{F}_q{q}.pdf', bbox='tight')
    plt.show()

def plot_theta_vs_time_start_undamped_undriven():
    # Plot theta vs time for the first 100 seconds
    plt.plot(t_array, theta_array, color = 'r')
    plt.plot(t_array, analytical_undamped_undriven_theta, linestyle = '--', color = 'k')
    plt.title(f'$\u03B8$ vs time for the first 100 seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03B8_0$ = {theta_0:.3f}, $\u03C9_0$ = {omega_0:.3f})\nNumber of oscillation = {num_of_oscillations}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    plt.legend(['$\u03B8$ from Runge-Kutta', 'Analytical solution'], loc='lower center')
    plt.grid(True)
    plt.subplots_adjust(left = 0.16, top = 0.85)
    plt.savefig(f'theta_vs_time_beg_osc_{num_of_oscillations}.pdf', bbox='tight')
    plt.show()

def plot_theta_vs_time_finish_undamped_undriven():
    # Plot theta vs time for the last seconds
    plt.plot(t_array[-500:], theta_array[-500:], color = 'r')
    plt.plot(t_array[-500:], analytical_undamped_undriven_theta[-500:], linestyle = '--', color = 'k')
    plt.title(f'$\u03B8$ vs time for the last seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03B8_0$ = {theta_0:.3f}, $\u03C9_0$ = {omega_0:.3f})\nNumber of oscillation = {num_of_oscillations}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    axes = plt.gca()
    plt.legend(['$\u03B8$ from Runge-Kutta', 'Analytical solution'], loc = 'lower center')
    plt.grid(True)
    plt.subplots_adjust(left=0.16, top=0.85)
    plt.savefig(f'theta_vs_time_fin_osc_{num_of_oscillations}.pdf', bbox='tight')
    plt.show()

def plot_theta_vs_time_start():
    # Plot theta vs time for the first 100 seconds
    # plt.axhline(y = 2*pi, color = 'b', label = '2$\u03C0$')
    # plt.axhline(y=-2 * pi, color='b', label='-2$\u03C0$')
    plt.plot(t_array, theta_array, color = 'k')
    plt.title(f'$\u03B8$ vs time for the first 100 seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03B8_0$ = {theta_0:.4f}, $\u03C9_0$ = {omega_0:.3f})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([-70, 5])
    plt.legend(['$\u03B8$ from Runge-Kutta', 'Analytical solution'], loc = 'best')
    plt.grid(True)
    # plt.subplots_adjust(left = 0.12)
    plt.savefig(f'theta_vs_time_beg_osc_{num_of_oscillations}_F{F}_theta0_{theta_0}_q_{q}.pdf', bbox = 'tight')
    plt.show()

def plot_theta_vs_time_finish():
    # Plot theta vs time for the last seconds
    plt.plot(t_array[-1500:], theta_array[-1500:], color = 'k')
    plt.title(f'$\u03B8$ vs time for the last seconds\n(q = {q:.3f}, F = {F:.3f}, $\u03B8_0$ = {theta_0:.4f}, $\u03C9_0$ = {omega_0:.3f})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('$\u03B8$ (radians)')
    axes = plt.gca()
    plt.legend(['$\u03B8$ from Runge-Kutta', 'Analytical solution'], loc = 'best')
    plt.grid(True)
    #plt.subplots_adjust(left = 0.08)
    plt.savefig(f'theta_vs_time_fin_osc_{num_of_oscillations}_F{F}_theta0_{theta_0}_q_{q}.pdf', bbox = 'tight')
    plt.show()

def plot_energy_vs_time_undamped_undriven():
    # Plot energy vs time if the constants are set to the undriven, undamped case
    if q == 0.0 and F == 0.0:
        # It is bad practice to compare floats but in this case if q and F should not be so small so as to be considered 0
        # This can be avoided by having the user specifying a boolean at the start that this is the undriven, undamped case
        energy_decimal_places_1 = 4
        plt.plot(t_array, np.around(energy_array, decimals=energy_decimal_places_1), color='r')
        energy_decimal_places_2 = 6
        plt.plot(t_array, np.around(energy_array, decimals=energy_decimal_places_2), color='k')
        plt.title(f'Energy vs time\n(q = {q:.3f}, F = {F:.3f}, $\u03B8_0$ = {theta_0:.3f}, $\u03C9_0$ = {omega_0:.3f})\nNumber of oscillation = {num_of_oscillations}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Energy (Joules)')
        plt.legend([f'Energy with {energy_decimal_places_1} decimal places',f'Energy with {energy_decimal_places_2} decimal places'], loc='center right')
        axes = plt.gca()
        # axes.set_xlim([0, 100])
        plt.subplots_adjust(left=0.16, top=0.85)
        plt.savefig(f'energy_vs_time_osc_{num_of_oscillations}.pdf', bbox='tight')
        plt.show()

generate_html_dataframe()
plot_theta_vs_time_start_undamped_undriven()
plot_theta_vs_time_finish_undamped_undriven()
plot_omega_vs_time()
plot_energy_vs_time_undamped_undriven()
plot_theta_vs_time_start()
plot_theta_vs_time_finish()

