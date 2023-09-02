from Integrators import Euler_integrator
import matplotlib.pyplot as plt
from numpy import pi
import numpy as np

if __name__ == "__main__":

    # Mass and spring properties for the oscillator
    k = 1.0
    m = 1.0

    # Initial conditions for the oscillator
    x0 = 1.0
    v0 = 0.0
    init_c = [ 0.0 , 6.0*np.pi , 100 ] # [ t_i , t_f , Nsteps ]
    # Initial time, final time, number of steps in the vecotr

    t_arr, x_arr, v_arr = Euler_integrator( m , k , x0 , v0 , init_c )

    # Create the real solution for comparison
    omega = np.sqrt( k/m )
    x_real = np.cos( omega*t_arr )

    fig, ax = plt.subplots( )
    ax.scatter( t_arr , x_arr , color = "red" , label = "Numerical" )
    ax.plot( t_arr , x_real , color = "green" , label = "Real" )

    ax.set_xlabel( "Time" )
    ax.set_ylabel( "X position" )

    ax.legend( loc = "upper right" )
    plt.grid( True )
    
    plt.show()


