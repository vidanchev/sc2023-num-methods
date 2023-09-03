from Integrators import Euler_integrator, Verlet_integrator, Verlet_Ballistic, RK4_integrator
import matplotlib.pyplot as plt
from numpy import pi
import numpy as np

if __name__ == "__main__":

    ##############################################################
    # THIS PART IS FOR THE OSCILLATOR
    ##############################################################
    # Mass and spring properties for the oscillator
    k = 1.0
    m = 1.0

    # Initial conditions for the oscillator
    x0 = 1.0
    v0 = 0.0
    init_c = [ 0.0 , 6.0*np.pi , 100 ] # [ t_i , t_f , Nsteps ]
    # Initial time, final time, number of steps in the vecotr

    # Call the Euler, Verlet and Runge-Kutta 4th order integrators and obtain solution from each
    t_arr, x_E, v_E = Euler_integrator( m , k , x0 , v0 , init_c )
    t_arr, x_V, v_V = Verlet_integrator( m , k , x0 , v0 , init_c )
    t_arr, x_R, v_R = RK4_integrator( m , k , x0 , v0 , init_c )

    # Create the real solution for comparison - considering x( t ) = cos( \omega*t ) initial conditions
    omega = np.sqrt( k/m )
    x_real = np.cos( omega*t_arr )

    # Plot the 3 numerical solutions side by side with the real one for comparison
    fig, ax = plt.subplots( )
    ax.scatter( t_arr , x_E , color = "red" , label = "Euler" , linewidth = 2 )
    ax.scatter( t_arr , x_V , color = "blue" , label = "Verlet" , linewidth = 2 )
    ax.scatter( t_arr , x_R , color = "gold" , label = "RK(4)" , linewidth = 2 )
    ax.plot( t_arr , x_real , color = "green" , label = "Real" , linewidth = 2 )

    ax.set_xlabel( "Time" )
    ax.set_ylabel( "X position" )

    ax.legend( loc = "upper right" )
    plt.grid( True )
    
    plt.show()

    # Plot the 3 numerical solutions error side by side
    fig, ax = plt.subplots( )

    # Create 3 arrays to be populated with the relative errors
    Npoints = init_c[ 2 ]
    err_E = np.zeros( Npoints )
    err_V = np.zeros( Npoints )
    err_R = np.zeros( Npoints )

    # Compute relative error as ( real - numerical )/real -> take absolute value and multiply by 100 for [%]
    for i in range( 0 , Npoints ):
        err_E[ i ] = abs( ( x_E[ i ] - x_real[ i ] )/x_real[ i ] )*100.0
        err_V[ i ] = abs( ( x_V[ i ] - x_real[ i ] )/x_real[ i ] )*100.0
        err_R[ i ] = abs( ( x_R[ i ] - x_real[ i ] )/x_real[ i ] )*100.0

    ax.scatter( t_arr , err_E , color = "red" , label = "Euler" , linewidth = 2 )
    ax.scatter( t_arr , err_V , color = "blue" , label = "Verlet" , linewidth = 2 )
    ax.scatter( t_arr , err_R , color = "gold" , label = "RK(4)" , linewidth = 2 )

    ax.set_xlabel( "Time" )
    ax.set_ylabel( "X position error in [%]" )

    ax.legend( loc = "upper right" )
    plt.grid( True )
    plt.yscale( "log" )  
    
    plt.show()
    
    ##############################################################
    # THIS PART IS FOR THE BALLISTIC TRAJECTORY
    ##############################################################
    '''
    # Parameters for the simulation
    beta = 0.1 # Linear drag coefficient [1/sec]
    g_acc = 9.8 # Grav acceleration [m/sec^2]

    # Initial conditions for the trajectory - PLAY WITH THEM
    alp_horizon = 60.0 # Angle relative to the horizon in [deg]
    speed_0 = 50.0 # Initial speed (magnitude of velocity) [m/sec]

    # Initialize the initial vectors - DON'T TOUCH
    r0_vec = [ 0.0 , 0.0 ] # Radius vector [ x , y ] components
    v0_vec = [ speed_0*np.cos( alp_horizon*np.pi/180.0 ) ,
               speed_0*np.sin( alp_horizon*np.pi/180.0 ) ]

    init_c = [ 0.0 , 8.0 , 100 ] # [ t_i , t_f , Nsteps ]
    # Initial time, final time, number of steps in the vecotr

    t_arr, r_V, v_V = Verlet_Ballistic( beta , g_acc , r0_vec , v0_vec , init_c )
    # REMEMBER r_V and v_V ARE [ N ][ 2 ] -> EACH ELEMENT IS A 2D VECTOR

    # Create the real solution for comparison
    Npoints = init_c[ 2 ]
    r_Real = np.zeros( ( Npoints , 2 ) )
    for i in range( 0 , Npoints ):
        r_Real[ i ] = [ v0_vec[ 0 ]*t_arr[ i ] ,
                        v0_vec[ 1 ]*t_arr[ i ] - g_acc*( t_arr[ i ]**2.0 )/2.0 ]

    fig, ax = plt.subplots( )
    ax.scatter( np.transpose( r_V )[ 0 ] , np.transpose( r_V )[ 1 ] , color = "red" , label = "Verlet" , linewidth = 3 )
    ax.plot( np.transpose( r_Real )[ 0 ] , np.transpose( r_Real )[ 1 ] , color = "green" , label = "Real" , linewidth = 3 )

    ax.set_xlabel( "X position [m]" )
    ax.set_ylabel( "Y position [m]" )

    ax.legend( loc = "upper right" )
    plt.grid( True )
    
    plt.show()
    '''
