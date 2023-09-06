import numpy as np 
import csv
import matplotlib.pyplot as plt
from Integrators import Kepler_RK4
from scipy.interpolate import CubicSpline

# Parse a .csv file with satellite time, position and velocity (PVT) for plotting
# Inputs:
# - filename of the .csv file where the results are saved
# NOTE: File structure is expected to be "Time (UTCG),x (km),y (km),z (km),vx (km/sec),vy (km/sec),vz (km/sec)"
# Output:
# It will return arrays of time, pos, vel as:
# - time[ Npoints ]: time in [sec] from simulation start
# - pos[ 3 ][ Npoints ]: 3D position in [km]
# - vel[ 3 ][ Npoints ]: 3D velocity in [km/s]
def parse_orbit_data( filename ):

    fp = open( filename , "r" )

    if fp.readable( ):
        data = csv.reader( fp )
        lst = [ ]
        for line in data:
            lst.append( line )
        ndata = len( lst ) - 1

        time = np.zeros( ndata )
        pos = np.zeros( ( 3 , ndata ) )
        vel = np.zeros( ( 3 , ndata ) )

        for i in range( 0 , ndata ):
            time[ i ] = float( lst[ i + 1 ][ 0 ] )
            for j in range( 0 , 3 ):
                pos[ j ][ i ] = float( lst[ i + 1 ][ j + 1 ] )
                vel[ j ][ i ] = float( lst[ i + 1 ][ j + 4 ] )
    else:
        print( "Unreadable data, something's wrong with the file " + filename )
    
    return time, pos, vel

if __name__ == "__main__":

    # File simulated from GMAT/STK
    file_name = "Satellite_PVT_GMAT.csv"

    time , pos , vel = parse_orbit_data( file_name )

    # Numerical parameters
    M = 5.972e24 # Earth Mass in [kg]
    R0 = 6378.137 # Earth Radius in [km]
    # Pass intial position and velocity from the .csv file parsed
    x0_vec = np.transpose( pos )[ 0 ]
    v0_vec = np.transpose( vel )[ 0 ]
    
    # Initialize initial and final time from the file, make time step approx 10 sec by choosing int of total time as number of points
    init_c = [ time[ 0 ] , time[ -1 ] , int( time[ - 1 ]/10.0 ) ]  # [ t_i , t_f , Nsteps ]

    # Call the RK4 integrator with the Keplerian potential
    t_num, pos_num, vel_num = Kepler_RK4( M , R0 , x0_vec , v0_vec , init_c )
    # Transpose pos_num & vel_num to keep the format as from the .csv
    pos_num = np.transpose( pos_num )
    vel_num = np.transpose( vel_num )

    # Interpolate from the GMAT/STK solution (it's adaptive step) so you find its prediction and velocity at each of the time steps used
    # Using cubic spline as a good general accuracy method
    spl_X = CubicSpline( time , pos[ 0 ] )
    spl_Y = CubicSpline( time , pos[ 1 ] )
    spl_Z = CubicSpline( time , pos[ 2 ]  )
    spl_VX = CubicSpline( time , vel[ 0 ] )
    spl_VY = CubicSpline( time , vel[ 1 ] )
    spl_VZ = CubicSpline( time , vel[ 2 ] )

    # Define error vector in position/velocity as the difference between the norm of the STK/GMAT and our predicted position/velocity
    err_pos = np.zeros( len( t_num ) )
    err_vel = np.zeros( len( t_num ) )

    for i in range( 0 , len( t_num ) ):
        pos_real = [ spl_X( t_num[ i ] ) , spl_Y( t_num[ i ] ) , spl_Z( t_num[ i ] ) ]
        vel_real = [ spl_VX( t_num[ i ] ) , spl_VY( t_num[ i ] ) , spl_VZ( t_num[ i ] ) ]
        dpos = np.transpose( pos_num )[ i ] - np.array( pos_real ) # Error vector for position
        err_pos[ i ] = np.sqrt( np.dot( dpos , dpos ) ) # Take norm (absolute diff in distance)
        dvel = np.transpose( vel_num )[ i ] - np.array( vel_real ) # Error vector for velocity
        err_vel[ i ] = np.sqrt( np.dot( dvel , dvel ) ) # Take norm (absolute diff in distance)

    fig, ax = plt.subplots()
    ax.plot( t_num , err_pos , color = "green" , linestyle = "solid" )#, label = r"Position error" )

    ax.set_xlabel( r"Time [sec]" )
    ax.set_ylabel( r"Position absolute error in [km]" )

    #ax.legend( loc = "upper right" )
    plt.show()
    #ax.set_ylim( 0.0 , pi/2.0 )
    plt.grid( True )

    fig, ax = plt.subplots()
    ax.plot( t_num , err_vel , color = "green" , linestyle = "solid" )#, label = r"Velocity error" )

    ax.set_xlabel( r"Time [sec]" )
    ax.set_ylabel( r"Velocity absolute error in [km/s]" )

    #ax.legend( loc = "upper right" )
    plt.show()
    #ax.set_ylim( 0.0 , pi/2.0 )
    plt.grid( True )

    # 3D Plot example
    ax = plt.axes( projection = "3d" )
    ax.plot3D( pos[ 0 ] , pos[ 1 ] , pos[ 2 ] , color = "blue" )
    ax.scatter( pos_num[ 0 ] , pos_num[ 1 ] , pos_num[ 2 ] , color = "red" )
    plt.show( )

    '''
    # 2D Plot example
    fig, ax = plt.subplots()
    ax.plot( time , pos[ 0 ] , color = "green" , linestyle = "solid" , label = r"pos X" )
    ax.plot( time , pos[ 1 ] , color = "red" , linestyle = "solid" , label = r"pos Y" )
    ax.plot( time , pos[ 2 ] , color = "blue" , linestyle = "solid" , label = r"pos Z" )

    ax.set_xlabel( r"Time [sec]" )
    ax.set_ylabel( r"Position in [km]" )

    ax.legend( loc = "upper right" )

    #ax.set_ylim( 0.0 , pi/2.0 )
    plt.grid( True )

    #fig.savefig( "fig_name.pdf" , format = "pdf" )
    
    plt.show()
    '''