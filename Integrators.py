import numpy as np

# Constants
G_const = 6.67e-11 # Grav constant in [ N*m^2/kg^2 ]

# Right-hand side for Keplerian Eq
# Inputs:
# - x_vec[ 3 ]: position vector in [R_e == Earth Radii]
# - v_vec[ 3 ]: velocity vector in [MG/R_e]
# Outputs:
# - rhs[ 6 ] -> [ rhs[ 0:3 ] , rhs[ 3:6 ] ] ( position and velocity RHS )
def rhs_Kepler( x_vec , v_vec ):

    rhs = np.zeros( 6 )
    r_3 = np.sqrt( np.dot( x_vec , x_vec ) )**3.0
    for i in range( 0 , 3 ):
        rhs[ i ] = v_vec[ i ]
        rhs[ i + 3 ] = - x_vec[ i ]/r_3

    return rhs

# Right-hand side for oscillator diff equation
# Inputs:
# - m: mass of the oscillator
# - k: spring constant
# - x: position
# - v: velocity
# Outputs:
# - rhs[ 2 ] -> [ rhs[ 0 ] , rhs[ 1 ] ] ( position and velocity RHS )
def rhs_oscillator( m , k , x , v ):
    # Compute the RHS for oscillator and return it
    rhs = [ v , - k*x/m ]

    return rhs

# Integrate 1D harmonic oscillator with Euler method
# Inputs:
# - m: mass of the oscillator
# - k: spring constant
# - x0: initial position
# - v0: initial velocity
# - init_c: [ t_i , t_f , Nsteps (int) ]
# Outputs:
# - t_arr[ N ] -> time array
# - x_sol[ N ] -> solution in position
# - v_sol[ N ] -> solution in velocity
def Euler_integrator( m , k , x0 , v0 , init_c ):

    # Unpackage the initial conditions vector into:
    t_i = init_c[ 0 ] # Initial time
    t_f = init_c[ 1 ] # Final time
    Nsteps = int( init_c[ 2 ] ) # Number of steps -> INTEGER

    # Create an array starting at t_i, ending at t_f with Nsteps steps
    t_arr = np.linspace( t_i , t_f , Nsteps )
    # Find our time step
    dt = ( t_f - t_i )/( float( Nsteps ) - 1.0 )

    # Initialize the arrays to hold the solution
    x_sol = np.zeros( Nsteps )
    v_sol = np.zeros( Nsteps )

    # Pass the initial conditions to the solution
    x_sol[ 0 ] = x0
    v_sol[ 0 ] = v0

    # INTEGRATE HERE
    for i in range( 0 , Nsteps - 1 ):
        
        rhs = rhs_oscillator( m , k , x_sol[ i ] , v_sol[ i ] )
        x_sol[ i + 1 ] = x_sol[ i ] + rhs[ 0 ]*dt
        v_sol[ i + 1 ] = v_sol[ i ] + rhs[ 1 ]*dt
        '''
        # Just to show that not using RHS but hardcoding the equations is the same!
        x_sol[ i + 1 ] = x_sol[ i ] + v_sol[ i ]*dt
        v_sol[ i + 1 ] = v_sol[ i ] - k*x_sol[ i ]*dt/m
        '''
    # Return the time array and solutions
    return t_arr, x_sol, v_sol

# Integrate 1D harmonic oscillator with Verlet method
# Inputs:
# - m: mass of the oscillator
# - k: spring constant
# - x0: initial position
# - v0: initial velocity
# - init_c: [ t_i , t_f , Nsteps (int) ]
# Outputs:
# - t_arr[ N ] -> time array
# - x_sol[ N ] -> solution in position
# - v_sol[ N ] -> solution in velocity
def Verlet_integrator( m , k , x0 , v0 , init_c ):

    # Unpackage the initial conditions vector into:
    t_i = init_c[ 0 ] # Initial time
    t_f = init_c[ 1 ] # Final time
    Nsteps = int( init_c[ 2 ] ) # Number of steps -> INTEGER

    # Create an array starting at t_i, ending at t_f with Nsteps steps
    t_arr = np.linspace( t_i , t_f , Nsteps )
    # Find our time step
    dt = ( t_f - t_i )/( float( Nsteps ) - 1.0 )

    # Initialize the arrays to hold the solution
    x_sol = np.zeros( Nsteps )
    v_sol = np.zeros( Nsteps )

    # Pass the initial conditions to the solution
    x_sol[ 0 ] = x0
    v_sol[ 0 ] = v0

    # INTEGRATE HERE - VERLET METHOD
    for i in range( 0 , Nsteps - 1 ):
        
        # Get right-hand side in the initial moment "i"
        rhs = rhs_oscillator( m , k , x_sol[ i ] , v_sol[ i ] )
        # Compute the half-moment velocity (between "i" and "i+1")
        v_half = v_sol[ i ] + rhs[ 1 ]*dt/2.0
        # Use the half-moment velocity to compute RHS for position
        rhs = rhs_oscillator( m , k , x_sol[ i ] , v_half )
        # Compute the next step in position based on the half-velocity step
        x_sol[ i + 1 ] = x_sol[ i ] + rhs[ 0 ]*dt
        # Now we have position in "i+1" -> use it to get RHS for velocity to get there
        rhs = rhs_oscillator( m , k , x_sol[ i + 1 ] , v_half )
        # Compute next step velocity from positon in "i+1" and estimate in v_half
        v_sol[ i + 1 ] = v_half + rhs[ 1 ]*dt/2.0

    # Return the time array and solutions
    return t_arr, x_sol, v_sol

# Integrate 1D harmonic oscillator with RK4 method
# Inputs:
# - m: mass of the oscillator
# - k: spring constant
# - x0: initial position
# - v0: initial velocity
# - init_c: [ t_i , t_f , Nsteps (int) ]
# Outputs:
# - t_arr[ N ] -> time array
# - x_sol[ N ] -> solution in position
# - v_sol[ N ] -> solution in velocity
def RK4_integrator( m , k , x0 , v0 , init_c ):

    # Unpackage the initial conditions vector into:
    t_i = init_c[ 0 ] # Initial time
    t_f = init_c[ 1 ] # Final time
    Nsteps = int( init_c[ 2 ] ) # Number of steps -> INTEGER

    # Create an array starting at t_i, ending at t_f with Nsteps steps
    t_arr = np.linspace( t_i , t_f , Nsteps )
    # Find our time step
    dt = ( t_f - t_i )/( float( Nsteps ) - 1.0 )

    # Initialize the arrays to hold the solution
    x_sol = np.zeros( Nsteps )
    v_sol = np.zeros( Nsteps )

    # Pass the initial conditions to the solution
    x_sol[ 0 ] = x0
    v_sol[ 0 ] = v0

    # INTEGRATE HERE - VERLET METHOD
    for i in range( 0 , Nsteps - 1 ):
        
        # Evaluate RHS at moment "i" and find the Runge-Kutta k1
        rhs = rhs_oscillator( m , k , x_sol[ i ] , v_sol[ i ] )
        kx_1 = rhs[ 0 ]
        kv_1 = rhs[ 1 ]
        # Find the intermediate X and V at the second point
        x_temp = x_sol[ i ] + kx_1*dt/2.0
        v_temp = v_sol[ i ] + kv_1*dt/2.0

        # Evaluate RHS at moment "i + k1/2" and find Runge-Kutta k2
        rhs = rhs_oscillator( m , k , x_temp , v_temp )
        kx_2 = rhs[ 0 ]
        kv_2 = rhs[ 1 ]
        # Find the intermediate X and V at the third point
        x_temp = x_sol[ i ] + kx_2*dt/2.0
        v_temp = v_sol[ i ] + kv_2*dt/2.0

        # Evaluate RHS at moment "i + k2/2" and find Runge-Kutta k3
        rhs = rhs_oscillator( m , k , x_temp , v_temp )
        kx_3 = rhs[ 0 ]
        kv_3 = rhs[ 1 ] 
        # Find the intermediate X and V at the forth point
        x_temp = x_sol[ i ] + kx_3*dt
        v_temp = v_sol[ i ] + kv_3*dt

        # Evaluate RHS at moment "i + k2/2" and find Runge-Kutta k3
        rhs = rhs_oscillator( m , k , x_temp , v_temp )
        kx_4 = rhs[ 0 ]
        kv_4 = rhs[ 1 ] 

        # At this point I have all k coefficient of RK method, I have to get i+1 solutions
        x_sol[ i + 1 ] = x_sol[ i ] + ( kx_1 + 2.0*kx_2 + 2.0*kx_3 + kx_4 )*dt/6.0
        v_sol[ i + 1 ] = v_sol[ i ] + ( kv_1 + 2.0*kv_2 + 2.0*kv_3 + kv_4 )*dt/6.0

    # Return the time array and solutions
    return t_arr, x_sol, v_sol

# RHS for ballistic trajectory
# Inputs:
# - beta: drag coefficient in [1/sec]
# - g_acc: Grav acceleration [m/sec^2]
# - r_vec[ 2 ]: position vector
# - v_vec[ 2 ]: velocity vector
# Outputs:
# - rhs[ 4 ]: vector which holds RHS for [ x , y , vx , vy ]
def rhs_ballistic( beta , g_acc , r_vec , v_vec ):

    rhs = [ v_vec[ 0 ] , # right term to x_dot 
            v_vec[ 1 ] , # right term to y_dot
            - beta*v_vec[ 0 ] ,            # right term to vx_dot
            - g_acc - beta*v_vec[ 1 ] ]    # right term to vy_dot
    
    return rhs

# Integrate 2D ballistic trajectory with Verlet
# Inputs:
# - beta: drag coefficient in [1/sec]
# - g_acc: Grav acceleration [m/sec^2]
# - r0_vec[ 2 ]: initial position vector
# - v0_vec[ 2 ]: initial velocity vector
# - init_c: [ t_i , t_f , Nsteps (int) ]
# Outputs:
# - t_arr[ N ] -> time array
# - r_sol[ N ][ 2 ] -> solution in position
# - v_sol[ N ][ 2 ] -> solution in velocity
def Verlet_Ballistic( beta , g_acc , r0_vec , v0_vec , init_c ):

    # Unpackage the initial conditions vector into:
    t_i = init_c[ 0 ] # Initial time
    t_f = init_c[ 1 ] # Final time
    Nsteps = int( init_c[ 2 ] ) # Number of steps -> INTEGER

    # Create an array starting at t_i, ending at t_f with Nsteps steps
    t_arr = np.linspace( t_i , t_f , Nsteps )
    # Find our time step
    dt = ( t_f - t_i )/( float( Nsteps ) - 1.0 )

    # Initialize the arrays to hold the solution
    r_sol = np.zeros( ( Nsteps , 2 ) )
    v_sol = np.zeros( ( Nsteps , 2 ) )

    # Pass the initial conditions to the solution
    r_sol[ 0 ] = r0_vec
    v_sol[ 0 ] = v0_vec

    # INTEGRATE HERE - VERLET METHOD
    for i in range( 0 , Nsteps - 1 ):

        # Get RHS at moment "i" and compute half-velocity for BOTH X and Y
        rhs = rhs_ballistic( beta , g_acc , r_sol[ i ] , v_sol[ i ] )
        vhalf_x = v_sol[ i ][ 0 ] + rhs[ 2 ]*dt/2.0
        vhalf_y = v_sol[ i ][ 1 ] + rhs[ 3 ]*dt/2.0

        # Get RHS at "middle point" and compute r_sol at "i+1"
        rhs = rhs_ballistic( beta , g_acc , r_sol[ i ] , [ vhalf_x , vhalf_y ] )
        r_sol[ i + 1 ][ 0 ] = r_sol[ i ][ 0 ] + rhs[ 0 ]*dt
        r_sol[ i + 1 ][ 1 ] = r_sol[ i ][ 1 ] + rhs[ 1 ]*dt

        # Get RHS at "i+1"st point and compute velocity at "i+1"
        rhs = rhs_ballistic( beta , g_acc , r_sol[ i + 1 ] , [ vhalf_x , vhalf_y ] )
        v_sol[ i + 1 ][ 0 ] = vhalf_x + rhs[ 2 ]*dt/2.0
        v_sol[ i + 1 ][ 1 ] = vhalf_y + rhs[ 3 ]*dt/2.0

        if r_sol[ i + 1 ][ 1 ] <= 0.0:
            r_sol[ i + 1 ][ 1 ] = 0.0
            v_sol[ i + 1 ][ 1 ] *= - 0.5

        # When more than 2, better to loop it :)
        #for j in range ( 0 , 2 ):
        #    r_sol[ i + 1 ][ j ] = r_sol[ i ][ j ] + rhs[ j ]*dt

    # Return the time array and solutions
    return t_arr, r_sol, v_sol

# Integrate Keplerian orbital propagation by RK4 method
# Inputs:
# - M: mass of the central body [kg]
# - R0: radius of the central body [km]
# - x0_vec[ 3 ]: initial position in [km] 
# - v0_vec[ 3 ]: initial velocity in [km/sec]
# - init_c: [ t_i , t_f , Nsteps (int) ]
# Outputs:
# - t_arr[ N ] -> time array
# - x_sol[ N ][ 3 ] -> solution in position
# - v_sol[ N ][ 3 ] -> solution in velocity
def Kepler_RK4( M , R0 , x0_vec , v0_vec , init_c ):

    # Unpackage the initial conditions vector into:
    t_i = init_c[ 0 ] # Initial time
    t_f = init_c[ 1 ] # Final time
    Nsteps = int( init_c[ 2 ] ) # Number of steps -> INTEGER

    # Create an array starting at t_i, ending at t_f with Nsteps steps
    t_arr = np.linspace( t_i , t_f , Nsteps )
    # Find our time step
    dt = ( t_f - t_i )/( float( Nsteps ) - 1.0 )

    # Initialize the arrays to hold the solution
    x_sol = np.zeros( ( Nsteps , 3 ) )
    v_sol = np.zeros( ( Nsteps , 3 ) )

    # Norming constants -> we will use dimensionless units, these will carry all dimensionality
    R_dim = R0 # Char radius in [km]
    V_dim = np.sqrt( M*G_const/( 1000*R0 ) )/1000.0 # Char speed in [km/sec]
    T_dim = R_dim/V_dim # Char time in [sec]

    # Pass the initial conditions to the solution and norm it
    t_arr /= T_dim
    dt /= T_dim
    x_sol[ 0 ] = x0_vec/R_dim
    v_sol[ 0 ] = v0_vec/V_dim

    # Initialize Runge-Kutta constants
    # k_RK[ i ] will hold the entire vector 
    k_RK = np.zeros( ( 4 , 6 ) )
    # Temporary vectors for holding intermediate RK steps
    x_temp = np.zeros( 3 )
    v_temp = np.zeros( 3 )

    # INTEGRATE HERE - RK(4) LOOP
    for i in range( 0 , Nsteps - 1 ):

        # Testng out just an Euler integrator
        #rhs_vec = rhs_Kepler( x_sol[ i ] , v_sol[ i ] )
        #x_sol[ i + 1 ] = x_sol[ i ] + rhs_vec[ 0:3 ]*dt
        #v_sol[ i + 1 ] = v_sol[ i ] + rhs_vec[ 3:6 ]*dt
        
        k_RK[ 0 ] = rhs_Kepler( x_sol[ i ] , v_sol[ i ] )

        for j in range( 0 , 3 ):
            x_temp[ j ] = x_sol[ i ][ j ] + k_RK[ 0 ][ j ]*dt/2.0
            v_temp[ j ] = v_sol[ i ][ j ] + k_RK[ 0 ][ j + 3 ]*dt/2.0
        k_RK[ 1 ] = rhs_Kepler( x_temp , v_temp )

        for j in range( 0 , 3 ):
            x_temp[ j ] = x_sol[ i ][ j ] + k_RK[ 1 ][ j ]*dt/2.0
            v_temp[ j ] = v_sol[ i ][ j ] + k_RK[ 1 ][ j + 3 ]*dt/2.0
        k_RK[ 2 ] = rhs_Kepler( x_temp , v_temp )

        for j in range( 0 , 3 ):
            x_temp[ j ] = x_sol[ i ][ j ] + k_RK[ 2 ][ j ]*dt
            v_temp[ j ] = v_sol[ i ][ j ] + k_RK[ 2 ][ j + 3 ]*dt
        k_RK[ 3 ] = rhs_Kepler( x_temp , v_temp )

        for j in range( 0 , 3 ):
            x_sol[ i + 1 ][ j ] = x_sol[ i ][ j ] + ( k_RK[ 0 ][ j ] + 2.0*k_RK[ 1 ][ j ] + 2.0*k_RK[ 2 ][ j ] + k_RK[ 3 ][ j ] )*dt/6.0
            v_sol[ i + 1 ][ j ] = v_sol[ i ][ j ] + ( k_RK[ 0 ][ j + 3 ] + 2.0*k_RK[ 1 ][ j + 3 ] + 2.0*k_RK[ 2 ][ j + 3 ] + k_RK[ 3 ][ j + 3 ] )*dt/6.0
        
        #print( k_RK )
    # Return the time array and solutions
    return t_arr*T_dim, x_sol*R_dim, v_sol*V_dim
