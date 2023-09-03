import numpy as np

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