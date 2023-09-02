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