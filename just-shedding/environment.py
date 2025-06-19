import numpy as np
from numpy.linalg import pinv
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
# type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
from matplotlib import colors
import matplotlib.animation as animation

# TODO: parallelize compute_bvs_gamma
# TODO: parallelize fvs_flow_velocity and bvs_flow_velocity
# TODO: but code in form to parallelize on HPC (numba)
# TODO: check vortex sheet in tangent to tail segment

from fish import Fish

class Environment:

    def __init__(self, fishies):

        # Integration parameters
        self.time_step = 0
        self.time_N = 4001
        self.t_max = 20
        self.delta_T = self.t_max/(self.time_N - 1)

        # Define fishies in the water
        self.fishies = fishies
        self.fish_N = len(self.fishies)

        # Store free vortex street (FVS)
        self.fvs_N = 0
        self.fvs_positions = np.empty((2,0)) # 2 by fvs_N
        self.fvs_velocities = np.empty((2,0)) # 2 by fvs_N
        self.fvs_Gamma = np.array([]) # fvs_N,

        # FVS dissapation parameters
        self.fvs_shedtime = np.array([]) # fvs_N,
        fvs_dissapation_time = 4
        self.fvs_max_time = np.round(fvs_dissapation_time/self.delta_T)

        # Newly shed vortex information
        self.new_fvs_positions = np.zeros((2,self.fish_N))
        self.new_fvs_Gamma_dot = np.zeros((self.fish_N))

        # Find the size of the initial state
        self.fish_state_size = 0
        for ii, fish in enumerate(self.fishies):
            self.fish_state_size += 4*fish.N

        # Define the initial state for integration
        self.init_state = np.zeros((self.fish_state_size))
        start = 0
        for fish in self.fishies:
            end = start + 4*fish.N
            self.init_state[start:end] = np.concatenate([fish.positions.reshape((2*fish.N), order='F'),
                                                         fish.velocities.reshape((2*fish.N), order='F')])
            start = end
        
        # Store outputs of integration
        self.output = []
        self.output_Gamma = []
        self.output_bvs = []
        self.output_bvs_gamma = []

    def fvs_flow_velocity(self, z, delta=0.2):

        """
        Compute the 2D velocity field induced by a collection of fish-like objects
        using the Biot-Savart law for free vortex sheets (FVS).

        Parameters
        ----------
        z : ndarray of shape (2, n_targets)
            Target points where the velocity is evaluated.
            - z[0, :] contains x-coordinates
            - z[1, :] contains y-coordinates

        delta : float, optional (default=0.2)
            Regularization parameter added to the squared distance to avoid singularities.
            Useful for numerical stability, especially when points are close.

        Returns
        -------
        velocity : ndarray of shape (2, n_targets)
            The induced velocity field at each of the M target points.
            - velocity[0, :] contains the x-component (u_x)
            - velocity[1, :] contains the y-component (u_y)
        """

        # Extract x and y positions from targets
        x = z[0, :]
        y = z[1, :]
        n_targets = x.size
        
        # Extract x and y positions from free vortices
        v_x = self.fvs_positions[0, :]
        v_y = self.fvs_positions[1, :]

        # Compute pairwise distances
        dx = x - v_x[:, np.newaxis]
        dy = y - v_y[:, np.newaxis]
        norm2 = dx**2 + dy**2 + delta**2

        # Strength of each vortex
        strength = self.fvs_Gamma / (2 * np.pi)
        strength_array = strength[:, np.newaxis]

        # Accumulate velocity contributions
        u_x = np.sum(-strength_array * dy / norm2, axis=0)
        u_y = np.sum( strength_array * dx / norm2, axis=0)

        return np.vstack((u_x, u_y))
    
    def bvs_flow_velocity(self, z, delta=0.01):

        """
        Compute the 2D velocity field induced by a collection of fish-like objects
        using the Biot-Savart law for bound vortex sheets (BVS).

        Parameters
        ----------
        z : ndarray of shape (2, n_targets)
            Target points where the velocity is evaluated.
            - z[0, :] contains x-coordinates
            - z[1, :] contains y-coordinates

        delta : float, optional (default=0.1)
            Regularization parameter added to the squared distance to avoid singularities.
            Useful for numerical stability, especially when points are close.

        Returns
        -------
        velocity : ndarray of shape (2, n_targets)
            The induced velocity field at each of the M target points.
            - velocity[0, :] contains the x-component (u_x)
            - velocity[1, :] contains the y-component (u_y)
        """

        # Extract x and y positions from targets
        x = z[0, :]
        y = z[1, :]
        n_targets = x.size

        # Initialize velocity vectors
        u_x = np.zeros(n_targets)
        u_y = np.zeros(n_targets)

        # Loop over all fish
        for fish in self.fishies:

            # Extract x and y positions from bound vortices
            v_x = fish.bvs_positions[0, :]
            v_y = fish.bvs_positions[1, :]

            # Compute pairwise distances
            dx = x - v_x[:, np.newaxis]
            dy = y - v_y[:, np.newaxis]
            norm2 = dx**2 + dy**2 + delta**2

            # Strength of each vortex
            strength = (fish.bvs_gamma * fish.bvs_length) / (2 * np.pi)
            strength_array = strength[:, np.newaxis]

            # Accumulate velocity contributions
            u_x += np.sum(-strength_array * dy / norm2, axis=0)
            u_y += np.sum( strength_array * dx / norm2, axis=0)

        return np.vstack((u_x, u_y))

    def compute_fvs_convection(self):
        
        self.fvs_velocities = self.fvs_flow_velocity(self.fvs_positions) + self.bvs_flow_velocity(self.fvs_positions)

    def compute_bvs_gamma(self):
        
        """
        Compute the bound vortex sheet (BVS) circulation strengths for all fish.

        This function assembles and solves a linear system representing the
        boundary conditions for inviscid, incompressible flow around multiple
        fish-like bodies using bound vortex sheets. It enforces two sets of 
        conditions:

        1. Non-penetration: Normal velocity across each body surface panel must be zero.
        2. Kelvin circulation: The net circulation must balance the free vortex sheet circulation.

        It aggregates all fish BVS nodes and panels into a global system,
        solves for the circulation strengths, and stores the resulting gamma
        values back into each fish object.

        """

        # Compute total number of bound vortices
        bvs_N = np.array([fish.bvs_N for fish in self.fishies])        
        total_BVS = np.sum(bvs_N)
        nonpenetration_conditions = total_BVS - self.fish_N

        # Preallocate all node values
        x = np.zeros((2, total_BVS))
        v = np.zeros((2, total_BVS))
        bvs_lengths = np.zeros(total_BVS)

        # Preallocate all edge values
        n = np.zeros((2, nonpenetration_conditions))
        midpoints = np.zeros((2, nonpenetration_conditions))
        midpoint_velocities = np.zeros((2, nonpenetration_conditions))
    
        # Start index trackers
        node_start = 0
        edge_start = 0

        for ii, fish in enumerate(self.fishies):

            # End index trackers
            node_end = node_start + bvs_N[ii]
            edge_end = edge_start + bvs_N[ii] - 1

            # Store all node data
            x[:,node_start:node_end] = fish.bvs_positions
            v[:,node_start:node_end] = fish.bvs_velocities
            bvs_lengths[node_start:node_end] = fish.bvs_length

            # Store all edge data
            e = x[:,node_start+1:node_end] - x[:,node_start:node_end-1]
            t = e / np.linalg.norm(e)
            n[:,edge_start:edge_end] = np.array([-t[1,:], t[0,:]])

            midpoints[:,edge_start:edge_end] = (x[:,node_start:node_end-1] + x[:,node_start+1:node_end])/2
            midpoint_velocities[:,edge_start:edge_end] = (v[:,node_start:node_end-1] + v[:,node_start+1:node_end])/2

            # Update index trackers
            node_start = node_end
            edge_start = edge_end
            
        # Set up linear system
        A = np.zeros((total_BVS,total_BVS))
        B = np.zeros((total_BVS))

        # RHS of non-penetration conditions
        flow_at_midpoints = self.fvs_flow_velocity(midpoints)
        B[:nonpenetration_conditions] = np.sum((midpoint_velocities - flow_at_midpoints) * n, axis=0)

        # LHS of non-penetration conditions
        delta_x = midpoints[0, :][:, np.newaxis] - x[0, :]
        delta_y = midpoints[1, :][:, np.newaxis] - x[1, :]
        norm2 = delta_x ** 2 + delta_y ** 2
        norm2[norm2 < 1e-12] = 1e-12
        direction = -n[0, :][:, np.newaxis] * delta_y + n[1, :][:, np.newaxis] * delta_x
        A[:nonpenetration_conditions,:] = bvs_lengths * direction / (2 * np.pi * norm2)

        # Kelvin circulation theorem condition
        start = 0
        kelvin_value = -np.sum(self.fvs_Gamma)
        for kk in range(self.fish_N):

            # Keep index tracker
            end = start + bvs_N[kk]

            # Add to set of linear equations
            A[kk + nonpenetration_conditions, start:end] = bvs_lengths[start:end]
            B[kk + nonpenetration_conditions] = kelvin_value

            # Update index tracker
            start = end

        # Solve system and update vortex densities
        if np.linalg.matrix_rank(A) < total_BVS:
            gammas = np.zeros((total_BVS))
        else:
            gammas =  np.linalg.solve(A,B) #np.linalg.pinv(A) @ B
            start = 0
            for fish in self.fishies:
                end = start + fish.bvs_N
                fish.bvs_gamma = gammas[start:end]
                start = end

    def shed_vortices(self):

        """
        Update the free vortex street (FVS) by shedding new vortices from each fish
        and removing old, dissipated vortices.
        
        First:
        - Appends the newly shed vortices to the FVS arrays (positions, velocities, strengths, shed times).
        - Filters out vortices that have existed longer than the maximum allowed dissipation time.
        - Updates the current number of active vortices (self.fvs_N).

        For each fish:
        - Computes the tangential flow at the tail to determine the strength of the shed vortex.
        - Stores the new vortex positions and strength derivatives.
        """

        # Check if a vortex has been shed
        if np.any(self.time_step != 0):

            # Shed past new vortex and update FVS lists
            self.fvs_velocities = np.hstack([self.fvs_velocities, np.zeros((2,self.fish_N))])
            self.fvs_positions = np.hstack([self.fvs_positions, self.new_fvs_positions])
            self.fvs_Gamma = np.hstack([self.fvs_Gamma, self.delta_T * self.new_fvs_Gamma_dot.flatten()])
            self.fvs_shedtime = np.hstack([self.fvs_shedtime, np.full(self.fish_N, self.time_step)])

            # Find vortices that have dissapated
            fvs_existance_time = self.time_step - self.fvs_shedtime 
            not_dissipated_mask = (fvs_existance_time < self.fvs_max_time).flatten()

            # Eliminate old vortices from the list
            self.fvs_positions = self.fvs_positions[:,not_dissipated_mask]
            self.fvs_velocities = self.fvs_velocities[:,not_dissipated_mask]
            self.fvs_Gamma = self.fvs_Gamma[not_dissipated_mask]
            self.fvs_shedtime = self.fvs_shedtime[not_dissipated_mask]
            self.fvs_N = self.fvs_positions.shape[1]

        # Each fish sheds a vortex
        for ii, fish in enumerate(self.fishies):

            # Compute velocity jump at tail
            tail_velocity = fish.bvs_velocities[:,-1]
            tail_tangent = ((fish.bvs_positions[:,-1] - fish.bvs_positions[:,-2]) / 
                            np.linalg.norm(fish.bvs_positions[:,-1] - fish.bvs_positions[:,-2]))
            bvs_flow = self.bvs_flow_velocity(fish.bvs_positions[:,-1][:, np.newaxis])
            fvs_flow = self.fvs_flow_velocity(fish.bvs_positions[:,-1][:, np.newaxis])
            tail_flow_velocity = (bvs_flow + fvs_flow).flatten()
            tail_tangent_flow = np.dot(tail_flow_velocity - tail_velocity, tail_tangent)

            # Solve for pressure difference
            u_minus = tail_tangent_flow - fish.bvs_gamma[-1]/2
            u_plus  = tail_tangent_flow + fish.bvs_gamma[-1]/2

            # Store new vortex position and rate of change of strength
            self.new_fvs_positions[:,ii] = fish.bvs_positions[:,-1]
            self.new_fvs_Gamma_dot[ii] = -(u_minus ** 2 - u_plus ** 2)/2

            # BRUTE FORCE CORRECTION
            if self.time_step < np.round(0.5/self.delta_T):
                self.new_fvs_Gamma_dot[ii] = -(self.time_step/np.round(0.5/self.delta_T))**2 * (u_minus ** 2 - u_plus ** 2)/2

    def state_derivatives(self, t, z0):

        # Preallocate space for derivatives
        dzdt = np.zeros_like(z0)

        # Find fish configuration derivatives 
        start = 0
        for fish in self.fishies:
            end = start + 4*fish.N
            dzdt[start:end] = fish.internal_forces(t, z0[start:end])
            start = end

        # Find FVS positions derivatives
        self.fvs_positions = z0[end:].reshape((2,self.fvs_N), order='F')
        # self.compute_bvs_gamma() # TEMP!
        self.compute_fvs_convection()
        if self.fvs_N > 0:
            dzdt[end:] = self.fvs_velocities.reshape((2*self.fvs_N), order='F')
        
        # Return state derivative
        return dzdt

    def integration_step(self):

        # Integrate forward one time step
        t_span = [self.delta_T*self.time_step, self.delta_T*(self.time_step + 1)]
        step = solve_ivp(self.state_derivatives, t_span, self.init_state, method='RK45', rtol=1e-3, atol=1e-6)
        
        # Shed new vortices at end of integration step (changes self.fvs_positions)
        self.compute_bvs_gamma() # TEMP!
        self.shed_vortices()

        # Update initial conditions
        fish_config = step.y[0:self.fish_state_size, -1]
        fvs_config = self.fvs_positions.reshape((2*self.fvs_N), order='F')
        self.init_state = np.concatenate((fish_config,fvs_config))
        self.time_step += 1

        # Store outputs in array
        self.output.append(self.init_state.copy())
        self.output_Gamma.append(self.fvs_Gamma.copy())
        for fish in self.fishies:
            self.output_bvs.append(fish.bvs_positions.copy())
            self.output_bvs_gamma.append(fish.bvs_gamma.copy())

    def run_simulation(self):

        # Run preset number of integration steps
        for _ in range(self.time_N):
            self.integration_step()

            # Display progress for debugging
            print(self.time_step)

    def simulation(self, output=None, output_Gamma=None):

        # Use passed-in data if given, otherwise fallback to self attributes
        output = output if output is not None else self.output
        output_Gamma = output_Gamma if output_Gamma is not None else self.output_Gamma

        # Preset vortex color maps
        all_Gamma = np.concatenate(output_Gamma)
        vmin = np.min(all_Gamma)
        vmax = np.max(all_Gamma)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('seismic') 

        # Define figure
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot fishies at each time
        for ii in range(0, self.time_N, 10):

            # Clear image and update state at each time step
            ax.clear()
            current_state = output[ii]
            current_Gamma = output_Gamma[ii]

            # Plot each fish configuration
            start = 0
            for fish in self.fishies:

                end = start + 2*fish.N

                # Get node positions from ODE output
                positions = current_state[start:end].reshape((2,fish.N), order='F')

                # Calculate centerline segment positions and orientations
                midpoints = (positions[:,0:fish.N-1] + positions[:,1:fish.N])/2
                edges = positions[:,1:fish.N] - positions[:,0:fish.N-1]
                tangents = edges/np.linalg.norm(edges, axis=0)
                normals = np.vstack([-tangents[1,:], tangents[0,:]])

                # Caluclate rectangular segment locations 
                southwestPoints = midpoints - 0.5*tangents*fish.lEdgeRef.T - 0.5*normals*fish.hEdgeRef.T
                northwestPoints = midpoints - 0.5*tangents*fish.lEdgeRef.T + 0.5*normals*fish.hEdgeRef.T
                northeastPoints = midpoints + 0.5*tangents*fish.lEdgeRef.T + 0.5*normals*fish.hEdgeRef.T
                southeastPoints = midpoints + 0.5*tangents*fish.lEdgeRef.T - 0.5*normals*fish.hEdgeRef.T

                # Draw rectangular segments
                for jj in range(fish.N-1):
                    rectangleX = [southwestPoints[0, jj], northwestPoints[0, jj],
                                northeastPoints[0, jj], southeastPoints[0, jj]]
                    rectangleY = [southwestPoints[1, jj], northwestPoints[1, jj],
                                northeastPoints[1, jj], southeastPoints[1, jj]]
                    ax.fill(rectangleX, rectangleY, 'k') #, label='Segments' if jj == 0 else "")

                start = end + 2*fish.N

            # Plot free vortex street
            vortex_data = current_state[start:]
            num_vortices = vortex_data.size // 2
            free_vortices = vortex_data.reshape((2,num_vortices), order='F')
            sc = ax.scatter(free_vortices[0,:], free_vortices[1,:], c=current_Gamma, 
            cmap=cmap, norm=norm, s=10)

            # Plotting parameters
            plt.xlim(-15.5,2.0)
            plt.ylim(-0.75,0.75)
            ax.set_aspect('equal')
            ax.grid(True)
            # ax.legend()
            plt.pause(0.1)

    # TODO: this to save simulation as a video
    def save_animation(self, output=None, output_Gamma=None):

        # Use passed-in data if given, otherwise fallback to self attributes
        output = output if output is not None else self.output
        output_Gamma = output_Gamma if output_Gamma is not None else self.output_Gamma

        # Preset vortex color maps
        all_Gamma = np.concatenate(output_Gamma)
        vmin = np.min(all_Gamma)
        vmax = np.max(all_Gamma)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('seismic') 

        # Define figure
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot fishies at each time
        for ii in range(0, self.time_N, 2):

            # Clear image and update state at each time step
            ax.clear()
            current_state = output[ii]
            current_Gamma = output_Gamma[ii]

            # Plot each fish configuration
            start = 0
            for fish in self.fishies:

                end = start + 2*fish.N

                # Get node positions from ODE output
                positions = current_state[start:end].reshape((2,fish.N), order='F')

                # Calculate centerline segment positions and orientations
                midpoints = (positions[:,0:fish.N-1] + positions[:,1:fish.N])/2
                edges = positions[:,1:fish.N] - positions[:,0:fish.N-1]
                tangents = edges/np.linalg.norm(edges, axis=0)
                normals = np.vstack([-tangents[1,:], tangents[0,:]])

                # Caluclate rectangular segment locations 
                southwestPoints = midpoints - 0.5*tangents*fish.lEdgeRef.T - 0.5*normals*fish.hEdgeRef.T
                northwestPoints = midpoints - 0.5*tangents*fish.lEdgeRef.T + 0.5*normals*fish.hEdgeRef.T
                northeastPoints = midpoints + 0.5*tangents*fish.lEdgeRef.T + 0.5*normals*fish.hEdgeRef.T
                southeastPoints = midpoints + 0.5*tangents*fish.lEdgeRef.T - 0.5*normals*fish.hEdgeRef.T

                # Draw rectangular segments
                for jj in range(fish.N-1):
                    rectangleX = [southwestPoints[0, jj], northwestPoints[0, jj],
                                northeastPoints[0, jj], southeastPoints[0, jj]]
                    rectangleY = [southwestPoints[1, jj], northwestPoints[1, jj],
                                northeastPoints[1, jj], southeastPoints[1, jj]]
                    ax.fill(rectangleX, rectangleY, 'k') #, label='Segments' if jj == 0 else "")

                start = end + 2*fish.N

            # Plot free vortex street
            vortex_data = current_state[start:]
            num_vortices = vortex_data.size // 2
            free_vortices = vortex_data.reshape((2,num_vortices), order='F')
            sc = ax.scatter(free_vortices[0,:], free_vortices[1,:], c=current_Gamma, 
            cmap=cmap, norm=norm, s=25)

            # Plotting parameters
            plt.xlim(-8,0.5)
            plt.ylim(-0.5,0.5)
            ax.set_aspect('equal')
            ax.grid(True)
            # ax.legend()
            plt.pause(0.1)
