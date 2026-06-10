import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
from matplotlib import colors

import imageio
from io import BytesIO
from PIL import Image
import gc

class Environment:

    def __init__(self, fishies, controller=True):

        # Integration parameters
        self.time_step = 0
        self.time_N = 8001
        self.t_max = 40
        self.delta_T = self.t_max/(self.time_N - 1)

        # Define fishies in the water
        self.fishies = fishies
        self.fish_N = len(self.fishies)
        for fish in self.fishies:
            fish.time_N = self.time_N
            fish.delta_T = self.delta_T
            fish.period_steps = round(2 * np.pi / fish.wave_frequency / fish.delta_T)
            fish.heading = np.zeros((fish.time_N))
            fish.control_inputs = np.zeros((fish.time_N))
        
        # Store free vortex street (FVS)
        self.fvs_N = np.zeros(self.fish_N, dtype=int)
        self.fvs_positions = [np.empty((2,0)) for _ in range(self.fish_N)]
        self.fvs_velocities = [np.empty((2,0)) for _ in range(self.fish_N)]
        self.fvs_Gamma = [np.array([]) for _ in range(self.fish_N)]

        # FVS dissapation parameters
        self.fvs_shedtime = [np.array([]) for _ in range(self.fish_N)]
        fvs_dissapation_time = 2
        self.fvs_max_time = np.round(fvs_dissapation_time/self.delta_T)

        # Newly shed vortex information
        self.new_fvs_positions = np.zeros((2,self.fish_N))
        self.new_fvs_Gamma_dot = np.zeros((self.fish_N))

        # Find the size of the initial state
        self.fish_state_size = 0
        for _, fish in enumerate(self.fishies):
            self.fish_state_size += 4*fish.N

        # Define the initial state for integration
        self.init_state = np.zeros((self.fish_state_size))
        start = 0
        for fish in self.fishies:
            end = start + 4*fish.N
            self.init_state[start:end] = np.concatenate([fish.positions.reshape((2*fish.N), order='F'),
                                                         fish.velocities.reshape((2*fish.N), order='F')])
            start = end
        
        # Store if control is on
        self.control = controller

        # Store outputs of integration
        self.output = []
        self.output_Gamma = []
        self.output_bvs = []
        self.output_bvs_gamma = []

    def fvs_flow_velocity(self, z, delta=0.2):

        # Extract x and y positions from targets
        x = z[0, :]
        y = z[1, :]

        # Initialzie velocity contributions
        u_x = np.zeros_like(x)
        u_y = np.zeros_like(y)

        for ii in range(self.fish_N):
            
            # Extract current FVS positions
            fvs_positions = self.fvs_positions[ii]

            # Extract x and y positions from free vortices
            v_x = fvs_positions[0, :]
            v_y = fvs_positions[1, :]

            # Compute pairwise distances
            dx = x[np.newaxis, :] - v_x[:, np.newaxis]
            dy = y[np.newaxis, :] - v_y[:, np.newaxis]
            norm2 = dx**2 + dy**2 + delta**2

            # Strength of each vortex
            strength = self.fvs_Gamma[ii] / (2 * np.pi)
            strength_array = strength[:, np.newaxis]

            # Accumulate velocity contributions
            u_x += np.sum(-strength_array * dy / norm2, axis=0)
            u_y += np.sum( strength_array * dx / norm2, axis=0)

        return np.vstack((u_x, u_y))
    
    def bvs_flow_velocity(self, z, delta=0.01):

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

    def bvs_exclude_current(self, z, current_fish, delta=0.01):

        # Extract x and y positions from targets
        x = z[0, :]
        y = z[1, :]
        n_targets = x.size

        # Initialize velocity vectors
        u_x = np.zeros(n_targets)
        u_y = np.zeros(n_targets)

        # Loop over all fish
        for fish in self.fishies:

            if fish == current_fish:
                continue

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
    
    def external_flow_velocity(self, z):

        x = z[0, :]
        y = z[1, :]

        u_x = 0.0 * np.ones_like(x)
        u_y = np.zeros_like(y)

        return np.vstack((u_x, u_y))

    def compute_fvs_convection(self):
        
        for ii in range(self.fish_N):
            fvs_flow = self.fvs_flow_velocity(self.fvs_positions[ii])
            bvs_flow = self.bvs_flow_velocity(self.fvs_positions[ii])
            ext_flow = self.external_flow_velocity(self.fvs_positions[ii])
            self.fvs_velocities[ii] = fvs_flow + bvs_flow + ext_flow

    def compute_bvs_gamma(self):
        
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
        flow_at_midpoints = self.fvs_flow_velocity(midpoints) + self.external_flow_velocity(midpoints)
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
        for kk in range(self.fish_N):

            # Keep index tracker
            end = start + bvs_N[kk]

            # Add to set of linear equations
            A[kk + nonpenetration_conditions, start:end] = bvs_lengths[start:end]
            B[kk + nonpenetration_conditions] = -np.sum(self.fvs_Gamma[kk])

            # Update index tracker
            start = end

        # Solve system and update vortex densities
        if np.linalg.matrix_rank(A) < total_BVS:
            gammas = np.zeros((total_BVS))
        else:
            gammas =  np.linalg.solve(A,B)
            start = 0
            for fish in self.fishies:
                end = start + fish.bvs_N
                fish.bvs_gamma = gammas[start:end]
                start = end

    def compute_flow_influence(self):

        for fish in self.fishies:
        
            # Caluclate rectangular segment locations
            midpoints = (fish.positions[:,0:fish.N-1] + fish.positions[:, 1:fish.N])/2
            southwest_points = midpoints - 0.5 * fish.t * fish.l_edge_ref.T - 0.5 * fish.n * fish.h_edge_ref.T
            northwest_points = midpoints - 0.5 * fish.t * fish.l_edge_ref.T + 0.5 * fish.n * fish.h_edge_ref.T
            northeast_points = midpoints + 0.5 * fish.t * fish.l_edge_ref.T + 0.5 * fish.n * fish.h_edge_ref.T
            southeast_points = midpoints + 0.5 * fish.t * fish.l_edge_ref.T - 0.5 * fish.n * fish.h_edge_ref.T

            # Store external flow averaged around segment
            edge_boundary_velocities = np.zeros((2,fish.N-1))
            n_left_right = 20
            n_top_down = 10

            # Iterate over segments
            for ii in range(fish.N-1):

                # Discretize edges
                left_edge = np.array([
                    np.linspace(southwest_points[0,ii], northwest_points[0,ii], n_top_down),
                    np.linspace(southwest_points[1,ii], northwest_points[1,ii], n_top_down)
                ])
                top_edge = np.array([
                    np.linspace(northwest_points[0,ii], northeast_points[0,ii], n_left_right),
                    np.linspace(northwest_points[1,ii], northeast_points[1,ii], n_left_right)
                ])
                right_edge = np.array([
                    np.linspace(northeast_points[0,ii], southeast_points[0,ii], n_top_down),
                    np.linspace(northeast_points[1,ii], southeast_points[1,ii], n_top_down)
                ])
                bottom_edge = np.array([
                    np.linspace(southeast_points[0,ii], southwest_points[0,ii], n_left_right),
                    np.linspace(southeast_points[1,ii], southwest_points[1,ii], n_left_right)
                ])
                
                # Average velocity across edges
                boundary_points = np.hstack([left_edge, top_edge, right_edge, bottom_edge])
                boundary_velocities = self.fvs_flow_velocity(boundary_points) + self.external_flow_velocity(boundary_points) + self.bvs_exclude_current(boundary_points, fish)
                edge_boundary_velocities[:, ii] = np.mean(boundary_velocities, axis=1)
            
            node_boundary_velocities = (np.hstack([edge_boundary_velocities, np.zeros((2, 1))]) + np.hstack([np.zeros((2, 1)), edge_boundary_velocities])) / 2
            fish.external_velocity = node_boundary_velocities
    
    def shed_vortices(self):

        # Check if a vortex has been shed
        if self.time_step != 0:

            # Iterate over each wake
            for ii in range(self.fish_N):

                # Shed past new vortex and update FVS lists
                self.fvs_velocities[ii] = np.hstack([self.fvs_velocities[ii], np.zeros((2, 1))])
                self.fvs_positions[ii] = np.hstack([self.fvs_positions[ii], self.new_fvs_positions[:, ii:ii+1]])
                self.fvs_Gamma[ii] = np.append(self.fvs_Gamma[ii], self.delta_T * self.new_fvs_Gamma_dot[ii])
                self.fvs_shedtime[ii] = np.append(self.fvs_shedtime[ii], self.time_step)

                # Find vortices that have dissapated
                fvs_existance_time = self.time_step - self.fvs_shedtime[ii] 
                not_dissipated_mask = (fvs_existance_time < self.fvs_max_time).flatten()

                # Eliminate old vortices from the list
                self.fvs_positions[ii] = self.fvs_positions[ii][:, not_dissipated_mask]
                self.fvs_velocities[ii] = self.fvs_velocities[ii][:, not_dissipated_mask]
                self.fvs_Gamma[ii] = self.fvs_Gamma[ii][not_dissipated_mask]
                self.fvs_shedtime[ii] = self.fvs_shedtime[ii][not_dissipated_mask]
                self.fvs_N[ii] = self.fvs_positions[ii].shape[1]

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
            self.new_fvs_positions[:,ii] = fish.bvs_positions[:, -1]
            self.new_fvs_Gamma_dot[ii] = -(u_minus ** 2 - u_plus ** 2)/2

            # Brute force correction
            if self.time_step < np.round(0.5/self.delta_T):
                self.new_fvs_Gamma_dot[ii] = -(self.time_step/np.round(0.5/self.delta_T))**2 * (u_minus ** 2 - u_plus ** 2)/2

    def state_derivatives(self, t, z0):

        # Preallocate space for derivatives
        dzdt = np.zeros_like(z0)
        
        # Find fish configuration derivatives 
        start = 0
        for fish in self.fishies:
            end = start + 4*fish.N
            if self.control:
                u = fish.heading_controller() # closed-loop 
            else:
                u = fish.wave_offset # open-loop
            dzdt[start:end] = fish.internal_forces(t, z0[start:end], u)
            start = end

        # Find FVS positions derivatives
        if self.fvs_N.any() > 0:
            fvs_start = start
            for ii in range(self.fish_N):
                end = start + 2 * self.fvs_N[ii]
                self.fvs_positions[ii] = z0[start:end].reshape((2,self.fvs_N[ii]), order='F')
                start = end
            self.compute_fvs_convection()
            start = fvs_start
            for ii in range(self.fish_N):
                end = start + 2 * self.fvs_N[ii]
                dzdt[start:end] = self.fvs_velocities[ii].reshape((2*self.fvs_N[ii]), order='F')
                start = end
        
        # Return state derivative
        return dzdt

    def integration_step(self):

        # Integrate forward one time step
        t_span = [self.delta_T*self.time_step, self.delta_T*(self.time_step + 1)]
        step = solve_ivp(self.state_derivatives, t_span, self.init_state, method='RK45', rtol=1e-3, atol=1e-6)
        
        # Shed new vortices at end of integration step (changes self.fvs_positions)
        self.compute_bvs_gamma()
        self.compute_flow_influence()
        self.shed_vortices()

        # Update initial conditions
        fish_config = step.y[0:self.fish_state_size, -1]
        total_fvs = int(np.sum(self.fvs_N))
        if total_fvs > 0:
            fvs_config = np.hstack(self.fvs_positions).reshape((2*total_fvs), order='F')
        else:
            fvs_config = np.array([])
        self.init_state = np.concatenate((fish_config,fvs_config))
        self.time_step += 1
        for fish in self.fishies:
            fish.time_step = self.time_step
        
        # Store outputs in array
        self.output.append(self.init_state.copy())
        self.output_Gamma.append(np.hstack(self.fvs_Gamma).copy())
        for fish in self.fishies:
            self.output_bvs.append(fish.bvs_positions.copy())
            self.output_bvs_gamma.append(fish.bvs_gamma.copy())

    def run_simulation(self):

        # Run preset number of integration steps
        for _ in range(self.time_N):
            self.integration_step()

            # Display progress for debugging
            print(self.time_step)

    def show_simulation(self, output=None, output_Gamma=None, output_bvs=None, output_bvs_gamma=None, show_bvs=False):

        # Use passed-in data if given, otherwise fallback to self attributes
        output = output if output is not None else self.output
        output_Gamma = output_Gamma if output_Gamma is not None else self.output_Gamma
        output_bvs = output_bvs if output_bvs is not None else self.output_bvs
        output_bvs_gamma = output_bvs_gamma if output_bvs_gamma is not None else self.output_bvs_gamma

        # Preset vortex color maps
        all_Gamma = np.concatenate(output_Gamma)
        if all_Gamma.size == 0:
            vmin = -1
            vmax = 1
        else:
            vmin = np.min(all_Gamma)
            vmax = np.max(all_Gamma)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('seismic')

        # Define figure
        fig, ax = plt.subplots(figsize=(16, 9))
        plt.grid(True, color='gray', linestyle=':', linewidth=0.5, zorder=0)
        
        # Plot fishies at each time
        for ii in range(0, 4001, 16): #self.time_N, 16):

            # Clear image and update state at each time step
            ax.clear()
            current_state = output[ii]
            current_Gamma = output_Gamma[ii]
            bvs_index = self.fish_N * ii

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
                southwestPoints = midpoints - 0.5*tangents*fish.l_edge_ref.T - 0.5*normals*fish.h_edge_ref.T
                northwestPoints = midpoints - 0.5*tangents*fish.l_edge_ref.T + 0.5*normals*fish.h_edge_ref.T
                northeastPoints = midpoints + 0.5*tangents*fish.l_edge_ref.T + 0.5*normals*fish.h_edge_ref.T
                southeastPoints = midpoints + 0.5*tangents*fish.l_edge_ref.T - 0.5*normals*fish.h_edge_ref.T

                # Draw rectangular segments
                for jj in range(fish.N-1):
                    rectangleX = [southwestPoints[0, jj], northwestPoints[0, jj],
                                northeastPoints[0, jj], southeastPoints[0, jj]]
                    rectangleY = [southwestPoints[1, jj], northwestPoints[1, jj],
                                northeastPoints[1, jj], southeastPoints[1, jj]]
                    ax.fill(rectangleX, rectangleY, 'k', zorder=2) #, label='Segments' if jj == 0 else "")

                start = end + 2*fish.N

                # Draw true and desired headings
                if self.control:
                    true_direction = positions[:, 0] - positions[:, 2]
                    quiver_true = true_direction / np.linalg.norm(true_direction)
                    quiver_desired = fish.fish_length * fish.desired_heading_vector
                    plt.quiver(positions[0, 0], positions[1, 0], quiver_true[0], quiver_true[1], 
                            angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=6, headlength=8, color='red')
                    plt.quiver(positions[0, 0], positions[1, 0], quiver_desired[0], quiver_desired[1],
                            angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=6, headlength=8, color='blue')

            # Plot free vortex street
            vortex_data = current_state[start:]
            num_vortices = vortex_data.size // 2
            free_vortices = vortex_data.reshape((2,num_vortices), order='F')
            sc = ax.scatter(free_vortices[0,:], free_vortices[1,:], c=current_Gamma, 
            cmap=cmap, norm=norm, s=10, zorder=1)

            # Plot bound vortex sheets
            if show_bvs:
                for fish in self.fishies:
                    current_bvs = output_bvs[bvs_index]
                    current_bvs_gamma = output_bvs_gamma[bvs_index]
                    sc_bound = ax.scatter(current_bvs[0,:], current_bvs[1,:], c=current_bvs_gamma*fish.bvs_length, 
                    cmap=cmap, norm=norm, s=10, zorder=3)
                    bvs_index += 1

            # Plotting parameters
            plt.xlim(-28.0,10.0)
            plt.ylim(-4.0,4.0)
            ax.set_aspect('equal')
            ax.grid(True)
            # ax.legend()
            plt.pause(0.1)

    def save_animation(self, video_filename, output=None, output_Gamma=None, output_bvs=None, output_bvs_gamma=None, show_bvs=False):

        # Video parameters
        num_frames = len(output)
        fps = 25

        # Use passed-in data if given, otherwise fallback to self attributes
        output = output if output is not None else self.output
        output_Gamma = output_Gamma if output_Gamma is not None else self.output_Gamma
        output_bvs = output_bvs if output_bvs is not None else self.output_bvs
        output_bvs_gamma = output_bvs_gamma if output_bvs_gamma is not None else self.output_bvs_gamma

        # Preset vortex color maps
        all_Gamma = np.concatenate(output_Gamma)
        if all_Gamma.size == 0:
            vmin = -1
            vmax = 1
        else:
            vmin = np.min(all_Gamma)
            vmax = np.max(all_Gamma)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('seismic')

        # Open video writer
        with imageio.get_writer(video_filename, fps=fps) as writer:

            # Define figure
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=600)
            plt.grid(True, color='gray', linestyle=':', linewidth=0.5, zorder=0)
            
            # Plot fishies at each time
            for ii in range(0, 5001, 8): # full speed  # 4): # half speed self.time_N

                # Clear image and update state at each time step
                ax.clear()
                current_state = output[ii]
                current_Gamma = output_Gamma[ii]
                current_bvs = output_bvs[ii]
                current_bvs_gamma = output_bvs_gamma[ii]

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
                    southwestPoints = midpoints - 0.5*tangents*fish.l_edge_ref.T - 0.5*normals*fish.h_edge_ref.T
                    northwestPoints = midpoints - 0.5*tangents*fish.l_edge_ref.T + 0.5*normals*fish.h_edge_ref.T
                    northeastPoints = midpoints + 0.5*tangents*fish.l_edge_ref.T + 0.5*normals*fish.h_edge_ref.T
                    southeastPoints = midpoints + 0.5*tangents*fish.l_edge_ref.T - 0.5*normals*fish.h_edge_ref.T

                    # Draw rectangular segments
                    for jj in range(fish.N-1):
                        rectangleX = [southwestPoints[0, jj], northwestPoints[0, jj],
                                    northeastPoints[0, jj], southeastPoints[0, jj]]
                        rectangleY = [southwestPoints[1, jj], northwestPoints[1, jj],
                                    northeastPoints[1, jj], southeastPoints[1, jj]]
                        ax.fill(rectangleX, rectangleY, 'k', zorder=2) #, label='Segments' if jj == 0 else "")

                    start = end + 2*fish.N

                    # Draw true and desired headings
                    if self.control:
                        true_direction = positions[:, 0] - positions[:, 2]
                        quiver_true = true_direction / np.linalg.norm(true_direction)
                        quiver_desired = fish.fish_length * fish.desired_heading_vector
                        plt.quiver(positions[0, 0], positions[1, 0], quiver_true[0], quiver_true[1], 
                                angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=6, headlength=8, color='red')
                        plt.quiver(positions[0, 0], positions[1, 0], quiver_desired[0], quiver_desired[1],
                                angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=6, headlength=8, color='blue')

                # Plot free vortex street
                vortex_data = current_state[start:]
                num_vortices = vortex_data.size // 2
                free_vortices = vortex_data.reshape((2,num_vortices), order='F')
                sc = ax.scatter(free_vortices[0,:], free_vortices[1,:], c=current_Gamma, 
                cmap=cmap, norm=norm, s=10, zorder=1)

                # Plot bound vortex sheets
                if show_bvs:
                    sc_bound = ax.scatter(current_bvs[0,:], current_bvs[1,:], c=current_bvs_gamma, 
                    cmap=cmap, norm=norm, s=10)

                # Plotting parameters
                plt.xlim(-28.0,10.0)
                plt.ylim(-4.0,6.0)
                # plt.xlim(-40.0,8.0)
                # plt.ylim(-8,8)
                ax.set_aspect('equal')
                ax.grid(True)
                # ax.legend()

                # Save plot to in-memory buffer
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                # plt.close(fig)

                # Load image from buffer using PIL, then convert to array
                image = Image.open(buf).copy()
                writer.append_data(np.array(image))
                buf.close()

                # Force garbage collection every few frames
                if ii % 20 == 0:
                    gc.collect()

                # Print to update progress
                print(f"Generated frame {ii + 1}/{num_frames}")
