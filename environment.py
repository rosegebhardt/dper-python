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

    def __init__(self, fishies):

        # Integration parameters
        self.time_step = 0
        self.time_N = 2001
        self.t_max = 10
        self.delta_T = self.t_max/(self.time_N - 1)

        # Define fishies in the water
        self.fishies = fishies
        self.fish_N = len(self.fishies)
        for fish in self.fishies:
            fish.time_N = self.time_N
            fish.delta_T = self.delta_T
            fish.period_steps = round(2 * np.pi / fish.wave_frequency / fish.delta_T)
            fish.heading = np.zeros((fish.time_N))
        
        # Store free vortex street (FVS)
        self.fvs_N = 0
        self.fvs_positions = np.empty((2,0)) # 2 by fvs_N
        self.fvs_velocities = np.empty((2,0)) # 2 by fvs_N
        self.fvs_Gamma = np.array([]) # fvs_N,

        # FVS dissapation parameters
        self.fvs_shedtime = np.array([]) # fvs_N,
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
        
        # Store outputs of integration
        self.output = []
        self.output_Gamma = []
        self.output_bvs = []
        self.output_bvs_gamma = []

    def fvs_flow_velocity(self, z, delta=0.2):

        # Extract x and y positions from targets
        x = z[0, :]
        y = z[1, :]

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

    def external_flow_velocity(self, z):

        x = z[0, :]
        y = z[1, :]

        u_x = 0.0 * np.ones_like(x)
        u_y = np.zeros_like(y)

        return np.vstack((u_x, u_y))

    def compute_fvs_convection(self):
        
        fvs_flow = self.fvs_flow_velocity(self.fvs_positions)
        bvs_flow = self.bvs_flow_velocity(self.fvs_positions)
        ext_flow = self.external_flow_velocity(self.fvs_positions)
        self.fvs_velocities = fvs_flow + bvs_flow + ext_flow

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
                    np.linspace(southwest_points[0,ii], northwest_points[0,ii], n_left_right),
                    np.linspace(southwest_points[1,ii], northwest_points[1,ii], n_left_right)
                ])
                top_edge = np.array([
                    np.linspace(northwest_points[0,ii], northeast_points[0,ii], n_top_down),
                    np.linspace(northwest_points[1,ii], northeast_points[1,ii], n_top_down)
                ])
                right_edge = np.array([
                    np.linspace(northeast_points[0,ii], southeast_points[0,ii], n_left_right),
                    np.linspace(northeast_points[1,ii], southeast_points[1,ii], n_left_right)
                ])
                bottom_edge = np.array([
                    np.linspace(southeast_points[0,ii], southwest_points[0,ii], n_top_down),
                    np.linspace(southeast_points[1,ii], southwest_points[1,ii], n_top_down)
                ])
                
                # Average velocity across edges
                boundary_points = np.hstack([left_edge, top_edge, right_edge, bottom_edge])
                boundary_velocities = self.fvs_flow_velocity(boundary_points) + self.external_flow_velocity(boundary_points) #+ self.bvs_flow_velocity(boundary_points)
                edge_boundary_velocities[:, ii] = np.mean(boundary_velocities, axis=1)
            
            node_boundary_velocities = (np.hstack([edge_boundary_velocities, np.zeros((2, 1))]) + np.hstack([np.zeros((2, 1)), edge_boundary_velocities])) / 2
            fish.external_velocity = node_boundary_velocities
    
    def shed_vortices(self):

        # Check if a vortex has been shed
        if self.time_step != 0:

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
            # u = fish.wave_offset # open-loop
            u = fish.controller() # closed-loop 
            dzdt[start:end] = fish.internal_forces(t, z0[start:end], u)
            start = end

        # Find FVS positions derivatives
        self.fvs_positions = z0[end:].reshape((2,self.fvs_N), order='F')
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
        self.compute_bvs_gamma()
        self.compute_flow_influence()
        self.shed_vortices()

        # Update initial conditions
        fish_config = step.y[0:self.fish_state_size, -1]
        fvs_config = self.fvs_positions.reshape((2*self.fvs_N), order='F')
        self.init_state = np.concatenate((fish_config,fvs_config))
        self.time_step += 1
        for fish in self.fishies:
            fish.time_step = self.time_step
        
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
        
        # Plot fishies at each time
        for ii in range(0, self.time_N, 16):

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
                    ax.fill(rectangleX, rectangleY, 'k') #, label='Segments' if jj == 0 else "")

                start = end + 2*fish.N

            # Plot free vortex street
            vortex_data = current_state[start:]
            num_vortices = vortex_data.size // 2
            free_vortices = vortex_data.reshape((2,num_vortices), order='F')
            sc = ax.scatter(free_vortices[0,:], free_vortices[1,:], c=current_Gamma, 
            cmap=cmap, norm=norm, s=10, zorder=0)

            # Plot bound vortex sheets
            if show_bvs:
                sc_bound = ax.scatter(current_bvs[0,:], current_bvs[1,:], c=current_bvs_gamma, 
                cmap=cmap, norm=norm, s=10)

            # Plotting parameters
            plt.xlim(-15.0,2.0)
            plt.ylim(-2.0,8.0)
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
            
            # Plot fishies at each time
            for ii in range(0, self.time_N, 8):

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
                        ax.fill(rectangleX, rectangleY, 'k') #, label='Segments' if jj == 0 else "")

                    start = end + 2*fish.N

                # Plot free vortex street
                vortex_data = current_state[start:]
                num_vortices = vortex_data.size // 2
                free_vortices = vortex_data.reshape((2,num_vortices), order='F')
                sc = ax.scatter(free_vortices[0,:], free_vortices[1,:], c=current_Gamma, 
                cmap=cmap, norm=norm, s=10, zorder=0)

                # Plot bound vortex sheets
                if show_bvs:
                    sc_bound = ax.scatter(current_bvs[0,:], current_bvs[1,:], c=current_bvs_gamma, 
                    cmap=cmap, norm=norm, s=10)

                # Plotting parameters
                plt.xlim(-12.0,2.0)
                plt.ylim(-2.0,2.0)
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
