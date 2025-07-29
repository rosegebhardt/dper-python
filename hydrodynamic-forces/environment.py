import numpy as np
from numpy.linalg import pinv
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
# type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
from matplotlib import colors
import matplotlib.animation as animation
import time

from fish import Fish
# import hydrodynamics

class Environment:

    def __init__(self, fishies):

        # Integration parameters
        self.time_step = 0
        self.time_N = 4001
        self.t_max = 40
        self.delta_T = self.t_max/(self.time_N - 1)

        # Define fishies in the water
        self.fishies = fishies
        self.fish_N = len(self.fishies)

        # Compute total number of bound vortices
        self.bvs_N = np.array([fish.bvs_N for fish in self.fishies])        
        self.total_BVS = np.sum(self.bvs_N)
        self.nonpenetration_conditions = self.total_BVS - self.fish_N

        # Compute bound vortex positions
        self.x = np.zeros((2, self.total_BVS))
        self.v = np.zeros((2, self.total_BVS))
        self.bvs_gammas = np.zeros(self.total_BVS)
        self.bvs_lengths = np.zeros(self.total_BVS)

        # Compute bound vortices geometry
        self.n = np.zeros((2, self.nonpenetration_conditions))
        self.n_dot = np.zeros((2, self.nonpenetration_conditions))
        self.midpoints = np.zeros((2, self.nonpenetration_conditions))
        self.midpoint_velocities = np.zeros((2, self.nonpenetration_conditions))

        # Compute bound vortex distances
        self.dx = self.midpoints[0, :][:, np.newaxis] - self.x[0, :]
        self.dy = self.midpoints[1, :][:, np.newaxis] - self.x[1, :]
        self.dx_dot = self.midpoint_velocities[0, :][:, np.newaxis] - self.v[0, :]
        self.dy_dot = self.midpoint_velocities[1, :][:, np.newaxis] - self.v[1, :]
        self.norm2 = self.dx ** 2 + self.dy ** 2
        self.norm2[self.norm2 < 1e-12] = 1e-12

        # Update bound vortex values
        self.compute_bvs_geometry()

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
        
        # Define initial hydrodynamic forces
        self.hydrodynamic_forces = self.compute_hydrodynamics()

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
            The induced velocity field at each of the target points.
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
            The induced velocity field at each of the target points.
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

    def fvs_flow_acceleration(self, z, z_dot, delta=0.2):

        """
        Compute the time derivative of a 2D velocity field induced by a collection of fish-like 
        objects using the Biot-Savart law for free vortex sheets (FVS).

        Parameters
        ----------
        z : ndarray of shape (2, n_targets)
            Positions of target points where acceleration is evaluated (x and y components).
            - z[0, :] contains x-coordinates
            - z[1, :] contains y-coordinates
        
        z_dot : ndarray of shape (2, n_target)
            Velocities of the target points (x_dot and y_dot components).
            - z_dot[0, :] contains x-velocities
            - z_dot[1, :] contains y-velocities

        delta : float, optional (default=0.2)
            Regularization parameter added to the squared distance to avoid singularities.
            Useful for numerical stability, especially when points are close.

        Returns
        -------
        acceleration : ndarray of shape (2, n_targets)
            The induced acceleration field at each of the target points.
            - acceleration[0, :] contains the x-component (accel_x)
            - acceleration[1, :] contains the y-component (accel_y)
        """

        # Extract x and y positions and velocities from targets
        x = z[0, :]
        y = z[1, :]
        x_dot = z_dot[0, :]
        y_dot = z_dot[1, :]

        # Extract x and y positions and velocities from free vortices
        x_v = self.fvs_positions[0, :]
        y_v = self.fvs_positions[1, :]
        x_dot_v = self.fvs_velocities[0, :]
        y_dot_v = self.fvs_velocities[1, :]

        # Extract x and y positions and velocities from new vortices
        x_tail = self.new_fvs_positions[0, :]
        y_tail = self.new_fvs_positions[1, :]

        # Compute pairwise distances from free vortices 
        dx = x - x_v[:, np.newaxis]
        dy = y - y_v[:, np.newaxis]
        dx_dot = x_dot - x_dot_v[:, np.newaxis]
        dy_dot = y_dot - y_dot_v[:, np.newaxis]
        norm2 = dx ** 2 + dy ** 2 + delta ** 2
        norm2_dot = 2 * (dx * dx_dot + dy * dy_dot)

        # Compute pairwise distances from new vortices
        dx_tail = x - x_tail[:, np.newaxis]
        dy_tail = y - y_tail[:, np.newaxis]
        norm2_tail = dx_tail ** 2 + dy_tail ** 2 + delta ** 2

        # Accumulate contributions of free vortices
        strength = self.fvs_Gamma[:, np.newaxis] / (2 * np.pi)
        accel_x = np.sum(strength * (-dy_dot * norm2 + dy * norm2_dot) / norm2, axis=0)
        accel_y = np.sum(strength * ( dx_dot * norm2 - dx * norm2_dot) / norm2, axis=0)

        # Accumulate contributions of new vortices
        strength_tail = self.new_fvs_Gamma_dot[:, np.newaxis] / (2 * np.pi)
        accel_x += np.sum(-strength_tail * dy_tail / norm2_tail, axis=0)
        accel_y += np.sum( strength_tail * dx_tail / norm2_tail, axis=0)

        # Return derivative of flow due to FVS
        return np.vstack((accel_x, accel_y))

    def compute_fvs_convection(self):
        
        self.fvs_velocities = self.fvs_flow_velocity(self.fvs_positions) + self.bvs_flow_velocity(self.fvs_positions)

    def compute_bvs_geometry(self): 

        # Start index trackers
        node_start = 0
        edge_start = 0

        for ii, fish in enumerate(self.fishies):

            # End index trackers
            node_end = node_start + self.bvs_N[ii]
            edge_end = edge_start + self.bvs_N[ii] - 1

            # Store all node data
            self.x[:,node_start:node_end] = fish.bvs_positions
            self.v[:,node_start:node_end] = fish.bvs_velocities
            self.bvs_gammas[node_start:node_end] = fish.bvs_gamma
            self.bvs_lengths[node_start:node_end] = fish.bvs_length

            # # Brute force correction for initial shedding problems
            # if self.time_step < np.round(0.5/self.delta_T):
            #     self.bvs_gammas[node_start:node_end] = (self.time_step/np.round(0.5/self.delta_T))**2 * fish.bvs_gamma

            # Store all edge data
            e = self.x[:, node_start+1:node_end] - self.x[:,node_start:node_end-1]
            t = e / np.linalg.norm(e, axis=0)
            self.n[:, edge_start:edge_end] = np.array([-t[1,:], t[0,:]])
            t_dot = (self.v[:, node_start+1:node_end] - self.v[:,node_start:node_end-1]) / np.linalg.norm(e, axis=0)
            self.n_dot[:, edge_start:edge_end] = np.array([-t_dot[1,:], t_dot[0,:]])

            self.midpoints[:, edge_start:edge_end] = (self.x[:, node_start:node_end-1] + self.x[:, node_start+1:node_end])/2
            self.midpoint_velocities[:, edge_start:edge_end] = (self.v[:, node_start:node_end-1] + self.v[:, node_start+1:node_end])/2

            # Update index trackers
            node_start = node_end
            edge_start = edge_end

        # Compute distances between sources and midpoints
        self.dx = self.midpoints[0, :][:, np.newaxis] - self.x[0, :]
        self.dy = self.midpoints[1, :][:, np.newaxis] - self.x[1, :]
        self.dx_dot = self.midpoint_velocities[0, :][:, np.newaxis] - self.v[0, :]
        self.dy_dot = self.midpoint_velocities[1, :][:, np.newaxis] - self.v[1, :]
        self.norm2 = self.dx ** 2 + self.dy ** 2
        self.norm2[self.norm2 < 1e-12] = 1e-12

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
        bvs_N = self.bvs_N       
        total_BVS = self.total_BVS
        nonpenetration_conditions = self.nonpenetration_conditions
            
        # Set up linear system
        A = np.zeros((total_BVS,total_BVS))
        B = np.zeros((total_BVS))

        # RHS of non-penetration conditions
        flow_at_midpoints = self.fvs_flow_velocity(self.midpoints)
        B[:nonpenetration_conditions] = np.sum((self.midpoint_velocities - flow_at_midpoints) * self.n, axis=0)

        # LHS of non-penetration conditions
        direction = -self.n[0, :][:, np.newaxis] * self.dy + self.n[1, :][:, np.newaxis] * self.dx
        A[:nonpenetration_conditions,:] = self.bvs_lengths * direction / (2 * np.pi * self.norm2)

        # Kelvin circulation theorem condition
        start = 0
        kelvin_value = -np.sum(self.fvs_Gamma)
        for kk in range(self.fish_N):

            # Keep index tracker
            end = start + bvs_N[kk]

            # Add to set of linear equations
            A[kk + nonpenetration_conditions, start:end] = self.bvs_lengths[start:end]
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

    def compute_hydrodynamics(self):

        # Redefine environment attributes
        bvs_gammas = self.bvs_gammas
        bvs_lengths = self.bvs_lengths
        total_BVS = self.total_BVS

        # Compute bound vortex distances
        dx = self.dx
        dy = self.dy
        midpoints = self.midpoints
        midpoint_velocities = self.midpoint_velocities
        n = self.n
        n_dot = self.n_dot
        dx_dot = self.dx_dot
        dy_dot = self.dy_dot
        norm2 = self.norm2
        
        ########## DYNAMIC NON-PENETRATION CONDITIONS LEFT HAND SIDE ##########

        # Initialize acceleration part of non-penetration conditions
        nonpenetration_LHS_1_blocks = []

        # Create a block for each fish
        for fish in self.fishies:

            current_block = np.zeros((fish.bvs_N-1, 2*fish.bvs_N))
            start = 0

            for ii in range(fish.bvs_N-1):

                end = start + 4
                current_normal = n[:, ii]
                current_block[ii, start:end] = np.hstack((current_normal,current_normal))/2
                start += 2

            nonpenetration_LHS_1_blocks.append(current_block)

        # Acceleration part of non-penetration conditions
        nonpenetration_LHS_1 = block_diag(*nonpenetration_LHS_1_blocks)

        # Gamma rate of change part of non-penetration conditions
        direction = -dy * n[0, :][:, np.newaxis] + dx * n[1, :][:, np.newaxis]
        nonpenetration_LHS_2 = bvs_lengths * direction / (2 * np.pi * norm2)

        ########## DYNAMIC NON-PENETRATION CONDITIONS RIGHT HAND SIDE ##########

        # Known component of BVS acceleration in normal direction
        known_bvs_accel_x = -np.sum(bvs_gammas * bvs_lengths * dy_dot / (2 * np.pi * norm2), axis=1)
        known_bvs_accel_y =  np.sum(bvs_gammas * bvs_lengths * dx_dot / (2 * np.pi * norm2), axis=1)
        known_bvs_accel_normal = known_bvs_accel_x * n[0, :] + known_bvs_accel_y * n[1, :]

        # Component of FVS acceleration in normal direction
        fvs_accel = self.fvs_flow_acceleration(midpoints, midpoint_velocities)
        fvs_accel_normal = fvs_accel[0, :] * n[0, :] + fvs_accel[1, :] * n[1, :]

        # Component of flow velocity relative to midpoints in normal derivative direction
        relative_velocity = self.bvs_flow_velocity(midpoints) + self.fvs_flow_velocity(midpoints) - midpoint_velocities
        relative_velocity_normal = relative_velocity[0, :] * n_dot[0, :] + relative_velocity[1, :] * n_dot[1, :]
        
        # RHS of nonpenetration conditions
        nonpenetration_RHS = known_bvs_accel_normal + fvs_accel_normal + relative_velocity_normal

        ########## DYNAMIC KELVIN CONDITION ##########

        # Zero acceleration component
        kutta_LHS_1 = np.zeros((self.fish_N, 2*total_BVS))

        # Non-zero gamma rate-of-change component
        kutta_LHS_2 = np.zeros((self.fish_N, total_BVS))
        start = 0
        for ii, fish in enumerate(self.fishies):
            end = start + fish.bvs_N
            kutta_LHS_2[ii, start:end] = fish.bvs_length
            start = end
    
        # RHS of Kelvin condition
        # kutta_RHS = np.zeros((self.fish_N))
        kutta_RHS = -sum(self.new_fvs_Gamma_dot) * np.ones((self.fish_N))

        ########## NEWTONS SECOND LAW (N2L) ##########

        # Initialize LHS of linear system
        n2l_LHS_1 = np.zeros((2*total_BVS, 2*total_BVS))
        n2l_LHS_2 = np.zeros((2*total_BVS, total_BVS))
        n2l_RHS = np.zeros((2*total_BVS))

        # Define start indices
        start_height = 0
        start_width = 0
        start_midpoints = 0

        for ii, fish in enumerate(self.fishies):
            
            # Define end indices
            end_height = start_height + 2*fish.bvs_N
            end_width = start_width + fish.bvs_N
            end_midpoints = start_midpoints + fish.bvs_N - 1

            # Compute node normals
            current_normal = n[:, start_midpoints:end_midpoints]
            normal_left_pad = np.hstack([current_normal[:, [0]], current_normal])
            normal_right_pad = np.hstack([current_normal, current_normal[:, [-1]]])
            node_normal = (normal_left_pad + normal_right_pad) / np.linalg.norm(normal_left_pad + normal_right_pad, axis=0)

            # Acceleration component of force balance
            repeat_mass = np.repeat(fish.bvs_mass, 2)
            n2l_LHS_1[start_height:end_height, start_height:end_height] = np.diag(repeat_mass)

            # Gamma rate-of-change component of force balance
            block_values = fish.bvs_length[None, :, None] * node_normal.T[:, None, :]
            block_values = block_values.transpose(2, 0, 1)
            upper_triangle = np.triu(np.ones((fish.bvs_N, fish.bvs_N), dtype=bool))
            triangular_blocks = block_values * upper_triangle[None, :, :]
            n2l_LHS_2[start_height:end_height, start_width:end_width] = triangular_blocks.transpose(1, 0, 2).reshape(2*fish.bvs_N, fish.bvs_N)
            # n2l_LHS_2[start_height:end_height, start_width:end_width] = np.repeat(fish.bvs_length, 2)[:, np.newaxis] * triangular_blocks.transpose(1, 0, 2).reshape(2*fish.bvs_N, fish.bvs_N)

            # Compute velocity jump at each link
            link_tangents = np.array([node_normal[1, :], -node_normal[0, :]])
            bvs_flow = self.bvs_flow_velocity(fish.bvs_positions)
            fvs_flow = self.fvs_flow_velocity(fish.bvs_positions)
            link_flow_velocities = bvs_flow + fvs_flow
            link_tangent_flow = np.sum((link_flow_velocities - fish.bvs_velocities) * link_tangents, axis=0)

            # Solve for pressure difference
            u_minus = link_tangent_flow - fish.bvs_gamma/2
            u_plus  = link_tangent_flow + fish.bvs_gamma/2

            # Solve for RHS
            n2l_RHS_scale = (-(u_minus ** 2 - u_plus ** 2)/2 - self.new_fvs_Gamma_dot[ii]) * fish.bvs_length
            n2l_RHS_direction = n2l_RHS_scale * node_normal
            n2l_RHS[start_height:end_height] = n2l_RHS_direction.reshape((2*fish.bvs_N), order='F')
            # n2l_RHS = np.zeros((2*fish.bvs_N))

            # Update indices
            start_height = end_height
            start_width = end_width
            start_midpoints = end_midpoints
        
        A_accel = np.block([[nonpenetration_LHS_1, nonpenetration_LHS_2],
                            [kutta_LHS_1, kutta_LHS_2],
                            [n2l_LHS_1, n2l_LHS_2]])
        B_accel = np.concatenate((nonpenetration_RHS, kutta_RHS, n2l_RHS))

        # Solve system and return BVS accelerations
        if np.linalg.matrix_rank(A_accel) < 3*total_BVS:
            x_ddot = np.zeros((2*total_BVS))
        else:
            solution = np.linalg.solve(A_accel, B_accel)
            x_ddot = solution[0:2*total_BVS]

        # # Zero out the hydrodynamic forces in the normal direction
        # x_ddot = x_ddot.reshape((2,total_BVS), order='F')
        # x_ddot[1, :] = np.zeros_like(x_ddot[1, :])
        # x_ddot = x_ddot.reshape((2*total_BVS), order='F')

        # # Cap the magnitude of each force
        # x_ddot = x_ddot.reshape((2,total_BVS), order='F')
        # magnitudes = np.linalg.norm(x_ddot, axis=0)
        # x_ddot_hat = x_ddot / (magnitudes + 1e-10)

        # CLIP_VALUE = 1.0
        # magnitudes = np.clip(magnitudes, a_min=0, a_max=CLIP_VALUE)
        # x_ddot = x_ddot_hat * magnitudes
        # x_ddot = x_ddot.reshape((2*total_BVS), order='F')

        return x_ddot

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

            # Brute force correction for initial shedding problems
            if self.time_step < np.round(0.5/self.delta_T):
                self.new_fvs_Gamma_dot[ii] = -(self.time_step/np.round(0.5/self.delta_T))**2 * (u_minus ** 2 - u_plus ** 2)/2

    def state_derivatives(self, t, z0):

        # Preallocate space for derivatives
        dzdt = np.zeros_like(z0)
        
        # Find fish configuration derivatives 
        start = 0
        start_BVS = 0

        for fish in self.fishies:

            end = start + 4*fish.N
            end_BVS = start + 2*fish.bvs_N

            mesh_hydrodynamics = self.hydrodynamic_forces[start_BVS:end_BVS].reshape((2,fish.bvs_N), order='F')
            dzdt[start:end] = fish.state_derivative(t, z0[start:end], mesh_hydrodynamics)
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
        
        # Update hydrodynamics and shed new vortices at end of integration step
        self.compute_bvs_geometry()
        self.compute_bvs_gamma()
        self.hydrodynamic_forces = self.compute_hydrodynamics()
        # self.shed_vortices()

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
        for ii in range(0, self.time_N, 5):

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
            plt.xlim(-25.0,2.0)
            # plt.ylim(-0.75,0.75)
            # plt.xlim(-10,2)
            plt.ylim(-2,2)
            ax.set_aspect('equal')
            ax.grid(False)
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
