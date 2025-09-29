import numpy as np
from numpy.linalg import pinv # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import json

class Fish:

    def __init__(self, config_file):

        # Load configuration from JSON file
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Fish geometry parameters
        self.N = config['N']
        self.fish_length = config['fish_length']
        self.bvs_N = config['bvs_N']
        delta_0 = config['delta_0']
        r0 = self.fish_length * (1 - delta_0) / 4

        # Time parameters
        self.time_N = 0
        self.delta_T = 0
        self.time_step = 0
        self.period_steps = 10

        # Control parameters 
        self.heading = np.deg2rad(config['initial_heading'])
        self.desired_heading = np.deg2rad(config['desired_heading'])
        self.max_offset_rate = config['max_offset_rate']
        self.past_offset = 0

        # Wave propagation parameters
        self.wave_number = config['normalized_wave_number'] / self.fish_length
        self.wave_frequency = config['wave_frequency']
        self.wave_amplitude = config['wave_amplitude']
        self.wave_phase = config['wave_phase']
        self.wave_offset = config['wave_offset']
        self.c1 = config['c1']
        self.c2 = config['c2']

        # Material parameters 
        self.E = config['E']
        rho = config['rho']
        self.d_b = config['d_b']
        self.d_s = config['d_s']

        # Fluid parameters
        self.fluid_density = config['fluid_density']
        self.external_velocity = np.zeros((2, self.N))
        self.CF = config['CF'] # tangent drag coefficient
        self.CD = config['CD'] # normal drag coefficient
        self.CA = config['CA'] # added mass coefficient

        # Edge properties
        self.e = np.zeros((2,self.N-1))
        self.t = np.zeros((2,self.N-1))
        self.n = np.zeros((2,self.N-1))
        self.l_edge = np.zeros(self.N-1)

        # Form fish polygon
        angle = np.linspace(0, 2*np.pi, 1000)
        circle = r0*(np.exp(1j*angle) + delta_0)
        k = r0*(delta_0 + 1)
        fish_boundary = circle + k**2/circle

        # Get discretized center line of foil
        foil_plus = 2*r0*(delta_0 + 1)
        foil_minus = 2*r0*(delta_0**2 + 1)/(delta_0 - 1)
        self.centerline = np.linspace(foil_minus, foil_plus, self.N)

        # Get outer edge of foil along discretized centerline
        radii = np.zeros((self.N))
        fish_path = Path(np.column_stack((fish_boundary.real, fish_boundary.imag)))
        search_y = np.linspace(0, foil_plus-foil_minus, 200)
        for ii in range(1, self.N-1): 
            x = self.centerline[ii]
            for y in search_y:
                point = (x, y)
                if not fish_path.contains_point(point):
                    break
            radii[ii] = y
        
        # Edge parameters
        self.l_edge_ref = (self.centerline[1] - self.centerline[0]) * np.ones((self.N-1))
        self.h_edge_ref = radii[1:] + radii[:-1]
        self.w_edge_ref = (self.fish_length/4) * np.ones((self.N-1))

        # Derived edge parameters
        self.a_edge_ref = self.h_edge_ref * self.w_edge_ref
        self.m_edge_ref = rho * self.a_edge_ref * self.l_edge_ref
        I_edge_ref = self.m_edge_ref * (self.w_edge_ref ** 2) / 12

        # Voronoi node parameters
        self.l_node_ref = (np.concatenate([self.l_edge_ref,[0]]) + np.concatenate([[0], self.l_edge_ref])) / 2
        self.m_node_ref = (np.concatenate([self.m_edge_ref,[0]]) + np.concatenate([[0], self.m_edge_ref])) / 2
        self.I_node_ref = (np.concatenate([I_edge_ref,[0]]) + np.concatenate([[0], I_edge_ref])) / 2

        # Scale the maximum curvature to imitate carngiform swimming
        self.recenterline = self.centerline[1:self.N-1] - self.centerline[0]
        self.centerline_scaling = self.c1 * (self.recenterline / self.fish_length) + self.c2 * (self.recenterline / self.fish_length) ** 2
        
        # Initialize fish body configuration
        self.positions = np.zeros((2,self.N))
        self.velocities = np.zeros((2,self.N))
        self.phi = np.zeros(self.N-1)
        self.kappa = np.zeros(self.N-1)

        # Define head configuration
        self.positions[:, 0] = np.array([config['head_position_x'], config['head_position_y']])
        self.velocities[:] = np.array([[config['head_velocity_x']], [config['head_velocity_y']]])
        self.phi[0] = self.heading - np.pi

        # Define rest of fish body configuration
        self.phi[1:] = self.centerline_scaling * self.wave_amplitude * (np.sin(self.wave_number * self.recenterline + self.wave_phase) + self.wave_offset)
        current_phi = self.phi[0]
        for ii in range(self.N - 1):
            self.positions[:, ii+1] = self.positions[:, ii] + self.l_edge_ref[ii] * np.array([np.cos(current_phi), np.sin(current_phi)])
            if ii < self.N - 2:
                current_phi += self.phi[ii+1]

        # Bound vortex sheet (BVS) distribution
        self.bvs_distribution = np.zeros((self.bvs_N))
        self.which_index = np.zeros((self.bvs_N))
        self.link_ratio = np.zeros((self.bvs_N))

        # BVS positions and velocities
        self.bvs_positions = np.zeros((2,self.bvs_N))
        self.bvs_velocities = np.zeros((2,self.bvs_N))
        self.bvs_gamma = np.zeros((self.bvs_N))
        self.bvs_length = np.zeros((self.bvs_N))
        self.bvs_mass = np.zeros((self.bvs_N))

        # Solve for all BVS values
        self.initialize_bvs()

    def initialize_bvs(self):

        # Define Chebyshev points
        chebyshev_index = np.linspace(self.bvs_N-1, 0, self.bvs_N)
        chebyshev_distribution = self.fish_length * (1 + np.cos(chebyshev_index * np.pi / (self.bvs_N-1))) / 2
        self.bvs_distribution = chebyshev_distribution

        # Find where each bound vortex falls relative to evenly spaced joints
        joint_distribution = np.linspace(0, self.fish_length, self.N)
        for ii in range(self.N-1):

            indices = np.where((self.bvs_distribution > joint_distribution[ii]) & 
                               (self.bvs_distribution <= joint_distribution[ii+1]))
            self.which_index[indices] = ii
            self.link_ratio[indices] = ((self.bvs_distribution[indices] - joint_distribution[ii]) / 
                                        (joint_distribution[ii+1]- joint_distribution[ii]))
            
        # Exception for final link
        self.which_index[-1] = self.N-2
        self.link_ratio[-1] = 1

        # Solve for BVS positions and velocities
        self.interpolate_bvs()

        # Compute Voronoi BVS edge lengths
        for ii in range(self.bvs_N-1):
            half_edge_length = np.linalg.norm((self.bvs_positions[:,ii+1] - self.bvs_positions[:,ii])/2)
            self.bvs_length[ii] += half_edge_length
            self.bvs_length[ii+1] += half_edge_length

        # Compute BVS masses
        link_densities = self.m_edge_ref / self.l_edge_ref
        for ii in range(self.bvs_N):

            # Exception for endpoints
            if ii == 0: 
                self.bvs_mass[0] = link_densities[0] * self.bvs_length[0]/2
                continue
            if ii == self.bvs_N - 1:
                self.bvs_mass[-1] = link_densities[-1] * self.bvs_length[-1]/2
                continue
            
            # Compute distance to nodes and midpoints
            node_idx = int(self.which_index[ii])
            distance_to_left_node = self.bvs_distribution[ii] - joint_distribution[node_idx]
            distance_to_right_node = joint_distribution[node_idx+1] - self.bvs_distribution[ii]
            distance_to_midpoint = self.bvs_length[ii]/2

            # Account for overlap on the left
            if distance_to_left_node < distance_to_midpoint:
                self.bvs_mass[ii] += link_densities[node_idx-1] * (distance_to_midpoint - distance_to_left_node)
                self.bvs_mass[ii] += link_densities[node_idx] * distance_to_left_node
            else:
                self.bvs_mass[ii] += link_densities[node_idx] * distance_to_midpoint

            # Account for overlap on the right
            if distance_to_right_node < distance_to_midpoint:
                self.bvs_mass[ii] += link_densities[node_idx+1] * (distance_to_midpoint - distance_to_right_node)
                self.bvs_mass[ii] += link_densities[node_idx] * distance_to_right_node
            else:
                self.bvs_mass[ii] += link_densities[node_idx] * distance_to_midpoint

    def interpolate_bvs(self):
        
        # Interpolate current position and velocites for BVS
        for ii in range(self.N-1):

            indices = (self.which_index == ii)
            self.bvs_positions[:,indices] = (self.positions[:,ii][:,np.newaxis] + self.link_ratio[indices][np.newaxis, :]
                                           *(self.positions[:,ii+1] - self.positions[:,ii])[:,np.newaxis])
            self.bvs_velocities[:,indices] = (self.velocities[:,ii][:,np.newaxis] + self.link_ratio[indices][np.newaxis, :]
                                            *(self.velocities[:,ii+1] - self.velocities[:,ii])[:,np.newaxis])
    
    def controller(self):

        # Compute and store new heading
        offset = 0
        tip_head = self.positions[:, 0] # tip of head
        end_head = self.positions[:, 2] # end of the head
        heading_vec = tip_head - end_head
        self.heading[self.time_step] = np.mod(np.arctan2(heading_vec[1], heading_vec[0]), 2*np.pi)
        
        # Apply control after one full cycle
        if self.time_step >= self.period_steps - 1:
            
            # Compute time-averaged heading
            average_heading = np.mean(self.heading[(self.time_step - self.period_steps + 1):self.time_step+1])
            error_heading = average_heading - self.desired_heading
            
            # Wave offset steering input
            offset = 0.2*np.tanh(2*error_heading)
            
            # Set limit on rate of change
            if offset > self.past_offset + self.max_offset_rate * self.delta_T:
                offset = self.past_offset + self.max_offset_rate * self.delta_T
            if offset < self.past_offset - self.max_offset_rate * self.delta_T:
                offset = self.past_offset - self.max_offset_rate * self.delta_T
        
        # Update past offset variable
        self.past_offset = offset

        # Return phase offset control input
        return offset

    def internal_forces(self, time, z0, u):

        ############ REDEFINE FREQUENTLY USED PARAMETERS ############

        N = self.N
        d_b = self.d_b
        d_s = self.d_s

        ############ COMPUTE CURRENT CONFIGURATION ############

        # Define positions and velocities based on state
        x = z0[0:2*N].reshape((2,N), order='F')
        v = z0[2*N:4*N].reshape((2,N), order='F')
        self.positions = x
        self.velocities = v
        self.interpolate_bvs()

        # Compute edge properties
        for ii in range(N-1):
            self.e[:,ii] = x[:,ii+1] - x[:,ii]
            self.l_edge[ii] = np.linalg.norm(self.e[:,ii])
            self.t[:,ii] = self.e[:,ii] / self.l_edge[ii]
            self.n[:,ii] = np.array([-self.t[1,ii], self.t[0,ii]])

        # Compute node properties
        self.phi[0] = np.arctan2(self.t[1,0], self.t[0,0])
        for ii in range(1, N-1):
            cross = self.t[0,ii-1] * self.t[1,ii] - self.t[1,ii-1] * self.t[0,ii]
            dot = np.dot(self.t[:,ii-1], self.t[:,ii])
            self.phi[ii] = np.arctan2(cross, dot)
        self.kappa = 2*np.tan(self.phi[1:]/2)

        # Redefine updated configurations variables
        t = self.t
        n = self.n
        l_edge = self.l_edge
        phi = self.phi
        kappa = self.kappa

        # Compute curvature and reference curvature
        angle_ref = self.centerline_scaling * self.wave_amplitude * (np.sin(self.wave_number * self.recenterline + self.wave_frequency * time + self.wave_phase) + u)
        kappa_ref = 2*np.tan(angle_ref/2)

        ############ ELASTIC FORCES ############

        # Compute stretching forces at each node
        f_stretch = self.E * self.a_edge_ref * (l_edge / self.l_edge_ref - 1)
        FS1 = np.vstack([f_stretch[:, np.newaxis] * t.T, np.array([[0, 0]])]).T
        FS2 = np.vstack([np.array([[0, 0]]), -f_stretch[:, np.newaxis] * t.T]).T
        FS = FS1 + FS2
        
        # Compute bending forces at each node
        bending_partials = (2 * self.E * self.I_node_ref[1:-1] * (kappa - kappa_ref) / 
                           (self.l_node_ref[1:-1] * (1 + np.cos(phi[1:]))))
        f_bending_1 = np.concatenate([[0], bending_partials])
        f_bending_2 = np.concatenate([-bending_partials, [0]])
        f_bending = (f_bending_1 + f_bending_2) / l_edge
        FB1 = np.vstack([f_bending[:, np.newaxis] * n.T, np.array([[0, 0]])]).T
        FB2 = np.vstack([np.array([[0, 0]]), -f_bending[:, np.newaxis] * n.T]).T
        FB = FB1 + FB2 

        # Total elastic (conservative) force
        FE = FS + FB

        ############ INTERNAL DISSIPATIVE FORCES ############

        # Curvature rate of change
        kappa_dot_scale = 2/(1 + np.cos(phi[1:N-1]))
        kappa_dot_direction = ((n[:,0:N-2] / l_edge[0:N-2]) * v[:,0:N-2] -
                             (n[:,0:N-2] / l_edge[0:N-2] + n[:,1:N-1] / l_edge[1:N-1]) * v[:,1:N-1] +
                             (n[:,1:N-1] / l_edge[1:N-1]) * v[:,2:N])
        kappa_dot = np.concatenate(([0], kappa_dot_scale * np.sum(kappa_dot_direction, axis=0), [0]))

        # Bending dissipative term
        FBD = np.zeros((2,N))
        FBD[:,0] = d_b * (-2*kappa_dot[1]/(1 + np.cos(phi[1]))) * n[:,0]/l_edge[0]      
        FBD[:,1] = (d_b * (2*kappa_dot[1]/(1 + np.cos(phi[1]))) * n[:,0]/l_edge[0] +
                    d_b * (2*kappa_dot[1]/(1 + np.cos(phi[1])) - 2*kappa_dot[2]/(1 + np.cos(phi[2]))) * n[:,1]/l_edge[1])
        for jj in range(2,N-2):
            FBD[:,jj] = (
                d_b * (-2*kappa_dot[jj-1]/(1 + np.cos(phi[jj-1])) + 2*kappa_dot[jj]/(1 + np.cos(phi[jj]))) * n[:,jj-1]/l_edge[jj-1] +
                d_b * (2*kappa_dot[jj]/(1 + np.cos(phi[jj])) - 2*kappa_dot[jj+1]/(1 + np.cos(phi[jj+1]))) * n[:,jj]/l_edge[jj]
            )       
        FBD[:,N-2] = (d_b * (-2*kappa_dot[N-3]/(1 + np.cos(phi[N-3])) + 2*kappa_dot[N-2]/(1 + np.cos(phi[N-2]))) * n[:,N-3]/l_edge[N-3] +
                      d_b * (2*kappa_dot[N-2]/(1 + np.cos(phi[N-2]))) * n[:,N-2]/l_edge[N-2])
        FBD[:,N-1] = d_b * (-2*kappa_dot[N-2]/(1 + np.cos(phi[N-2]))) * n[:,N-2]/l_edge[N-2]

        # Stretching dissipative term
        FSD = np.zeros((2,N))
        FSD[:,0] = d_s * np.outer(t[:,0],t[:,0]) @ (v[:,1] - v[:,0])
        for jj in range(1,N-1):
            FSD[:,jj] = (d_s * np.outer(t[:,jj],t[:,jj]) @ (v[:,jj+1] - v[:,jj]) - 
                         d_s * np.outer(t[:,jj-1],t[:,jj-1]) @ (v[:,jj] - v[:,jj-1]))
        FSD[:,N-1] = -d_s * np.outer(t[:,N-2],t[:,N-2]) @ (v[:,N-1] - v[:,N-2])

        # Total internal dissipative force
        FID = FBD + FSD

        ############ HYDRODYNAMIC DRAG FORCES ############

        # Drag coefficients
        CT = self.fluid_density * np.pi * self.CF * (self.h_edge_ref + self.w_edge_ref)/(4 * self.l_edge_ref)
        CN = self.fluid_density * self.CD * self.h_edge_ref * self.l_edge_ref

        # Tangent and normal components of edge velocities
        v_ext = (self.external_velocity[:,0:N-1] + self.external_velocity[:,1:N])/2
        v_edge = (v[:,0:N-1] + v[:,1:N]) / 2
        v_rel = v_edge - v_ext
        v_edge_T = np.sum(v_rel*t, axis=0)
        v_edge_N = np.sum(v_rel*n, axis=0)

        # Hydrodynamic drag forces
        FHD_edge = (-CT * (v_edge_T + v_edge_T * np.abs(v_edge_T)) * t
                    -CN * (v_edge_N + v_edge_N * np.abs(v_edge_N)) * n)
        FHD = (np.hstack((FHD_edge, np.zeros((2,1)))) + np.hstack((np.zeros((2,1)), FHD_edge))) / 2

        ############ ADDED MASS FORCES ############

        # Set up set of linear equations
        FOther = (FE + FID + FHD).reshape(-1, order='F')
        mu_normal = self.fluid_density * np.pi * self.CA * self.l_edge_ref * self.h_edge_ref**2
        added_mass_matrix = np.zeros((2*N, 2*N))

        # Top block row
        added_mass_matrix[0:2,0:2] = 0.25*mu_normal[0]*np.outer(n[:,0],n[:,0]) + self.m_node_ref[0]*np.eye(2)
        added_mass_matrix[0:2,2:4] = 0.25*mu_normal[0]*np.outer(n[:,0],n[:,0])

        # Middle block rows
        for jj in range(1,N-1):
            
            # Lower diagonal
            added_mass_matrix[2*jj:2*jj+2, 2*jj-2:2*jj] = 0.25*mu_normal[jj-1]*np.outer(n[:,jj-1],n[:,jj-1])
            
            # Diagonal
            added_mass_matrix[2*jj:2*jj+2, 2*jj:2*jj+2] = (self.m_node_ref[jj]*np.eye(2) + 
                0.25*mu_normal[jj-1]*np.outer(n[:,jj-1],n[:,jj-1]) + 0.25*mu_normal[jj]*np.outer(n[:,jj],n[:,jj]))
            
            # Upper diagonal
            added_mass_matrix[2*jj:2*jj+2, 2*jj+2:2*jj+4] = 0.25*mu_normal[jj]*np.outer(n[:,jj],n[:,jj])
        
        # Bottom block row
        added_mass_matrix[2*N-2:2*N, 2*N-4:2*N-2] = 0.25*mu_normal[N-2]*np.outer(n[:,N-2],n[:,N-2])
        added_mass_matrix[2*N-2:2*N, 2*N-2:2*N] = 0.25*mu_normal[N-2]*np.outer(n[:,N-2],n[:,N-2]) + self.m_node_ref[N-1]*np.eye(2)

        ############ VELOCITY AND ACCELERATION ############

        # Solve linear system for accelerations
        acceleration = pinv(added_mass_matrix) @ FOther

        # Output time-derivative of state
        dzdt = np.concatenate([z0[2*N:4*N], acceleration])

        return dzdt
