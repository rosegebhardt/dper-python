import numpy as np
from numpy.linalg import pinv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]

class Fish:

    def __init__(self, x0, v0):

        # Fish geometry parameters
        self.N = 7
        self.fishLength = 1.0
        self.bvs_N = 21
        delta0 = 0.2
        r0 = self.fishLength*(1 - delta0)/4

        # Fish configurations
        self.positions = x0
        self.velocities = v0

        # Edge properties
        self.e = np.zeros((2,self.N-1))
        self.t = np.zeros((2,self.N-1))
        self.n = np.zeros((2,self.N-1))
        self.lEdge = np.zeros(self.N-1)

        # Node properties
        self.phi = np.zeros(self.N-1)
        self.kappa = np.zeros(self.N-1)

        # Wave propogation parameters
        self.waveNum = -6/self.fishLength
        self.waveFreq = 2*np.pi
        self.waveAmp = 0.4
        self.waveOffset = 0
        self.c1 = 1; self.c2 = 0.2

        # Material parameters 
        self.E = 1e6
        rho = 2000
        self.d_b = 0.5
        self.d_s = 0.5

        # Form fish polygon
        angle = np.linspace(0,2*np.pi,1000)
        circle = r0*(np.exp(1j*angle) + delta0)
        k = r0*(delta0 + 1)
        fish = circle + k**2/circle

        # Get discretized center line of foil
        foil_plus = 2*r0*(delta0 + 1)
        foil_minus = 2*r0*(delta0**2 + 1)/(delta0 - 1)
        self.centerline = np.linspace(foil_minus, foil_plus, self.N).reshape(-1, 1)

        # Get outer edge of foil along discretized centerline
        radii = np.zeros((self.N, 1))
        fish_path = Path(np.column_stack((fish.real, fish.imag)))
        search_y = np.linspace(0, foil_plus-foil_minus, 200)
        for i in range(1, self.N-1): 
            x = self.centerline[i, 0]
            for y in search_y:
                point = (x, y)
                if not fish_path.contains_point(point):
                    break
            radii[i] = y

        # Scale the maximum curvature so head is steady
        recenterline = self.centerline[1:self.N-1] - self.centerline[0]
        self.centerlineScaling = self.c1 * (recenterline/self.fishLength) + self.c2 * (recenterline/self.fishLength)**2
        # self.centerlineScaling = (recenterline/self.fishLength)**2

        # Edge parameters
        self.lEdgeRef = (self.centerline[1] - self.centerline[0]) * np.ones((self.N-1, 1))
        self.hEdgeRef = radii[1:] + radii[:-1]
        self.wEdgeRef = (self.fishLength/4) * np.ones((self.N-1, 1))

        # Derived edge parameters
        self.AEdgeRef = self.hEdgeRef * self.wEdgeRef
        self.MEdgeRef = rho * self.AEdgeRef * self.lEdgeRef
        IEdgeRef = self.MEdgeRef * (self.wEdgeRef ** 2) / 12

        # Voronoi node parameters
        self.lNodeRef = np.vstack([self.lEdgeRef[0], self.lEdgeRef[:-1] + self.lEdgeRef[1:], self.lEdgeRef[-1]]) / 2
        self.MNodeRef = np.vstack([self.MEdgeRef[0], self.MEdgeRef[:-1] + self.MEdgeRef[1:], self.MEdgeRef[-1]]) / 2
        self.INodeRef = np.vstack([IEdgeRef[0], IEdgeRef[:-1] + IEdgeRef[1:], IEdgeRef[-1]]) / 2

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
        chebyshev_distribution = self.fishLength*(1 + np.cos(chebyshev_index * np.pi / (self.bvs_N-1))) / 2
        self.bvs_distribution = chebyshev_distribution

        # Find where each bound vortex falls relative to evenly spaced joints
        joint_distribution = np.linspace(0,self.fishLength,self.N)
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
        link_densities = self.MEdgeRef.flatten() / self.lEdgeRef.flatten()
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

    def internal_forces(self, time, z0):

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
            self.lEdge[ii] = np.linalg.norm(self.e[:,ii])
            self.t[:,ii] = self.e[:,ii] / self.lEdge[ii]
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
        lEdge = self.lEdge
        phi = self.phi
        kappa = self.kappa

        # Compute curvature and reference curvature
        angleRef = self.centerlineScaling * self.waveAmp * np.sin(self.waveNum * self.centerline[1:N-1] + self.waveFreq * time) + self.waveOffset
        kappaRef = 2*np.tan(angleRef/2)

        ############ ELASTIC FORCES ############

        # Compute stretching forces at each node
        fStretch = self.E * self.AEdgeRef.flatten() * (lEdge / self.lEdgeRef.flatten() - 1)
        FS1 = np.vstack([fStretch[:, np.newaxis] * t.T, np.array([[0, 0]])]).T
        FS2 = np.vstack([np.array([[0, 0]]), -fStretch[:, np.newaxis] * t.T]).T
        FS = FS1 + FS2
        
        # Compute bending forces at each node
        bendingPartials = (2*self.E*self.INodeRef[1:-1].flatten()*(kappa - kappaRef.flatten()) / 
                           (self.lNodeRef[1:-1].flatten() * (1 + np.cos(phi[1:]))))
        fBending1 = np.concatenate([[0], bendingPartials])
        fBending2 = np.concatenate([-bendingPartials, [0]])
        fBending = (fBending1 + fBending2) / lEdge
        FB1 = np.vstack([fBending[:, np.newaxis] * n.T, np.array([[0, 0]])]).T
        FB2 = np.vstack([np.array([[0, 0]]), -fBending[:, np.newaxis] * n.T]).T
        FB = FB1 + FB2 

        # Total elastic (conservative) force
        FE = FS + FB

        ############ INTERNAL DISSIPATIVE FORCES ############

        # Curvature rate of change
        kappaDotScale = 2/(1 + np.cos(phi[1:N-1]))
        kappaDotDirection = ((n[:,0:N-2]/lEdge[0:N-2]) * v[:,0:N-2] -
                             (n[:,0:N-2]/lEdge[0:N-2] + n[:,1:N-1]/lEdge[1:N-1]) * v[:,1:N-1] +
                             (n[:,1:N-1]/lEdge[1:N-1]) * v[:,2:N])
        kappaDot = np.concatenate(([0], kappaDotScale*np.sum(kappaDotDirection, axis=0), [0]))

        # Bending dissipative term
        FBD = np.zeros((2,N))
        FBD[:,0] = d_b * (-2*kappaDot[1]/(1 + np.cos(phi[1]))) * n[:,0]/lEdge[0]      
        FBD[:,1] = (d_b * (2*kappaDot[1]/(1 + np.cos(phi[1]))) * n[:,0]/lEdge[0] +
                    d_b * (2*kappaDot[1]/(1 + np.cos(phi[1])) - 2*kappaDot[2]/(1 + np.cos(phi[2]))) * n[:,1]/lEdge[1])
        for jj in range(2,N-2):
            FBD[:,jj] = (
                d_b * (-2*kappaDot[jj-1]/(1 + np.cos(phi[jj-1])) + 2*kappaDot[jj]/(1 + np.cos(phi[jj]))) * n[:,jj-1]/lEdge[jj-1] +
                d_b * (2*kappaDot[jj]/(1 + np.cos(phi[jj])) - 2*kappaDot[jj+1]/(1 + np.cos(phi[jj+1]))) * n[:,jj]/lEdge[jj]
            )       
        FBD[:,N-2] = (d_b * (-2*kappaDot[N-3]/(1 + np.cos(phi[N-3])) + 2*kappaDot[N-2]/(1 + np.cos(phi[N-2]))) * n[:,N-3]/lEdge[N-3] +
                      d_b * (2*kappaDot[N-2]/(1 + np.cos(phi[N-2]))) * n[:,N-2]/lEdge[N-2])
        FBD[:,N-1] = d_b * (-2*kappaDot[N-2]/(1 + np.cos(phi[N-2]))) * n[:,N-2]/lEdge[N-2]

        # Stretching dissipative term
        FSD = np.zeros((2,N))
        FSD[:,0] = d_s * np.outer(t[:,0],t[:,0]) @ (v[:,1] - v[:,0])
        for jj in range(1,N-1):
            FSD[:,jj] = (d_s * np.outer(t[:,jj],t[:,jj]) @ (v[:,jj+1] - v[:,jj]) - 
                         d_s * np.outer(t[:,jj-1],t[:,jj-1]) @ (v[:,jj] - v[:,jj-1]))
        FSD[:,N-1] = -d_s * np.outer(t[:,N-2],t[:,N-2]) @ (v[:,N-1] - v[:,N-2])

        # Total internal dissipative force
        FID = FBD + FSD

        ############ FORCES AND ACCELERATION ############

        FI = FE + FID
        return FI

    def hydrodynamic_forces(self, mesh_hydrodynamics):

        """
        Compute hydrodynamic forces at each node by distributing segment-level 
        forces from the mesh to connected nodes based on interpolation weights.

        Parameters
        ----------
        mesh_hydrodynamics : ndarray of shape (2, N_bvs)
            Hydrodynamic forces computed at each mesh segment or link. The first
            row corresponds to x-components, the second to y-components.

        Returns
        -------
        FH : ndarray of shape (2, N)
            Interpolated hydrodynamic forces at each of the N nodes.
        """
        
        # Initialize hydrodynamic force array
        FH = np.zeros((2,self.N))

        # Interpolate current position and velocites for BVS
        for ii in range(self.N-1):

            # Select mesh points within link
            indices = (self.which_index == ii)

            # Add hydrodynamic contribution to left node
            left_weights = (1 - self.link_ratio[indices])[np.newaxis, :]
            FH[:, ii] += np.sum(left_weights * mesh_hydrodynamics[:, indices], axis=1)

            # Add hydrodynamic contribution to right node
            right_weights = self.link_ratio[indices][np.newaxis, :]
            FH[:, ii + 1] += np.sum(right_weights * mesh_hydrodynamics[:, indices], axis=1)

        return FH
    
    def state_derivative(self, time, z0, mesh_hydrodynamics):

        node_forces = self.internal_forces(time, z0) + self.hydrodynamic_forces(mesh_hydrodynamics)
        node_accelerations = (node_forces / self.MNodeRef.flatten())

        dzdt = np.concatenate([self.velocities.reshape(2*self.N,order='F'), node_accelerations.reshape(2*self.N,order='F')])
        # print(time)
        return dzdt



