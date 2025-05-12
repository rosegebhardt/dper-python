import numpy as np
from numpy.linalg import pinv
from scipy.integrate import solve_ivp
# type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]

class Fish:

    def __init__(self, z0):

        # Fish geometry parameters
        self.N = 7
        self.fishLength = 0.4
        delta0 = 0.2
        r0 = self.fishLength*(1 - delta0)/4

        # Wave propogation parameters
        self.waveNum = -6/self.fishLength
        self.waveFreq = 3
        self.waveAmp = 0.4
        self.c1 = 1; self.c2 = 0.2

        # Integration settings
        self.timeStep = 0
        self.timeN = 51 #501
        self.tMax = 2 #20
        self.deltaT = self.tMax/(self.timeN-1)
        self.initState = z0

        # Store outputs
        self.output = np.zeros((self.timeN,4*self.N))
        self.output[0, :] = self.initState

        # Control parameters
        self.periodSteps = round(2*np.pi/self.waveFreq/self.deltaT)
        self.pastOffset = 0
        self.maxOffsetRate = 0.05
        self.desiredHeading = np.radians(180)  
        self.heading = np.zeros((self.timeN,1))  

        # Material parameters 
        self.E = 1.875e4
        rho = 2000

        # Fluid parameters
        self.d_b = 0.5
        self.d_s = 0.5
        self.rhoFluid = 1000
        self.CF = 1e-5 # tangent
        self.CD = 1.75 # normal
        self.CA = 10

        # Form fish polygon
        angle = np.linspace(0,2*np.pi,1000)
        circle = r0*(np.exp(1j*angle) + delta0)
        k = r0*(delta0 + 1)
        fish = circle + k**2/circle

        # Get discretized center line of foil
        foilPlus = 2*r0*(delta0 + 1)
        foilMinus = 2*r0*(delta0**2 + 1)/(delta0 - 1)
        self.centerline = np.linspace(foilMinus, foilPlus, self.N).reshape(-1, 1)

        # Get outer edge of foil along discretized centerline
        radii = np.zeros((self.N, 1))
        fish_path = Path(np.column_stack((fish.real, fish.imag)))
        search_y = np.linspace(0, foilPlus-foilMinus, 200)
        for i in range(1, self.N-1): 
            x = self.centerline[i, 0]
            for y in search_y:
                point = (x, y)
                if not fish_path.contains_point(point):
                    break
            radii[i] = y

        # Edge parameters
        self.lEdgeRef = (self.centerline[1] - self.centerline[0]) * np.ones((self.N-1, 1))
        self.hEdgeRef = radii[1:] + radii[:-1]
        self.wEdgeRef = (self.fishLength/4) * np.ones((self.N-1, 1))

        # Derived edge parameters
        self.AEdgeRef = self.hEdgeRef * self.wEdgeRef
        MEdgeRef = rho * self.AEdgeRef * self.lEdgeRef
        IEdgeRef = MEdgeRef * (self.wEdgeRef ** 2) / 12

        # Voronoi node parameters
        self.lNodeRef = np.vstack([self.lEdgeRef[0], self.lEdgeRef[:-1] + self.lEdgeRef[1:], self.lEdgeRef[-1]]) / 2
        self.MNodeRef = np.vstack([MEdgeRef[0], MEdgeRef[:-1] + MEdgeRef[1:], MEdgeRef[-1]]) / 2
        self.INodeRef = np.vstack([IEdgeRef[0], IEdgeRef[:-1] + IEdgeRef[1:], IEdgeRef[-1]]) / 2

    def fluidFlow(self, time, x, t, n):

        ############ UNIFORM FLOW ############
    
        u_inf = 0.0
        aoa = np.deg2rad(90)
        def v_x(x_val,y_val): return u_inf*np.cos(aoa)*np.ones_like(x_val)
        def v_y(x_val,y_val): return u_inf*np.sin(aoa)*np.ones_like(y_val)

        ############ FREE VORTEX ############
        
        # Gamma = 0.1
        # z_vortex = -self.fishLength
        # def vel(x_val,y_val): return 1j * Gamma / (2*np.pi*np.conj((x_val + 1j*y_val) - z_vortex))
        # def v_x(x_val,y_val): return np.real(vel(x_val,y_val))
        # def v_y(x_val,y_val): return np.imag(vel(x_val,y_val))
        
        ############ LINE INTEGRAL ############
        
        # Caluclate rectangular segment locations
        midpoints = (x[:,0:self.N-1] + x[:,1:self.N])/2
        southwestPoints = midpoints - 0.5*t*self.lEdgeRef.T - 0.5*n*self.hEdgeRef.T
        northwestPoints = midpoints - 0.5*t*self.lEdgeRef.T + 0.5*n*self.hEdgeRef.T
        northeastPoints = midpoints + 0.5*t*self.lEdgeRef.T + 0.5*n*self.hEdgeRef.T
        southeastPoints = midpoints + 0.5*t*self.lEdgeRef.T - 0.5*n*self.hEdgeRef.T

        # Store external flow averaged around segment
        v_ext = np.zeros((2,self.N-1))
        nLeftRight = 20
        nTopDown = 10

        # Iterate over segments
        for ii in range(self.N-1):

            # Discretize edges
            leftEdge = np.array([
                np.linspace(southwestPoints[0,ii], northwestPoints[0,ii], nLeftRight),
                np.linspace(southwestPoints[1,ii], northwestPoints[1,ii], nLeftRight)
            ])
            topEdge = np.array([
                np.linspace(northwestPoints[0,ii], northeastPoints[0,ii], nTopDown),
                np.linspace(northwestPoints[1,ii], northeastPoints[1,ii], nTopDown)
            ])
            rightEdge = np.array([
                np.linspace(northeastPoints[0,ii], southeastPoints[0,ii], nLeftRight),
                np.linspace(northeastPoints[1,ii], southeastPoints[1,ii], nLeftRight)
            ])
            bottomEdge = np.array([
                np.linspace(southeastPoints[0,ii], southwestPoints[0,ii], nTopDown),
                np.linspace(southeastPoints[1,ii], southwestPoints[1,ii], nTopDown)
            ])
            
            # Average velocity across edges
            v_ext[0,ii] = np.mean(np.hstack([
                v_x(leftEdge[0,:], leftEdge[1,:]),
                v_x(topEdge[0,:], topEdge[1,:]),
                v_x(rightEdge[0,:], rightEdge[1,:]),
                v_x(bottomEdge[0,:], bottomEdge[1,:])
            ]))

            v_ext[1,ii] = np.mean(np.hstack([
                v_y(leftEdge[0,:], leftEdge[1,:]),
                v_y(topEdge[0,:], topEdge[1,:]),
                v_y(rightEdge[0,:], rightEdge[1,:]),
                v_y(bottomEdge[0,:], bottomEdge[1,:])
            ]))

        return v_ext

    def forcedDPER(self, time, z, u):

        ############ REDEFINE FREQUENTLY USED PARAMETERS ############

        N = self.N
        centerline = self.centerline
        d_b = self.d_b
        d_s = self.d_s

        ############ COMPUTE CURRENT CONFIGURATION ############

        # Define positions, velocities, and accelerations based on state
        positions = z[:2*N]
        x = positions.reshape((2,N), order='F')
        velocities = z[2*N:4*N]
        v = velocities.reshape((2,N), order='F')
        
        # Preallocate edge properties
        e = np.zeros((2,N-1))
        t = np.zeros((2,N-1))
        n = np.zeros((2,N-1))
        lEdge = np.zeros(N-1)

        # Compute edge properties
        for ii in range(N-1):
            e[:,ii] = x[:,ii+1] - x[:,ii]
            lEdge[ii] = np.linalg.norm(e[:,ii])
            t[:,ii] = e[:,ii] / lEdge[ii]
            n[:,ii] = np.array([-t[1,ii], t[0,ii]])

        # Compute node properties
        phi = np.zeros(N-1)
        phi[0] = np.arctan2(t[1,0], t[0,0])
        for ii in range(1, N-1):
            cross = t[0,ii-1] * t[1,ii] - t[1,ii-1] * t[0,ii]
            dot = np.dot(t[:,ii-1], t[:,ii])
            phi[ii] = np.arctan2(cross, dot)
        kappa = 2*np.tan(phi[1:]/2)

        # Compute curvature and reference curvature
        recenterline = centerline[1:N-1] - centerline[0]
        centerlineScaling = self.c1 * recenterline + self.c2 * recenterline**2
        angleRef = centerlineScaling * self.waveAmp * np.sin(self.waveNum * centerline[1:N-1] + self.waveFreq * time) + u
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

        # TODO: should the second term of kappaDotDirection be subtracted?
        # Curvature rate of change
        kappaDotScale = 2/(1 + np.cos(phi[1:N-1]))
        kappaDotDirection = ((n[:,0:N-2]/lEdge[0:N-2]) * v[:,0:N-2] +
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

        ############ HYDRODYNAMIC DRAG FORCES ############
        
        # Compute average external flow on edges
        vExt = self.fluidFlow(time, x, t, n)

        # Drag coefficients
        CT = self.rhoFluid*np.pi*self.CF*(self.hEdgeRef + self.wEdgeRef)/(4*self.lEdgeRef)
        CN = self.rhoFluid*self.CD*self.hEdgeRef*self.lEdgeRef

        # Tangent and normal components of edge velocities
        vEdge = (v[:,0:N-1] + v[:,1:N]) / 2
        vEdgeRel = vEdge - vExt
        vEdgeT = np.sum(vEdgeRel*t, axis=0)
        vEdgeN = np.sum(vEdgeRel*n, axis=0)

        # Hydrodynamic drag forces
        FHDEdge = (-CT.flatten() * (vEdgeT + vEdgeT * np.abs(vEdgeT)) * t
                   -CN.flatten() * (vEdgeN + vEdgeN * np.abs(vEdgeN)) * n)
        FHD = (np.hstack((FHDEdge, np.zeros((2,1)))) + np.hstack((np.zeros((2,1)), FHDEdge))) / 2

        ############ ADDED MASS FORCES ############

        # Set up set of linear equations
        FOther = (FE + FID + FHD).reshape(-1, order='F')
        muNormal = self.rhoFluid * np.pi * self.CA * self.lEdgeRef * self.hEdgeRef**2
        addedMassMatrix = np.zeros((2*N, 2*N))

        # Top block row
        addedMassMatrix[0:2,0:2] = 0.25*muNormal[0]*np.outer(n[:,0],n[:,0]) + self.MNodeRef[0]*np.eye(2)
        addedMassMatrix[0:2,2:4] = 0.25*muNormal[0]*np.outer(n[:,0],n[:,0])

        # Middle block rows
        for jj in range(1,N-1):
            
            # Lower diagonal
            addedMassMatrix[2*jj:2*jj+2, 2*jj-2:2*jj] = 0.25*muNormal[jj-1]*np.outer(n[:,jj-1],n[:,jj-1])
            
            # Diagonal
            addedMassMatrix[2*jj:2*jj+2, 2*jj:2*jj+2] = (self.MNodeRef[jj]*np.eye(2) + 
                0.25*muNormal[jj-1]*np.outer(n[:,jj-1],n[:,jj-1]) + 0.25*muNormal[jj]*np.outer(n[:,jj],n[:,jj]))
            
            # Upper diagonal
            addedMassMatrix[2*jj:2*jj+2, 2*jj+2:2*jj+4] = 0.25*muNormal[jj]*np.outer(n[:,jj],n[:,jj])
        
        # Bottom block row
        addedMassMatrix[2*N-2:2*N, 2*N-4:2*N-2] = 0.25*muNormal[N-2]*np.outer(n[:,N-2],n[:,N-2])
        addedMassMatrix[2*N-2:2*N, 2*N-2:2*N] = 0.25*muNormal[N-2]*np.outer(n[:,N-2],n[:,N-2]) + self.MNodeRef[N-1]*np.eye(2)

        ############ VELOCITY AND ACCELERATION ############

        # Solve linear system for accelerations
        acceleration = pinv(addedMassMatrix) @ FOther

        # Output time-derivative of state
        dzdt = np.concatenate([z[2*N:4*N], acceleration])

        return dzdt
     
    def integration_step(self):
        
        # Compute and store new heading
        offset = 0
        tipPos = self.output[self.timeStep,0:2] # tip of head
        endPos = self.output[self.timeStep,4:6] # end of the head
        headingVec = tipPos - endPos
        self.heading[self.timeStep] = np.mod(np.arctan2(headingVec[1], headingVec[0]), 2*np.pi) # [0,2*pi)
        
        # Apply control after one full cycle
        if self.timeStep >= self.periodSteps - 1:
            
            # Compute time-averaged heading
            averageHeading = np.mean(self.heading[(self.timeStep - self.periodSteps + 1):self.timeStep+1])
            errorHeading = averageHeading - self.desiredHeading
            
            # Wave offset steering input
            offset = 0.1*np.tanh(2*errorHeading)
            
            # Set limit on rate of change
            if offset > self.pastOffset + self.maxOffsetRate*self.deltaT:
                offset = self.pastOffset + self.maxOffsetRate*self.deltaT
            if offset < self.pastOffset - self.maxOffsetRate*self.deltaT:
                offset = self.pastOffset - self.maxOffsetRate*self.deltaT
            
        # Integrate forward one time step
        def steerDPER(time,z): return self.forcedDPER(time,z,offset)
        tSpan = [self.deltaT*self.timeStep, self.deltaT*(self.timeStep + 1)]
        step = solve_ivp(steerDPER, tSpan, self.initState, method='RK45', rtol=1e-6, atol=1e-6)
        
        # Update initial conditions
        self.initState = step.y[:, -1]
        self.pastOffset = offset
        self.timeStep += 1
        if self.timeStep != self.timeN:
            self.output[self.timeStep, :] = (self.initState).T

    def integration_full(self):

        for _ in range(self.timeN-1):
            self.integration_step()
            print(self.timeStep)



############ DEBUGGING ############

# Define initial conditions
posStack = np.vstack((np.linspace(0, 0.4, 7), np.zeros((1,7))))
posLine = posStack.reshape(1,14, order='F')
z0 = (np.concatenate((posLine, np.zeros((1,14))), axis=1)).flatten()

# Write and print fish
fish = Fish(z0)
fish.integration_full()
print(fish.output)
# print(fish.output[fish.timeStep,:])