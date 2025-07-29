import numpy as np
from numpy.linalg import pinv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]

from fish import Fish
from environment import Environment
import pickle

########## SINGLE FISH GENERATING WAKE ##########

# Define fish
num = 7
Nemo_position = np.vstack((np.linspace(0, 1.0, num), np.zeros((num)))) 
Nemo_velocity = np.vstack((-0.1*np.ones(num), np.zeros((num))))
Nemo = Fish(Nemo_position, Nemo_velocity)

# Reinitialize fish position to have reference curvature
angles = Nemo.centerlineScaling * Nemo.waveAmp * np.sin(Nemo.waveNum * Nemo.centerline[1:num-1]) + Nemo.waveOffset
angles = angles.flatten()
current_phi = 0.0
for ii in range(num-1):
    Nemo_position[:, ii+1] = Nemo_position[:, ii] + Nemo.lEdgeRef[ii]*np.array([np.cos(current_phi), np.sin(current_phi)])
    if ii < num-2:
        current_phi += angles[ii]

# Define the environment containing the fish
school = [Nemo]
fish_tank = Environment(school)

# Run simulation and save outputs for later
fish_tank.run_simulation()
with open('e.pkl', 'wb') as f:
    pickle.dump(fish_tank.output, f)
with open('e_Gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_Gamma, f)
with open('e_bvs.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs, f)
with open('e_bvs_gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs_gamma, f)

# Load simulation outputs
with open('e.pkl', 'rb') as f:
    sim_output = pickle.load(f)
with open('e_Gamma.pkl', 'rb') as f:
    sim_output_Gamma = pickle.load(f)
with open('e_bvs.pkl', 'rb') as f:
    sim_output_bvs = pickle.load(f)
with open('e_bvs_gamma.pkl', 'rb') as f:
    sim_output_bvs_gamma = pickle.load(f)

# simulation c: shedding and hydro forces with small initial velocity, zero out forces in the y-direction
# simulation d: no shedding hydro forces with small initial velocity, forces in all directions
# simulation e: simulation c but run until reaching steady-state velocity

# Simulate environment
fish_tank.simulation(sim_output, sim_output_Gamma)
