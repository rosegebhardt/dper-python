import numpy as np
from numpy.linalg import pinv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import pickle

from fish import Fish
from environment import Environment

########## SINGLE FISH GENERATING WAKE ##########

# Define fish
num = 7
Nemo_position = np.vstack((np.linspace(0, 0.5, num), np.zeros((num)))) 
Nemo_velocity = np.vstack((-0.75*np.ones(num), np.zeros((num))))
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

'''
# Run simulation and save outputs for later
fish_tank.run_simulation()
with open('one_fish_output.pkl', 'wb') as f:
    pickle.dump(fish_tank.output, f)
with open('one_fish_output_Gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_Gamma, f)
with open('one_fish_output_bvs.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs, f)
with open('one_fish_output_bvs_gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs_gamma, f)
'''

# Load simulation outputs
with open('one_fish_output.pkl', 'rb') as f:
    sim_output = pickle.load(f)
with open('one_fish_output_Gamma.pkl', 'rb') as f:
    sim_output_Gamma = pickle.load(f)
with open('one_fish_output_bvs.pkl', 'rb') as f:
    sim_output_bvs = pickle.load(f)
with open('one_fish_output_bvs_gamma.pkl', 'rb') as f:
    sim_output_bvs_gamma = pickle.load(f)

# Simulate environment
fish_tank.simulation(sim_output, sim_output_Gamma)

'''
########## TWO FISH INTERACTING INLINE ##########

# Define fishy number 1
one_fish_position = np.vstack((np.linspace(0, 0.5, 7), np.zeros((7)))) 
one_fish_velocity = np.vstack((-0.75*np.ones(7), np.zeros((7))))
one_fish = Fish(one_fish_position, one_fish_velocity)

# Define fishy number 2
two_fish_position = np.vstack((np.linspace(1.0, 1.5, 7), np.zeros((7)))) 
two_fish_velocity = np.vstack((-0.75*np.ones(7), np.zeros((7))))
two_fish = Fish(two_fish_position, two_fish_velocity)

# Reinitialize fish position to have reference curvature
one_fish_angles = one_fish.centerlineScaling * one_fish.waveAmp * np.sin(one_fish.waveNum * one_fish.centerline[1:6]) + one_fish.waveOffset
one_fish_angles = one_fish_angles.flatten()
one_fish_current_phi = 0.0
two_fish_angles = two_fish.centerlineScaling * two_fish.waveAmp * np.sin(two_fish.waveNum * two_fish.centerline[1:6]) + two_fish.waveOffset
two_fish_angles = two_fish_angles.flatten()
two_fish_current_phi = 0.0

# Reinitialize fish position to have reference curvature
for ii in range(6):
    one_fish_position[:, ii+1] = one_fish_position[:, ii] + one_fish.lEdgeRef[ii]*np.array([np.cos(one_fish_current_phi), np.sin(one_fish_current_phi)])
    two_fish_position[:, ii+1] = two_fish_position[:, ii] + two_fish.lEdgeRef[ii]*np.array([np.cos(two_fish_current_phi), np.sin(two_fish_current_phi)])
    if ii < 5:
        one_fish_current_phi += one_fish_angles[ii]
        two_fish_current_phi += two_fish_angles[ii]

# Define the fishy school in the water
fishy_school = [one_fish,two_fish]
fish_tank = Environment(fishy_school)

# Run simulation and save outputs for later
fish_tank.run_simulation()
with open('two_fish_output.pkl', 'wb') as f:
    pickle.dump(fish_tank.output, f)
with open('two_fish_output_Gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_Gamma, f)
with open('two_fish_output_bvs.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs, f)
with open('two_fish_output_bvs_gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs_gamma, f)

# Load simulation outputs
with open('two_fish_output.pkl', 'rb') as f:
    sim_output = pickle.load(f)
with open('two_fish_output_Gamma.pkl', 'rb') as f:
    sim_output_Gamma = pickle.load(f)
with open('two_fish_output_bvs.pkl', 'rb') as f:
    sim_output_bvs = pickle.load(f)
with open('two_fish_output_bvs_gamma.pkl', 'rb') as f:
    sim_output_bvs_gamma = pickle.load(f)

# Load simulation outputs and simulate environment
with open('two_fish_output.pkl', 'rb') as f:
    sim_output = pickle.load(f)
with open('two_fish_output_Gamma.pkl', 'rb') as f:
    sim_output_Gamma = pickle.load(f)
fish_tank.simulation(sim_output, sim_output_Gamma)
'''