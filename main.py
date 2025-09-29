import numpy as np
import matplotlib.pyplot as plt # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import pickle

from fish import Fish
from environment import Environment

# Define school of fish and environment
Nemo = Fish('Nemo.json')
Dory = Fish('Dory.json')
school = [Nemo, Dory]
fish_tank = Environment(school)

# Run simulation and save outputs for later
fish_tank.run_simulation()
with open('finding_nemo.pkl', 'wb') as f:
    pickle.dump(fish_tank.output, f)
with open('finding_nemo_Gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_Gamma, f)
with open('finding_nemo_bvs.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs, f)
with open('finding_nemo_bvs_gamma.pkl', 'wb') as f:
    pickle.dump(fish_tank.output_bvs_gamma, f)

# Load simulation outputs
with open('finding_nemo.pkl', 'rb') as f:
    sim_output = pickle.load(f)
with open('finding_nemo_Gamma.pkl', 'rb') as f:
    sim_output_Gamma = pickle.load(f)
with open('finding_nemo_bvs.pkl', 'rb') as f:
    sim_output_bvs = pickle.load(f)
with open('finding_nemo_bvs_gamma.pkl', 'rb') as f:
    sim_output_bvs_gamma = pickle.load(f)

# Simulate environment
fish_tank.show_simulation(sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)
# fish_tank.save_animation("finding_nemo.mp4", sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)