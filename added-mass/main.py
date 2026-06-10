import numpy as np
import matplotlib.pyplot as plt # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import pickle

from fish import Fish
from environment import Environment

# # ONE FISH SIMULATIONS

# # Define school of fish and environment
# Nemo = Fish('Nemo.json')
# school = [Nemo]
# fish_tank = Environment(school, controller=True)

# # Run simulation and save outputs for later
# fish_tank.run_simulation()
# with open('robosoft_results/one_fish_CL.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output, f)
# with open('robosoft_results/one_fish_CL_Gamma.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_Gamma, f)
# with open('robosoft_results/one_fish_CL_bvs.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_bvs, f)
# with open('robosoft_results/one_fish_CL_bvs_gamma.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_bvs_gamma, f)

# # Store control inputs
# with open('robosoft_results/one_fish_CL_control_input.pkl', 'wb') as f:
#     pickle.dump(Nemo.control_inputs, f)

# # Load simulation outputs
# with open('robosoft_results/one_fish_CL.pkl', 'rb') as f:
#     sim_output = pickle.load(f)
# with open('robosoft_results/one_fish_CL_Gamma.pkl', 'rb') as f:
#     sim_output_Gamma = pickle.load(f)
# with open('robosoft_results/one_fish_CL_bvs.pkl', 'rb') as f:
#     sim_output_bvs = pickle.load(f)
# with open('robosoft_results/one_fish_CL_bvs_gamma.pkl', 'rb') as f:
#     sim_output_bvs_gamma = pickle.load(f)

# # Simulate environment
# # fish_tank.show_simulation(sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)
# fish_tank.save_animation("one_fish_CL_arrows.mp4", sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)

# # TWO FISH

# # Define school of fish and environment
# Nemo = Fish('Nemo.json')
# Dory = Fish('Dory.json')
# school = [Nemo, Dory]
# fish_tank = Environment(school, controller=True)

# # Run simulation and save outputs for later
# fish_tank.run_simulation()
# with open('two_fish_OL.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output, f)
# with open('two_fish_OL_Gamma.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_Gamma, f)
# with open('two_fish_OL_bvs.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_bvs, f)
# with open('two_fish_OL_bvs_gamma.pkl', 'wb') as f:
#    pickle.dump(fish_tank.output_bvs_gamma, f)

# # Store control inputs
# with open('Nemo_two_fish_OL_control_input.pkl', 'wb') as f:
#     pickle.dump(Nemo.control_inputs, f)
# with open('Dory_two_fish_OL_control_input.pkl', 'wb') as f:
#     pickle.dump(Dory.control_inputs, f)

# # Load simulation outputs
# with open('robosoft_results/two_fish_CL.pkl', 'rb') as f:
#     sim_output = pickle.load(f)
# with open('robosoft_results/two_fish_CL_Gamma.pkl', 'rb') as f:
#     sim_output_Gamma = pickle.load(f)
# with open('robosoft_results/two_fish_CL_bvs.pkl', 'rb') as f:
#     sim_output_bvs = pickle.load(f)
# with open('robosoft_results/two_fish_CL_bvs_gamma.pkl', 'rb') as f:
#     sim_output_bvs_gamma = pickle.load(f)

# # Simulate environment
# fish_tank.show_simulation(sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)
# # fish_tank.save_animation("two_fish_CL_arrows.mp4", sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)

# FOUR FISH: DIAMOND

# Define school of fish and environment
OneFish = Fish('one_fish.json')
TwoFish = Fish('two_fish.json')
RedFish = Fish('red_fish.json')
BlueFish = Fish('blue_fish.json')
school = [OneFish, TwoFish, RedFish, BlueFish]
fish_tank = Environment(school, controller=False)

# # Run simulation and save outputs for later
# fish_tank.run_simulation()
# with open('dr_seuss_OL.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output, f)
# with open('dr_seuss_OL_Gamma.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_Gamma, f)
# with open('dr_seuss_OL_bvs.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_bvs, f)
# with open('dr_seuss_OL_bvs_gamma.pkl', 'wb') as f:
#     pickle.dump(fish_tank.output_bvs_gamma, f)

# # Store control inputs
# with open('diamond_one_fish_control_input.pkl', 'wb') as f:
#     pickle.dump(OneFish.control_inputs, f)
# with open('diamond_two_fish_control_input.pkl', 'wb') as f:
#     pickle.dump(TwoFish.control_inputs, f)
# with open('diamond_red_fish_control_input.pkl', 'wb') as f:
#     pickle.dump(RedFish.control_inputs, f)
# with open('diamond_blue_fish_control_input.pkl', 'wb') as f:
#     pickle.dump(BlueFish.control_inputs, f)

# Load simulation outputs
with open('robosoft_results/dr_seuss_OL.pkl', 'rb') as f:
    sim_output = pickle.load(f)
with open('robosoft_results/dr_seuss_OL_Gamma.pkl', 'rb') as f:
    sim_output_Gamma = pickle.load(f)
with open('robosoft_results/dr_seuss_OL_bvs.pkl', 'rb') as f:
    sim_output_bvs = pickle.load(f)
with open('robosoft_results/dr_seuss_OL_bvs_gamma.pkl', 'rb') as f:
    sim_output_bvs_gamma = pickle.load(f)

# Simulate environment
# fish_tank.show_simulation(sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)
fish_tank.save_animation("dr_seuss_OL.mp4", sim_output, sim_output_Gamma, sim_output_bvs, sim_output_bvs_gamma, show_bvs=False)
