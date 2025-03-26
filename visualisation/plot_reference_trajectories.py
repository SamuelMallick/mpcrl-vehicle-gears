import sys, os
from matplotlib.backends.backend_pgf import _tex_escape as mpl_common_texification
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

# from utils.tikz import save2tikz
from env import VehicleTracking
from vehicle import Vehicle

env = VehicleTracking(Vehicle(), 100, 15, "type_2")
env.reset(seed=10)
num_trajectories = 25
representative_trajectories = [8, 12, 13, 15, 17]
trajectories = [
    env.next_x_ref(trajectory_type="type_2") for _ in range(num_trajectories)
]

fig, ax = plt.subplots(2, 1, sharex=True)
for trajectory in trajectories:
    ax[0].plot(trajectory[:, 0], alpha=0.5, color="grey", linestyle="-")
    ax[1].plot(trajectory[:, 1], alpha=0.5, color="grey", linestyle="-")
for i in representative_trajectories:
    ax[0].plot(trajectories[i][:, 0], label=f"Trajectory {i}")
    ax[1].plot(trajectories[i][:, 1], label=f"Trajectory {i}")
ax[0].set_ylabel("p [m]")
ax[1].set_ylabel("v [m/s]")
# save2tikz(plt.gcf())
plt.show()
