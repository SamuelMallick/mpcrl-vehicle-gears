import os
import sys

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pgf import _tex_escape as mpl_common_texification

sys.path.append(os.getcwd())

from env import VehicleTracking
from utils.tikz import save2tikz
from vehicle import Vehicle

env = VehicleTracking(Vehicle(), 100, 15, windy=True)
env.reset(seed=10)

plt.plot(env.wind[1:-1])
plt.xlabel("time [s]")
plt.ylabel("wind [m/s]")
save2tikz(plt.gcf())
plt.show()
