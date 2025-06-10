import sys, os
from matplotlib.backends.backend_pgf import _tex_escape as mpl_common_texification
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

from utils.tikz import save2tikz
from env import VehicleTracking
from vehicle import Vehicle

env = VehicleTracking(Vehicle(), 100, 15, windy=True)
env.reset(seed=10)

plt.plot(env.wind[1:-1])
plt.xlabel("time [s]")
plt.ylabel("wind [m/s]")
save2tikz(plt.gcf())
plt.show()
