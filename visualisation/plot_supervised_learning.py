import pickle
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.getcwd())
from utils.tikz import save2tikz

# with open("results/sl_data/explicit/ep_500_loss_history_5000.pkl", "rb") as f:
#     loss_history_1 = pickle.load(f)
with open("results/sl_data/explicit/9_ep_500_loss_history_5000.pkl", "rb") as f:
    loss_history_2 = pickle.load(f)

# plt.plot(loss_history_1)
plt.plot(loss_history_2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss history")
plt.yscale("log")
save2tikz(plt.gcf())
plt.show()
