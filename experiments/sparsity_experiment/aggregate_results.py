import glob

import matplotlib.pyplot as plt
import numpy as np

files = sorted(glob.glob("results/f1_n*_r*.npz"))
data = [np.load(f) for f in files]
print(f"{len(files)} results found")

n_active_all = np.array([int(d["n_active"]) for d in data])
f1_all = np.array([float(d["f1"]) for d in data])

n_active = np.unique(n_active_all)
mean = np.array([f1_all[n_active_all == n].mean() for n in n_active])
std = np.array([f1_all[n_active_all == n].std() for n in n_active])
count = np.array([(n_active_all == n).sum() for n in n_active])

for n, m, s, c in zip(n_active, mean, std, count):
    print(f"n_active={n:4d}  F1={m:.3f} ± {s:.3f}  (n={c})")

np.savez("f1_scores.npz", n_active=n_active, mean=mean, std=std, count=count,
         n_active_all=n_active_all, f1_all=f1_all)

plt.figure()
plt.plot(n_active, mean, marker="o")
plt.fill_between(n_active, mean - std, mean + std, alpha=0.25)
plt.title("F1 Score vs Number of Active Dipoles")
plt.xlabel("Number of Active Dipoles")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.grid()
plt.savefig("f1_scores_plot.png")