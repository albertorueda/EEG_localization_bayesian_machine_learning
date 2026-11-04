import glob
import matplotlib.pyplot as plt
import numpy as np

files = sorted(glob.glob("results/size_*.npz"))
data = [np.load(f) for f in files]
print(f"{len(files)} results found")

d = np.load("results/size_0042.npz")
print(d.files)
thetas = d["thetas"][100:]          # quita el warm-up

plt.figure(figsize=(5, 5))
plt.scatter(thetas[:, 0], thetas[:, 1], s=3, alpha=0.3)
plt.xlabel(r"$\theta_0$"); plt.ylabel(r"$\theta_1$")
plt.title("Posterior HMC, size=42")
plt.axis("equal")
plt.savefig("posterior_hmc_size_42.png")