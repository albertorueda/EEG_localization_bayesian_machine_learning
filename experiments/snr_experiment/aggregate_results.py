import glob

import matplotlib.pyplot as plt
import numpy as np

files = sorted(glob.glob("results/f1_n*_r*.npz"))
data = [np.load(f) for f in files]
print(f"{len(files)} results found")

snr_all = np.array([int(d["snr"]) for d in data])
f1_all = np.array([float(d["f1"]) for d in data])

snr_values = np.unique(snr_all)
mean = np.array([f1_all[snr_all == snr].mean() for snr in snr_values])
std = np.array([f1_all[snr_all == snr].std() for snr in snr_values])
count = np.array([(snr_all == snr).sum() for snr in snr_values])

for n, m, s, c in zip(snr_values, mean, std, count):
    print(f"snr={n:4d}  F1={m:.3f} ± {s:.3f}  (n={c})")

np.savez("f1_scores.npz", snr=snr_values, mean=mean, std=std, count=count,
         snr_all=snr_all, f1_all=f1_all)

plt.figure()
plt.plot(snr_values, mean, marker="o")
plt.fill_between(snr_values, mean - std, mean + std, alpha=0.25)
plt.title("F1 Score vs Signal-to-Noise Ratio")
plt.xlabel("Signal-to-Noise Ratio (dB)")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.grid()
plt.savefig("f1_scores_plot.png")