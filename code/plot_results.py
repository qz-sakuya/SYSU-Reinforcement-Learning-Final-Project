import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_comparison():
    files = {
        "CPPO": "rewards_cppo.npy",
        "IPPO": "rewards_ippo.npy",
        "MAPPO (Baseline)": "rewards_mappo_baseline.npy",
        "MAPPO (Improved)": "rewards_mappo_improved.npy"
    }

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for label, filename in files.items():
        try:
            data = np.load(filename)
            window_size = 10
            if len(data) > window_size:
                data_smooth = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
            else:
                data_smooth = data

            x = np.arange(len(data_smooth)) * 10
            plt.plot(x, data_smooth, label=label, linewidth=2)
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")

    plt.title("Transport Scenario Learning Curve", fontsize=16)
    plt.xlabel("Updates", fontsize=12)
    plt.ylabel("Average Team Reward", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("final_comparison.png")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
