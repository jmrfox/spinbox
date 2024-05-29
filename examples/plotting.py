import matplotlib.pyplot as plt
import pickle as pkl

import pandas as pd

def print_dict(d):
    for key in d.keys():
        print(f"{key} : {d[key]}")



def plot_rbm_error_vs_dt():
    filename = "examples/outputs/experiment_1716958852.pkl"
    with open(filename, "rb") as f:
        data = pkl.load(f)


    df = pd.DataFrame(data)
    print(df)

    dt = df["dt"].unique()
    error_no_balance_no_mix = df["abs_error"].where((df["balance"]==False) & (df["mix"]==False)).dropna()
    error_yes_balance_no_mix = df["abs_error"].where((df["balance"]==True) & (df["mix"]==False)).dropna()
    error_yes_balance_yes_mix = df["abs_error"].where((df["balance"]==True) & (df["mix"]==True)).dropna()

    plt.figure(figsize=(5,3))
    plt.loglog(dt, error_no_balance_no_mix, label="RBM error", color="k", linestyle="-", marker="o")
    plt.loglog(dt, error_yes_balance_no_mix, label="balanced", color="b", linestyle="--", marker="s")
    plt.loglog(dt, error_yes_balance_yes_mix, label="balanced and mixed", color="g", linestyle=":", marker="+")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/outputs/error_plot.pdf")
    

def plot_nuclear_results():
    pass

if __name__=="__main__":
    plot_rbm_error_vs_dt()