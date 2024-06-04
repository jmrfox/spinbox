import matplotlib.pyplot as plt
import pickle as pkl

import pandas as pd

def print_dict(d):
    for key in d.keys():
        print(f"{key} : {d[key]}")



def plot_rbm_error_vs_dt():
    filename = "examples/outputs/experiment_1716962309.pkl"
    with open(filename, "rb") as f:
        data = pkl.load(f)


    df = pd.DataFrame(data)
    print(df)

    dt = df["dt"].unique()
    hs_no_balance = df["abs_error"].where((df["method"]=="hs") & (df["balance"]==False) & (df["mix"]==False)).dropna()
    hs_yes_balance = df["abs_error"].where((df["method"]=="hs") & (df["balance"]==True) & (df["mix"]==False)).dropna()
    rbm_no_balance = df["abs_error"].where((df["method"]=="rbm") & (df["balance"]==False) & (df["mix"]==False)).dropna()
    rbm_yes_balance = df["abs_error"].where((df["method"]=="rbm") & (df["balance"]==True) & (df["mix"]==False)).dropna()
    
    plt.figure(figsize=(5,3))
    plt.loglog(dt, hs_no_balance, label="HS", color="k", linestyle=":", marker="o")
    plt.loglog(dt, hs_yes_balance, label="HS balanced", color="k", linestyle="-", marker="d")
    plt.loglog(dt, rbm_no_balance, label="RBM", color="b", linestyle=":", marker="o")
    plt.loglog(dt, rbm_yes_balance, label="RBM balanced", color="b", linestyle="-", marker="d")
    plt.xlabel(r"$\delta\tau$")
    plt.ylabel("Relative error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/outputs/error_plot.pdf")
    

def plot_nuclear_results():
    pass

if __name__=="__main__":
    plot_rbm_error_vs_dt()