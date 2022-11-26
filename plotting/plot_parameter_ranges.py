import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def plot_parameters(data_file, folds, data_id):
    df = pd.read_csv(data_file, sep=";", header=0)
    names = sorted(set(df['ID']))

    weight = []
    weight_means = []
    weight_stds = []
    tc = []
    tc_means = []
    tc_stds = []
    ps = []
    ps_means = []
    ps_stds = []
    cp = []
    cp_means = []
    cp_stds = []
    mp = []
    mp_means = []
    mp_stds = []
    ts = []
    ts_means = []
    ts_stds = []

    for i in range(0, len(names)):
        weight.append(df['WEIGHT'][i * folds:i * folds + folds].values)
        weight_means.append(np.mean(df['WEIGHT'][i * folds:i * folds + folds].values))
        weight_stds.append(np.std(df['WEIGHT'][i * folds:i * folds + folds].values, ddof=1))
        tc.append(df['TC'][i * folds:i * folds + folds].values)
        tc_means.append(np.mean(df['TC'][i * folds:i * folds + folds].values))
        tc_stds.append(np.std(df['TC'][i * folds:i * folds + folds].values, ddof=1))
        ps.append(df['PS'][i * folds:i * folds + folds].values)
        ps_means.append(np.mean(df['PS'][i * folds:i * folds + folds].values))
        ps_stds.append(np.std(df['PS'][i * folds:i * folds + folds].values, ddof=1))
        cp.append(df['CP'][i * folds:i * folds + folds].values)
        cp_means.append(np.mean(df['CP'][i * folds:i * folds + folds].values))
        cp_stds.append(np.std(df['CP'][i * folds:i * folds + folds].values, ddof=1))
        mp.append(df['MP'][i * folds:i * folds + folds].values)
        mp_means.append(np.mean(df['MP'][i * folds:i * folds + folds].values))
        mp_stds.append(np.std(df['MP'][i * folds:i * folds + folds].values, ddof=1))
        ts.append(df['TS'][i * folds:i * folds + folds].values)
        ts_means.append(np.mean(df['TS'][i * folds:i * folds + folds].values))
        ts_stds.append(np.std(df['TS'][i * folds:i * folds + folds].values, ddof=1))

    # weights
    plt.errorbar(names, weight_means, yerr=weight_stds, label='Simulated datasets', linestyle="None", marker='x',
                 markersize=10.0, color='darkcyan')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("average weight")
    plt.savefig(data_id + '_weights.png', bbox_inches="tight")

    plt.boxplot(weight, labels=names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("weight")
    plt.savefig(data_id + '_weights_bp.png', bbox_inches="tight")

    # tc
    plt.errorbar(names, tc_means, yerr=tc_stds, label='Simulated datasets', linestyle="None", marker='x',
                 markersize=10.0, color='forestgreen')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("average termination criterion")
    plt.savefig(data_id + '_tc.png', bbox_inches="tight")

    plt.boxplot(tc, labels=names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("termination criterion")
    plt.savefig(data_id + '_tc_bp.png', bbox_inches="tight")

    # ps
    plt.errorbar(names, ps_means, yerr=ps_stds, label='Simulated datasets', linestyle="None", marker='x',
                 markersize=10.0, color='darkorange')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("average population size")
    plt.savefig(data_id + '_ps.png', bbox_inches="tight")

    plt.boxplot(ps, labels=names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("population size")
    plt.savefig(data_id + '_ps_bp.png', bbox_inches="tight")

    # cp
    plt.errorbar(names, cp_means, yerr=cp_stds, label='Simulated datasets', linestyle="None", marker='x',
                 markersize=10.0, color='gold')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("average crossover probability")
    plt.savefig(data_id + '_cp.png', bbox_inches="tight")

    plt.boxplot(cp, labels=names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("crossover probability")
    plt.savefig(data_id + '_cp_bp.png', bbox_inches="tight")

    # mp
    plt.errorbar(names, mp_means, yerr=mp_stds, label='Simulated datasets', linestyle="None", marker='x',
                 markersize=10.0, color='goldenrod')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("average mutation probability")
    plt.savefig(data_id + '_mp.png', bbox_inches="tight")

    plt.boxplot(mp, labels=names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("mutation probability")
    plt.savefig(data_id + '_mp_bp.png', bbox_inches="tight")

    # ts
    plt.errorbar(names, ts_means, yerr=ts_stds, label='Simulated data', linestyle="None", marker='x',
                 markersize=10.0, color='olivedrab')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("average tournament size")
    plt.savefig(data_id + '_ts.png', bbox_inches="tight")

    plt.boxplot(ts, labels=names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("tournament size")
    plt.savefig(data_id + '_ts_bp.png', bbox_inches="tight")

    plt.close('all')


if __name__ == "__main__":

    # parameter parser
    parser = argparse.ArgumentParser(description='A genetic algorithm (GA) optimizing a set of miRNA-based cell '
                                                 'classifiers for in situ cancer classification. Written by Melania '
                                                 'Nowicka, FU Berlin, 2019.\n\n')

    # adding arguments
    parser.add_argument('-d', '-d', dest="data", type=str,
                        help='data set file name')
    parser.add_argument('-f', '-f', dest="folds", type=int, default=None,
                        help='')
    parser.add_argument('-i', '-i', dest="id", type=str, default=None,
                        help='')

    # parse arguments
    parameters = parser.parse_args(sys.argv[1:])

    plot_parameters(parameters.data, parameters.folds, parameters.id)
