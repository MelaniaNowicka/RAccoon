import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def plot_results(data_file, folds, data_id):

    df = pd.read_csv(data_file, sep=";", header=0)
    names = sorted(set(df['ID']))

    rules = []
    bacc_train = []
    bacc_test = []

    for i in range(0, len(names)):
        rules.append(df['RULES'][i * folds:i * folds + folds].values)
        bacc_train.append(df['TRAIN'][i * folds:i * folds + folds].values)
        bacc_test.append(df['TEST'][i * folds:i * folds + folds].values)

    fig = plt.figure()

    fig, ax = plt.subplots()
    plt.boxplot(rules, labels=names, patch_artist=True, boxprops=dict(facecolor='#bfcd53ff'),
                medianprops=dict(color='#14645aff'))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("number of rules")
    plt.savefig(data_id + '_bacc_rules_bp.png', bbox_inches="tight")

    fig, ax = plt.subplots()
    plt.boxplot(bacc_train, labels=names, patch_artist=True, boxprops=dict(facecolor='#bfcd53ff'),
                medianprops=dict(color='#14645aff'))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("balanced accuracy")
    plt.savefig(data_id + '_bacc_train_bp.png', bbox_inches="tight")

    fig, ax = plt.subplots()
    plt.boxplot(bacc_test, labels=names, patch_artist=True, boxprops=dict(facecolor='#bfcd53ff'),
                medianprops=dict(color='#14645aff'))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("balanced accuracy")
    plt.savefig(data_id + '_bacc_test_bp.png', bbox_inches="tight")

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

    plot_results(parameters.data, parameters.folds, parameters.id)
