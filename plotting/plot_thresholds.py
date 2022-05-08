import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os


def plot_thresholds(directory, folds, names, data_id):

    thresholds = []
    fold_counter = 0
    names = names.split(';')

    print(os.listdir(directory))
    thresholds_temp = []

    for filename in os.listdir(directory):

        print(filename)
        df = pd.read_csv(directory+filename, sep=';')
        thresholds_temp = thresholds_temp + df["THRESHOLD"].to_list()
        fold_counter += 1
        print(len(thresholds_temp))

        if fold_counter == 5:

            thresholds.append(thresholds_temp)
            thresholds_temp = []
            fold_counter = 0

    print(len(thresholds))
    plt.boxplot(thresholds, labels=names)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0.25, 1.25, 0.25))
    plt.tight_layout()
    plt.ylabel("threshold")
    plt.savefig(data_id + '_weights_bp.png', bbox_inches="tight")


if __name__ == "__main__":

    # parameter parser
    parser = argparse.ArgumentParser(description='A genetic algorithm (GA) optimizing a set of miRNA-based cell '
                                                 'classifiers for in situ cancer classification. Written by Melania '
                                                 'Nowicka, FU Berlin, 2019.\n\n')

    # adding arguments
    parser.add_argument('-d', '-d', dest="directory", type=str,
                        help='data set file name')
    parser.add_argument('-f', '-f', dest="folds", type=int, help='')
    parser.add_argument('-n', '-n', dest="names", type=str)
    parser.add_argument('-i', '-i', dest="id", type=str, help='')

    # parse arguments
    parameters = parser.parse_args(sys.argv[1:])

    plot_thresholds(parameters.directory, parameters.folds, parameters.names, parameters.id)




