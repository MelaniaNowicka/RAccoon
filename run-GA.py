'''
An genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import sys
import pandas as pd


# removing irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset):

    relevant_mirna = []
    irrelevant_mirna = []

    # sum of miRNA levels (0/1) in each column
    column_sum = dataset.sum(axis=0, skipna=True)


    # if miRNA levels sum up to 0 or the number of samples in the dataset - miRNA is irrelevant (non-regulated)
    # (in other words: the whole column is filled in with 0s or 1s)
    for id, sum in column_sum.items():
        if id not in ["ID", "Annots"]:
            if sum == 0 or sum == len(dataset.index):
                irrelevant_mirna.append(id)
            else:
                relevant_mirna.append(id)

    print("Number of relevant miRNAs according to a given threshold:", len(relevant_mirna))
    print("Number of irrelevant miRNAs according to a given threshold:", len(irrelevant_mirna))

    dataset = dataset.drop(irrelevant_mirna, axis=1)

    return dataset

# Reading binarized data set.
def read_data(dataset_filename):

    # trying to read the data
    # throws an exception when datafile not found
    try:
        dataset = pd.read_csv(dataset_filename, sep='\t', header=0)
    except IOError:
        print("Error: No such file or directory.")
        sys.exit(0)

    # simple check whether data is in the right format
    # needs to be improved
    header = []
    header = dataset.columns.values.tolist()

    if header[0] != 'ID' or header[1] != 'Annots':
        print("Error: wrong format. The first column must include sample IDs and the second "
              "- the annotation of samples.")
        sys.exit(0)

    return dataset




def run_genetic_algorithm(dataset_filename):

    # read data
    data = read_data(dataset_filename)
    # remove irrelevant miRNAs
    datasetR = remove_irrelevant_mirna(data)






if __name__== "__main__":
    dataset_filename = sys.argv[1]
    run_genetic_algorithm(dataset_filename)
