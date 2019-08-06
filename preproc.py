import sys
import pandas as pd


# reading binarized data set.
def read_data(dataset_filename, log_message):

    # reading the data
    # throws an exception when datafile not found
    try:
        dataset = pd.read_csv(dataset_filename, sep=';', header=0)
    except IOError:
        print("Error: No such file or directory.")
        sys.exit(0)

    # simple check whether data is in the right format
    # needs to be improved
    header = dataset.columns.values.tolist()

    if header[0] != 'ID' or header[1] != 'Annots':
        print("Error: wrong format. The first column must include sample IDs and the second "
              "- the annotation of samples.")
        sys.exit(0)

    # extract miRNA's names
    mirnas = header[2:]
    # counting negative and positive samples
    samples = len(dataset.index)
    negatives = dataset[dataset["Annots"] == 0].count()["Annots"]
    positives = samples - negatives

    log_message = log_message + "Number of samples: " + str(samples) + "\n"
    log_message = log_message + "Number of negative samples: " + str(negatives) + "\n"
    log_message = log_message + "Number of positive samples: " + str(positives) + "\n\n"

    print("Number of samples: " + str(samples))
    print("Number of negative samples: " + str(negatives))
    print("Number of positive samples: " + str(positives))

    if negatives == 0 or positives == 0:
        print("Error: no negative or positive samples in the dataset!")
        sys.exit(0)

    return dataset, negatives, positives, mirnas, log_message


# removal of irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset, log_message):

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

    # removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    # creating log message
    log_message = log_message + "Number of relevant miRNAs according to a given threshold: " + str(len(relevant_mirna)) \
                  + "\n"
    log_message = log_message + "Number of irrelevant miRNAs according to a given threshold: " \
                  + str(len(irrelevant_mirna)) + "\n\n"

    log_message = log_message + "Relevant miRNAs: "

    for i in relevant_mirna:
        log_message = log_message + str(i) + "; "

    log_message = log_message + "\n\n"

    return dataset, relevant_mirna, log_message