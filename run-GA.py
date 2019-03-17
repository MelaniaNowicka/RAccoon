'''
An genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import sys
import random
import pandas as pd


#single boolean function class
#inputs are connected with and AND
class SingleFunction:

    def __init__(self, size, pos_inputs, neg_inputs):
        self.size = size #size of a single function
        self.pos_inputs = pos_inputs #non-negated inputs
        self.neg_inputs = neg_inputs #negated inputs

#initialization of a single rule
def initialize_single_rule(mirnas):

    pos_inputs = []
    neg_inputs = []

    #size of a single rule
    size = random.randrange(1, 3)

    #drawing miRNAs for a rule (without returning)
    for i in range(0, size):

        #randomly choosing miRNA sign
        mirna_sign = random.randrange(0, 2)
        #randomly choosing miRNA ID
        mirna_id = random.randrange(0, len(mirnas))


        #checking the miRNA sign to assign inputs to positive or negative group
        if mirna_sign == 0:
            pos_inputs.append(mirnas[mirna_id])

        if mirna_sign == 1:
            neg_inputs.append(mirnas[mirna_id])

        #removal of used miRNAs (individual must consist of i=unique miRNA IDs)
        del mirnas[mirna_id]

    #initialization of a new single rule
    single_rule = SingleFunction(size,pos_inputs,neg_inputs)

    return single_rule, mirnas

#classifier (individual)
class Classifier:

    def __init__(self, size, rule_set):
        self.size = size #size of a classifier
        self.rule_set = rule_set #list of rules


#initialization of a new classifier
def initialize_classifier(mirnas):

    #size of a classifier
    size = random.randrange(1, 6)

    #rules
    rule_set = []

    temp_mirnas = mirnas

    #initialization of new rules
    for i in range(0, size):
        rule, temp_mirnas = initialize_single_rule(temp_mirnas)
        rule_set.append(rule)

    #initialization of a new classifier
    classifier = Classifier(size, rule_set)

    return classifier

#population initialization
def initialize_population(population_size, mirnas):

    population = []

    for i in range (0, population_size):
        classifier = initialize_classifier(mirnas)
        population.append(classifier)

    return population

#removal of irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset):

    relevant_mirna = []
    irrelevant_mirna = []

    #sum of miRNA levels (0/1) in each column
    column_sum = dataset.sum(axis=0, skipna=True)

    #if miRNA levels sum up to 0 or the number of samples in the dataset - miRNA is irrelevant (non-regulated)
    #(in other words: the whole column is filled in with 0s or 1s)
    for id, sum in column_sum.items():
        if id not in ["ID", "Annots"]:
            if sum == 0 or sum == len(dataset.index):
                irrelevant_mirna.append(id)
            else:
                relevant_mirna.append(id)

    print("Number of relevant miRNAs according to a given threshold:", len(relevant_mirna))
    print("Number of irrelevant miRNAs according to a given threshold:", len(irrelevant_mirna))

    #removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    return dataset, relevant_mirna

#reading binarized data set.
def read_data(dataset_filename):

    #trying to read the data
    #throws an exception when datafile not found
    try:
        dataset = pd.read_csv(dataset_filename, sep='\t', header=0)
    except IOError:
        print("Error: No such file or directory.")
        sys.exit(0)

    #simple check whether data is in the right format
    #needs to be improved
    header = dataset.columns.values.tolist()

    if header[0] != 'ID' or header[1] != 'Annots':
        print("Error: wrong format. The first column must include sample IDs and the second "
              "- the annotation of samples.")
        sys.exit(0)

    return dataset

def run_genetic_algorithm(dataset_filename, population_size):

    #read data
    data = read_data(dataset_filename)
    #remove irrelevant miRNAs
    datasetR, mirnas = remove_irrelevant_mirna(data)
    #population initialization
    population = initialize_population(population_size, mirnas)

if __name__ == "__main__":

    dataset_filename = sys.argv[1]
    run_genetic_algorithm(dataset_filename, 10)
