'''
A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import datetime
import sys
import random
import pandas as pd

#random.seed(0)

#single boolean function class
#inputs are connected with and AND
class SingleFunction:

    def __init__(self, size, pos_inputs, neg_inputs):
        self.size = size #size of a single function
        self.pos_inputs = pos_inputs #non-negated inputs
        self.neg_inputs = neg_inputs #negated inputs


#classifier (individual)
class Classifier:

    def __init__(self, size, rule_set, error_rates, bacc):
        self.size = size #size of a classifier
        self.rule_set = rule_set #list of rules
        self.error_rates = error_rates #dictionary of error rates (tp, tn, fp, fn)
        self.bacc = bacc #balanced accuracy

#reading binarized data set.
def read_data(dataset_filename, log_message):

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

    #counting negative and positive samples
    samples = len(dataset.index)
    negatives = dataset[dataset["Annots"]==0].count()["Annots"]
    positives = samples - negatives

    log_message = log_message + "Number of samples: " + str(samples) + "\n"
    log_message = log_message + "Number of negative samples: " + str(negatives) + "\n"
    log_message = log_message + "Number of positive samples: " + str(positives) + "\n\n"

    if negatives == 0 or positives == 0:
        print("Error: no negative or positive samples in the dataset!")
        sys.exit(0)

    return dataset, negatives, positives, log_message


#removal of irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset, log_message):

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

    log_message = log_message + "Number of relevant miRNAs according to a given threshold:" + str(len(relevant_mirna)) \
                  + "\n"
    log_message = log_message + "Number of irrelevant miRNAs according to a given threshold:" \
                  + str(len(irrelevant_mirna)) + "\n\n"

    log_message = log_message + "Relevant miRNAs: "

    for i in relevant_mirna:
        log_message = log_message + str(i) + "; "

    log_message = log_message + "\n\n"

    #removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    return dataset, relevant_mirna, log_message


#initialization of a single rule
def initialize_single_rule(temp_mirnas):

    pos_inputs = []
    neg_inputs = []

    #size of a single rule
    size = random.randrange(1, 3)

    #drawing miRNAs for a rule (without returning)
    for i in range(0, size):

        #randomly choosing miRNA sign
        mirna_sign = random.randrange(0, 2)
        #randomly choosing miRNA ID
        mirna_id = random.randrange(0, len(temp_mirnas))

        #checking the miRNA sign to assign inputs to positive or negative group
        if mirna_sign == 0:
            pos_inputs.append(temp_mirnas[mirna_id])

        if mirna_sign == 1:
            neg_inputs.append(temp_mirnas[mirna_id])

        #removal of used miRNAs (individual must consist of i=unique miRNA IDs)
        del temp_mirnas[mirna_id]

    #initialization of a new single rule
    single_rule = SingleFunction(size, pos_inputs, neg_inputs)

    return single_rule, temp_mirnas


#initialization of a new classifier
def initialize_classifier(mirnas, log_message):

    #size of a classifier
    size = random.randrange(1, 5)

    #rules
    rule_set = []

    temp_mirnas = []
    temp_mirnas = mirnas.copy()

    #initialization of new rules
    for i in range(0, size):
        rule, temp_mirnas = initialize_single_rule(temp_mirnas)
        rule_set.append(rule)


    #initialization of a new classifier
    classifier = Classifier(size, rule_set, error_rates={}, bacc={})

    return classifier, log_message


#population initialization
def initialize_population(population_size, mirnas, log_message):

    population = []

    #initialization of n=population_size classifiers
    for i in range(0, population_size):
        classifier, log_message = initialize_classifier(mirnas, log_message)
        population.append(classifier)

    return population, log_message

#balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy

def write_generation_to_log(population, log_message):

    id = 1

    for classifier in population:
        log_message = log_message + "C" + str(id) + ": "
        id = id + 1
        classifier_message = ""
        for rule in classifier.rule_set:
            rule_message = ""

            if rule.size == 1 and len(rule.pos_inputs) != 0:
                rule_message = "(" + str(rule.pos_inputs[0]) + ")"
            if rule.size == 1 and len(rule.neg_inputs) != 0:
                rule_message = "(NOT " + str(rule.neg_inputs[0]) + ")"

            if rule.size != 1:
                for input in rule.pos_inputs:
                    rule_message = rule_message + "(" + input + ")"
                for input in rule.neg_inputs:
                    rule_message = rule_message + "(NOT " + input + ")"

            rule_message = " [" + rule_message + "] "

            rule_message = rule_message.replace(")(", ") AND (")

            classifier_message = classifier_message + rule_message

        log_message = log_message + classifier_message + "\n"
        log_message = log_message + "BACC: " + str(classifier.bacc) + "; TP: " + str(classifier.error_rates["tp"]) \
                      + "; TN: " + str(classifier.error_rates["tn"]) + "\n"

    return log_message

#evaluation of the population
def evaluate_individuals(population, dataset, negatives, positives):

    #a list of rule results
    rule_outputs = []

    error_rates = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    bacc = 0.0

    i = 1

    for classifier in population:  # evaluating every classfier
        print("C" + str(i))
        i = i + 1

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for sample_id, sample_profile in dataset.iterrows(): #iterate through dataset skipping the header
            sample_output = 0 #single sample output
            rule_outputs.clear() #clearing the rule outputs for a single sample

            for rule in classifier.rule_set: #evaluating every rule in the classifier
                rule_output = 1
                for input in rule.pos_inputs: #positive inputs
                    rule_output = rule_output and dataset.iloc[sample_id][input]

                for input in rule.neg_inputs: #negative inputs
                    rule_output = rule_output and not dataset.iloc[sample_id][input]

                rule_outputs.append(rule_output) #adding a single rule output to a list of rule outputs

            for result in rule_outputs: #evaluating the final classifier output for a given sample
                sample_output = sample_output or result

            #counting tps, tns, fps and fns
            if dataset.iloc[sample_id]['Annots'] == 1 and sample_output == 1:
                true_positives = true_positives + 1
            if dataset.iloc[sample_id]['Annots'] == 0 and sample_output == 0:
                true_negatives = true_negatives + 1
            if dataset.iloc[sample_id]['Annots'] == 1 and sample_output == 0:
                false_positives = false_positives + 1
            if dataset.iloc[sample_id]['Annots'] == 0 and sample_output == 1:
                false_negatives = false_negatives + 1

        #assigning classifier scores
        classifier.error_rates["tp"] = true_positives
        classifier.error_rates["tn"] = true_negatives
        classifier.error_rates["fp"] = false_positives
        classifier.error_rates["fn"] = false_negatives
        classifier.bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)

def run_genetic_algorithm(dataset_filename, population_size, iterations):

    log_message = "A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer " \
                  "classification. Written by Melania Nowicka, FU Berlin, 2019.\n\n"

    #read data
    data, negatives, positives, log_message = read_data(dataset_filename, log_message)
    #remove irrelevant miRNAs
    datasetR, mirnas, log_message = remove_irrelevant_mirna(data, log_message)
    #population initialization
    population, log_message = initialize_population(population_size, mirnas, log_message)
    evaluate_individuals(population, datasetR, negatives, positives)

    log_message = write_generation_to_log(population, log_message)

    log_file_name = "log_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".txt"
    log_file = open(log_file_name, "w")
    log_file.write(log_message)


if __name__ == "__main__":

    dataset_filename = sys.argv[1]
    run_genetic_algorithm(dataset_filename, 1)
