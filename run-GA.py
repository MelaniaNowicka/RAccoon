'''
A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import datetime
import sys
import random
import pandas as pd

# random.seed(0)


# single boolean function class
# inputs are connected with and AND
class SingleRule:

    def __init__(self, pos_inputs, neg_inputs):
        self.pos_inputs = pos_inputs  # non-negated inputs
        self.neg_inputs = neg_inputs  # negated inputs


# classifier (individual)
class Classifier:

    def __init__(self, rule_set, error_rates, bacc):
        self.rule_set = rule_set  # list of rules
        self.error_rates = error_rates  # dictionary of error rates (tp, tn, fp, fn)
        self.bacc = bacc  # balanced accuracy


# reading binarized data set.
def read_data(dataset_filename, log_message):

    # reading the data
    # throws an exception when datafile not found
    try:
        dataset = pd.read_csv(dataset_filename, sep='\t', header=0)
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

    # counting negative and positive samples
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

    log_message = log_message + "Number of relevant miRNAs according to a given threshold:" + str(len(relevant_mirna)) \
                  + "\n"
    log_message = log_message + "Number of irrelevant miRNAs according to a given threshold:" \
                  + str(len(irrelevant_mirna)) + "\n\n"

    log_message = log_message + "Relevant miRNAs: "

    for i in relevant_mirna:
        log_message = log_message + str(i) + "; "

    log_message = log_message + "\n\n"

    # removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    return dataset, relevant_mirna, log_message


# initialization of a single rule
def initialize_single_rule(temp_mirnas):

    pos_inputs = []
    neg_inputs = []

    #size of a single rule
    size = random.randrange(1, 3)

    for i in range(0, size): # drawing miRNAs for a rule (without replacement)

        mirna_sign = random.randrange(0, 2)  # randomly choosing miRNA sign
        mirna_id = random.randrange(0, len(temp_mirnas))  # randomly choosing miRNA ID

        # checking the miRNA sign to assign inputs to positive or negative group
        if mirna_sign == 0:
            pos_inputs.append(temp_mirnas[mirna_id])

        if mirna_sign == 1:
            neg_inputs.append(temp_mirnas[mirna_id])

        # removal of used miRNAs (individual must consist of i=unique miRNA IDs)
        del temp_mirnas[mirna_id]

    # initialization of a new single rule
    single_rule = SingleRule(pos_inputs, neg_inputs)

    return single_rule, temp_mirnas


# initialization of a new classifier
def initialize_classifier(classifier_size, mirnas, log_message):

    # size of a classifier
    size = random.randrange(1, classifier_size+1)

    # rules
    rule_set = []

    temp_mirnas = []
    temp_mirnas = mirnas.copy()

    # initialization of new rules
    for i in range(0, size):
        rule, temp_mirnas = initialize_single_rule(temp_mirnas)
        rule_set.append(rule)

    # initialization of a new classifier
    classifier = Classifier(rule_set, error_rates={}, bacc={})

    return classifier, log_message


# population initialization
def initialize_population(population_size, mirnas, classifier_size, log_message):

    population = []

    # initialization of n=population_size classifiers
    for i in range(0, population_size):
        classifier, log_message = initialize_classifier(classifier_size, mirnas, log_message)
        population.append(classifier)

    return population, log_message


# balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy


def write_generation_to_log(population, log_message):

    id = 1
    log_message = log_message + "\n\n"
    for classifier in population:

        log_message = log_message + "C" + str(id) + ": "
        id = id + 1
        classifier_message = ""
        for rule in classifier.rule_set:
            rule_message = ""

            if len(rule.neg_inputs) == 0 and len(rule.pos_inputs) == 1:
                rule_message = "(" + str(rule.pos_inputs[0]) + ")"
            if len(rule.pos_inputs) == 0 and len(rule.neg_inputs) == 1:
                rule_message = "(NOT " + str(rule.neg_inputs[0]) + ")"
            else:
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


# evaluation of the population
def evaluate_individuals(population, dataset, negatives, positives, best_bacc, best_classifier):

    # a list of rule results
    rule_outputs = []

    error_rates = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for classifier in population:  # evaluating every classifier
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for sample_id in range(0, len(dataset.index)):  # iterate through dataset skipping the header
            sample_output = 0  # single sample output
            rule_outputs.clear()  # clearing the rule outputs for a single sample

            for rule in classifier.rule_set:  # evaluating every rule in the classifier
                rule_output = 1
                for input in rule.pos_inputs:  # positive inputs
                    rule_output = rule_output and dataset.iloc[sample_id][input]

                for input in rule.neg_inputs:  # negative inputs
                    rule_output = rule_output and not dataset.iloc[sample_id][input]

                rule_outputs.append(rule_output)  # adding a single rule output to a list of rule outputs

            for result in rule_outputs:  # evaluating the final classifier output for a given sample
                sample_output = sample_output or result

            # counting tps, tns, fps and fns
            if dataset.iloc[sample_id]['Annots'] == 1 and sample_output == 1:
                true_positives = true_positives + 1
            if dataset.iloc[sample_id]['Annots'] == 0 and sample_output == 0:
                true_negatives = true_negatives + 1
            if dataset.iloc[sample_id]['Annots'] == 1 and sample_output == 0:
                false_positives = false_positives + 1
            if dataset.iloc[sample_id]['Annots'] == 0 and sample_output == 1:
                false_negatives = false_negatives + 1

        # assigning classifier scores
        classifier.error_rates["tp"] = true_positives
        classifier.error_rates["tn"] = true_negatives
        classifier.error_rates["fp"] = false_positives
        classifier.error_rates["fn"] = false_negatives
        classifier.bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)

        if best_bacc < classifier.bacc:
            best_bacc = classifier.bacc
            best_classifier = classifier

    return best_bacc, best_classifier


# one parent selection function
def select_parent(population, tournament_size, first_parent_id):

    tournament = []

    # drawing parents from a population without replacement
    for i in range(0, int(tournament_size/100*len(population))):
        candidate = random.randrange(0, len(population))
        while candidate in tournament or candidate == first_parent_id:
            candidate = random.randrange(0, len(population))
        else:
            tournament.append(candidate)

    # choosing the best parent for crossover
    best_candidate = tournament[0]
    for candidate in tournament:
        if population[candidate].bacc > population[best_candidate].bacc:
            best_candidate = candidate

    parent = best_candidate

    return parent


# tournament selection of parents for crossover
def selection(population, tournament_size):

    first_parent_id = select_parent(population, tournament_size, -1)
    second_parent_id = select_parent(population, tournament_size, first_parent_id)

    return first_parent_id, second_parent_id


# crossover
def crossover(population, first_parent_id, second_parent_id):

    # checking sizes of parents and assigning rule sets
    # first parent = more rules, second parents = less rules
    # if equal - assign first to first, second to second
    if len(population[first_parent_id].rule_set) < len(population[second_parent_id].rule_set):
        first_parent_rule_set = population[second_parent_id].rule_set
        second_parent_rule_set = population[first_parent_id].rule_set
    else:
        first_parent_rule_set = population[first_parent_id].rule_set
        second_parent_rule_set = population[second_parent_id].rule_set

    #creating new population

    # creating empty offspring
    first_child = Classifier([], {}, 0.0)
    second_child = Classifier([], {}, 0.0)

    # if the first parent consists of more rules
    if len(first_parent_rule_set) > len(second_parent_rule_set):
        difference = len(first_parent_rule_set) - len(second_parent_rule_set)  # difference between sizes of parents
        # crossover index specifies the position of the second parent in relation to the first one
        crossover_index_second_parent = random.randrange(0, difference+1)

        for i in range(0, len(first_parent_rule_set)):  # iterating through the first parent
            swap_mask = random.randrange(0, 2)  # randomly choosing the mask
            if swap_mask == 1:  # if mask=1 swap elements

                # check the position of the second classifier
                # if the parents are not aligned in i move element i from the first parent to the second child
                if i < crossover_index_second_parent or i >= crossover_index_second_parent + len(second_parent_rule_set):
                    second_child.rule_set.append(first_parent_rule_set[i])  # move element i to the second child
                else:  # else, swap elements
                    second_child.rule_set.append(first_parent_rule_set[i])
                    first_child.rule_set.append(second_parent_rule_set[i-crossover_index_second_parent])
            else:  # if mask=0 do not swap elements and copy elements from parents to offspring if possible
                if i < crossover_index_second_parent or i >= crossover_index_second_parent + len(second_parent_rule_set):
                    first_child.rule_set.append(first_parent_rule_set[i])
                else:
                    second_child.rule_set.append(second_parent_rule_set[i-crossover_index_second_parent])
                    first_child.rule_set.append(first_parent_rule_set[i])

    else:  # if parents have the same length
        for i in range(0, len(first_parent_rule_set)):  # iterate over first parent
            swap_mask = random.randrange(0, 2)
            if swap_mask == 1:  # swap
                second_child.rule_set.append(first_parent_rule_set[i])
                first_child.rule_set.append(second_parent_rule_set[i])
            else:  # do not swap, just copy
                second_child.rule_set.append(second_parent_rule_set[i])
                first_child.rule_set.append(first_parent_rule_set[i])

    return first_child, second_child


def mutate(rule, mirnas):

    mutation_type = random.randrange(0, 3)  # choose mutation type

    positive_inputs = len(rule.pos_inputs)
    negative_inputs = len(rule.neg_inputs)

    # if rule consists of positive and negative inputs
    if positive_inputs != 0 and negative_inputs != 0:

        if mutation_type == 0:  # miRNA-id swap
            input_type = random.randrange(0, 2)  # choose input type

            if input_type == 0:  # positive inputs
                input_id = random.randrange(0, positive_inputs)
                new_input_id = random.randrange(0, len(mirnas))
                rule.pos_inputs.append(mirnas[new_input_id])
                del rule.pos_inputs[input_id]

            if input_type == 1:  # negative inputs
                input_id = random.randrange(0, negative_inputs)
                new_input_id = random.randrange(0, len(mirnas))
                rule.neg_inputs.append(mirnas[new_input_id])
                del rule.neg_inputs[input_id]

        if mutation_type == 1:  # add/remove NOT
            input_type = random.randrange(0, 2)

            if input_type == 0:  # positive inputs
                input_id = random.randrange(0, positive_inputs)
                rule.neg_inputs.append(rule.pos_inputs[input_id])
                del rule.pos_inputs[input_id]

            if input_type == 1:  # negative inputs
                input_id = random.randrange(0, negative_inputs)
                rule.pos_inputs.append(rule.neg_inputs[input_id])
                del rule.neg_inputs[input_id]

        if mutation_type == 2:  # remove or add miRNA
            input_type = random.randrange(0, 2)

            if input_type == 0:
                input_id = random.randrange(0, positive_inputs)
                del rule.pos_inputs[input_id]

            if input_type == 1:
                input_id = random.randrange(0, negative_inputs)
                del rule.neg_inputs[input_id]

    if positive_inputs == 0:

        if mutation_type == 0:
            input_id = random.randrange(0, negative_inputs)
            new_input_id = random.randrange(0, len(mirnas))
            rule.neg_inputs.append(mirnas[new_input_id])
            del rule.neg_inputs[input_id]

        if mutation_type == 1:
            input_id = random.randrange(0, negative_inputs)
            rule.pos_inputs.append(rule.neg_inputs[input_id])
            del rule.neg_inputs[input_id]

        if mutation_type == 2:  # remove or add miRNA

            if negative_inputs == 1:
                add_pos_or_neg = random.randrange(0, 2)
                if add_pos_or_neg == 0:
                    new_input_id = random.randrange(0, len(mirnas))
                    rule.pos_inputs.append(mirnas[new_input_id])
                if add_pos_or_neg == 1:
                    new_input_id = random.randrange(0, len(mirnas))
                    rule.neg_inputs.append(mirnas[new_input_id])

            if negative_inputs != 1:
                input_id = random.randrange(0, negative_inputs)
                del rule.neg_inputs[input_id]

    if negative_inputs == 0:

        if mutation_type == 0:
            input_id = random.randrange(0, positive_inputs)
            new_input_id = random.randrange(0, len(mirnas))
            rule.pos_inputs.append(mirnas[new_input_id])
            del rule.pos_inputs[input_id]

        if mutation_type == 1:
            input_id = random.randrange(0, positive_inputs)
            rule.neg_inputs.append(rule.pos_inputs[input_id])
            del rule.pos_inputs[input_id]

        if mutation_type == 2:  # remove or add miRNA

            if positive_inputs == 1:
                add_pos_or_neg = random.randrange(0,2)
                if add_pos_or_neg == 0:
                    new_input_id = random.randrange(0, len(mirnas))
                    rule.pos_inputs.append(mirnas[new_input_id])
                if add_pos_or_neg == 1:
                    new_input_id = random.randrange(0, len(mirnas))
                    rule.neg_inputs.append(mirnas[new_input_id])

            if positive_inputs != 1:
                input_id = random.randrange(0, positive_inputs)
                del rule.pos_inputs[input_id]


# mutation
def mutation(population, mirnas, mutation_probability):

    for classifier in population:
        rand = random.random()
        if rand <= mutation_probability:
            rule_id = random.randrange(0, len(classifier.rule_set))  # choose rule for mutation
            rule = classifier.rule_set[rule_id]
            mutate(rule, mirnas)

    return population


def run_genetic_algorithm(dataset_filename, iterations, population_size, classifier_size, crossover_probability,
                          mutation_probability, tournament_size):

    # starting log message
    log_message = "A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer " \
                  "classification. Written by Melania Nowicka, FU Berlin, 2019.\n\n"
    log_file_name = "log_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".txt"  # log name

    # read data
    data, negatives, positives, log_message = read_data(dataset_filename, log_message)

    # remove irrelevant miRNAs
    datasetR, mirnas, log_message = remove_irrelevant_mirna(data, log_message)

    # population initialization
    population, log_message = initialize_population(population_size, mirnas, classifier_size, log_message)

    best_classifier = Classifier(rule_set=[], error_rates={}, bacc=0.0)
    best_bacc = 0.0

    # first evaluation of individuals
    best_bacc, best_classifier = evaluate_individuals(population, datasetR, negatives, positives, best_bacc, best_classifier)

    # write log to a file
    log_message = write_generation_to_log(population, log_message)

    with open(log_file_name, "w") as log_file:
        log_file.write(log_message)

    log_message = ""

    # next generations
    for iteration in range(0, iterations):

        print(int(iteration/iterations*100), "% | BACC: ", best_bacc)  # show progress and current best score
        new_population = []  # create new population
        parents = []  # parents index (parent used for crossover cannot be copied to the new population)

        for i in range(0, int(population_size)):  # iterating through
            crossover_rand = random.random()
            # if the crossover_rand is lower than probability - apply crossover
            if crossover_rand <= crossover_probability:
                # selection of parents for crossover
                first_parent_id, second_parent_id = selection(population, tournament_size)
                # selected parents are added to a list of used parents
                parents.append(first_parent_id)
                parents.append(second_parent_id)
                # crossover
                first_child, second_child = crossover(population, first_parent_id, second_parent_id)
                # adding offspring to the new population
                new_population.append(first_child)
                new_population.append(second_child)
        # adding parents that were not used for crossover to the population
        for j in range(0, int(population_size)):
            if j not in parents:
                new_population.append(population[j])
        # mutation
        population = mutation(population, mirnas, mutation_probability)
        # evaluation
        best_bacc, best_classifier = evaluate_individuals(population, data, negatives, positives, best_bacc, best_classifier)

        # writting log message to a file
        log_message = write_generation_to_log(population, log_message)
        with open(log_file_name, "w") as log_file:
            log_file.write(log_message)

    print("BEST BACC:", best_bacc)
    classifier_message = ""

    for rule in best_classifier.rule_set:
        rule_message = ""

        if len(rule.neg_inputs) == 0 and len(rule.pos_inputs) == 1:
            rule_message = "(" + str(rule.pos_inputs[0]) + ")"
        if len(rule.pos_inputs) == 0 and len(rule.neg_inputs) == 1:
            rule_message = "(NOT " + str(rule.neg_inputs[0]) + ")"
        else:
            for input in rule.pos_inputs:
                rule_message = rule_message + "(" + input + ")"
            for input in rule.neg_inputs:
                rule_message = rule_message + "(NOT " + input + ")"

        rule_message = " [" + rule_message + "] "

        rule_message = rule_message.replace(")(", ") AND (")

        classifier_message = classifier_message + rule_message

    print("BEST CLASSIFIER: ", classifier_message)

    # writing the log message to file
    log_message = write_generation_to_log(population, log_message)
    with open(log_file_name, "w") as log_file:
        log_file.write(log_message)


if __name__ == "__main__":

    dataset_filename = sys.argv[1]
    run_genetic_algorithm(dataset_filename, 10, 10, 3, 0.75, 0.5, 30)
