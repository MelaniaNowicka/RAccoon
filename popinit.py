import random


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


# initialization of a single rule
def initialize_single_rule(temp_mirnas):

    pos_inputs = []
    neg_inputs = []

    # size of a single rule
    size = random.randrange(1, 3)

    for i in range(0, size):  # drawing miRNAs for a rule (without replacement)

        mirna_sign = random.randrange(0, 2)  # randomly choosing miRNA sign
        mirna_id = random.randrange(0, len(temp_mirnas))  # randomly choosing miRNA ID

        # checking the miRNA sign to assign inputs to positive or negative group
        if mirna_sign == 0:
            pos_inputs.append(temp_mirnas[mirna_id])

        if mirna_sign == 1:
            neg_inputs.append(temp_mirnas[mirna_id])

        # removal of used miRNAs (rule must consist of i=unique miRNA IDs)
        del temp_mirnas[mirna_id]

    # initialization of a new single rule
    single_rule = SingleRule(pos_inputs, neg_inputs)

    #check_rule(single_rule)

    return single_rule, temp_mirnas


# initialization of a new classifier
def initialize_classifier(classifier_size, mirnas, log_message):

    # size of a classifier
    size = random.randrange(1, classifier_size+1)

    # rules
    rule_set = []

    # copy of mirna list
    temp_mirnas = mirnas.copy()

    # initialization of new rules
    for i in range(0, size):
        rule, temp_mirnas = initialize_single_rule(temp_mirnas)
        rule_set.append(rule)

    # initialization of a new classifier
    classifier = Classifier(rule_set, error_rates={}, bacc={})

    return classifier, log_message


# population initialization
def initialize_population(population_size,
                          mirnas,
                          classifier_size,
                          log_message):

    population = []

    # initialization of n=population_size classifiers
    for i in range(0, population_size):
        classifier, log_message = initialize_classifier(classifier_size, mirnas, log_message)
        population.append(classifier)

    return population, log_message
