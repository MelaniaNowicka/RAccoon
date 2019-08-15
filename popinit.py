import random


# single boolean function class
# inputs are connected with and AND
class SingleRule:

    def __init__(self, pos_inputs, neg_inputs):
        self.pos_inputs = pos_inputs  # non-negated inputs
        self.neg_inputs = neg_inputs  # negated inputs

    # copy object
    def __copy__(self):
        new_pos_inputs = []
        new_neg_inputs = []

        for input in self.pos_inputs:
            new_pos_inputs.append(input)

        for input in self.neg_inputs:
            new_neg_inputs.append(input)

        return SingleRule(new_pos_inputs, new_neg_inputs)


# classifier (individual)
class Classifier:

    def __init__(self, rule_set, error_rates, bacc):
        self.rule_set = rule_set  # list of rules
        self.error_rates = error_rates  # dictionary of error rates (tp, tn, fp, fn)
        self.bacc = bacc  # balanced accuracy

    # copy object
    def __copy__(self):
        new_rule_set = []

        for rule in self.rule_set:
            new_rule_set.append(rule.__copy__())

        return Classifier(new_rule_set, self.error_rates, self.bacc)

    # remove repeating rules in classifiers
    def remove_duplicates(self):
        single_pos = []  # list of positive single-input rules and their ids
        single_neg = []  # list of negative single-input rules and their ids
        multi_input = []  # list of mixed rules with size > 1 and their ids

        to_del = []  # rules to be deleted

        # separating by rule type
        for id in range(0, len(self.rule_set)):
            # collecting positive single-input rules
            if len(self.rule_set[id].pos_inputs) == 0 and len(self.rule_set[id].neg_inputs) == 1:
                single_neg.append((self.rule_set[id], id))
            # collecting negative single-input rules
            elif len(self.rule_set[id].pos_inputs) == 1 and len(self.rule_set[id].neg_inputs) == 0:
                single_pos.append((self.rule_set[id], id))
            # collecting mixed rules
            else:
                multi_input.append((self.rule_set[id], id))

        # collecting ids of rules for removal
        # for positive single-input rules
        if len(single_pos) > 1:  # if more than one rule exist
            for i in range(0, len(single_pos)-1):  # iterate over rules
                for j in range(i+1, len(single_pos)):
                    if single_pos[i][0].pos_inputs == single_pos[j][0].pos_inputs:  # if inputs are identical
                        to_del.append(single_pos[j][1])  # add rule for removal
        # for negative single-input rules
        if len(single_neg) > 1:  # if more than one rule exist
            for i in range(0, len(single_neg)-1):  # iterate over rules
                for j in range(i+1, len(single_neg)):
                    if single_neg[i][0].neg_inputs == single_neg[j][0].neg_inputs:  # if inputs are identical
                        to_del.append(single_neg[j][1])  # add rule for removal
        # for mixed rules
        if len(multi_input) > 1:  # if more than one rule exist
            for i in range(0, len(multi_input)-1):  # iterate over rules
                for j in range(i+1, len(multi_input)):
                    # for rules with two positive inputs
                    if len(multi_input[i][0].pos_inputs) == 2 and len(multi_input[j][0].pos_inputs) == 2:
                        if multi_input[i][0].pos_inputs == multi_input[j][0].pos_inputs:  # if inputs are identical
                            to_del.append(multi_input[j][1])  # add rule for removal
                    # for rules with two negative inputs
                    elif len(multi_input[i][0].neg_inputs) == 2 and len(multi_input[j][0].neg_inputs) == 2:
                        if multi_input[i][0].neg_inputs == multi_input[j][0].neg_inputs:  # if inputs are identical
                            to_del.append(multi_input[j][1])  # add rule for removal
                    # for rules with mixed inputs
                    elif len(multi_input[i][0].pos_inputs) == 1 and len(multi_input[i][0].neg_inputs) == 1:
                        if len(multi_input[j][0].pos_inputs) == 1 and len(multi_input[j][0].neg_inputs) == 1:
                            if multi_input[i][0].pos_inputs == multi_input[j][0].pos_inputs:  # if inputs are identical
                                if multi_input[i][0].neg_inputs == multi_input[j][0].neg_inputs:
                                    to_del.append(multi_input[j][1])  # add rule for removal

        # sort indices in descending order for removal
        to_del = list(set(to_del))
        to_del.sort(reverse=True)
        # remove duplicates
        for id in to_del:
            del self.rule_set[id]


# initialization of a single rule
# temp_mirnas consists of mirnas that were not used in the classifier yet
def initialize_single_rule(temp_mirnas):

    pos_inputs = []  # list of positive inputs
    neg_inputs = []  # list of negative inputs

    # size of a single rule
    size = random.randrange(1, 3)

    for i in range(0, size):  # drawing miRNAs for a rule (without replacement) respecting size

        mirna_sign = random.randrange(0, 2)  # randomly choosing miRNA sign
        mirna_id = random.randrange(0, len(temp_mirnas))  # randomly choosing miRNA ID

        # checking the miRNA sign to assign inputs to positive or negative group
        if mirna_sign == 0:
            neg_inputs.append(temp_mirnas[mirna_id])

        if mirna_sign == 1:
            pos_inputs.append(temp_mirnas[mirna_id])

        # removal of used miRNAs (rule must consist of i=unique miRNA IDs)
        del temp_mirnas[mirna_id]

    # initialization of a new single rule
    single_rule = SingleRule(pos_inputs, neg_inputs)

    return single_rule, temp_mirnas


# initialization of a new classifier
def initialize_classifier(classifier_size, mirnas
                          ):

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

    return classifier


# population initialization
def initialize_population(population_size,
                          mirnas,
                          classifier_size):

    population = []  # empty population

    # initialization of n=population_size classifiers
    for i in range(0, population_size):
        classifier = initialize_classifier(classifier_size, mirnas)
        population.append(classifier)

    return population
