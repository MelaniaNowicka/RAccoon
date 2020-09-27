import popinit
import random

random.seed(1)


# mutate single rule
def mutate_rule(rule, features):

    # choose mutation type (0, 1, 2)
    # 0 - feature-id swap
    # 1 - add/remove NOT
    # 2 - remove input (cannot add input as the maximal size is 2 inputs per rule)
    mutation_type = random.randint(0, 2)
    temp_features = features.copy()  # copy features to a temporary list

    # remove inputs that are in the rule
    for input in rule.pos_inputs:  # remove positive inputs
        temp_features.remove(input)

    for input in rule.neg_inputs:  # remove negative inputs
        temp_features.remove(input)

    mutated = False  # rule is unchanged

    pos_inputs_len = len(rule.pos_inputs)  # number of positive inputs in a rule
    neg_inputs_len = len(rule.neg_inputs)  # number of negative inputs in a rule

    # if rule consists of positive and negative inputs
    if mutated is False:  # if rule is unchanged
        if pos_inputs_len != 0 and neg_inputs_len != 0:

            if mutation_type == 0:  # feature-id swap
                input_type = random.randrange(0, 2)  # choose input type

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)  # choose input to swap
                    new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                    rule.pos_inputs.append(temp_features[new_mirna_id])  # add new mirna to positive inputs
                    del rule.pos_inputs[input_id]  # delete old miRNA

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)  # choose input to swap
                    new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                    rule.neg_inputs.append(temp_features[new_mirna_id])  # add new mirna to positive inputs
                    del rule.neg_inputs[input_id]  # delete old miRNA

            if mutation_type == 1:  # add/remove NOT
                input_type = random.randrange(0, 2)  # choose input type

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)  # choose input to add NOT
                    rule.neg_inputs.append(rule.pos_inputs[input_id])  # move input to negative inputs
                    del rule.pos_inputs[input_id]  # delete old positive input

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)  # choose input to remove NOT
                    rule.pos_inputs.append(rule.neg_inputs[input_id])  # move input to positive inputs
                    del rule.neg_inputs[input_id]  # delete old negative input
                    rule.gate = random.randint(0, 1)

            if mutation_type == 2:  # remove miRNA (cannot add miRNA as the maximal size is 2 inputs per rule)
                input_type = random.randrange(0, 2)  # choose input type

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)  # choose input to remove miRNA
                    del rule.pos_inputs[input_id]  # remove miRNA

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)  # choose input to remove miRNA
                    del rule.neg_inputs[input_id]  # remove miRNA

            mutated = True

    # if rule consists of only negative inputs
    if mutated is False:  # if rule is unchanged
        if pos_inputs_len == 0:  # if no positive inputs in the rule

            if mutation_type == 0:  # miRNA-id swap
                input_id = random.randrange(0, neg_inputs_len)  # choose input to swap
                new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                rule.neg_inputs.append(temp_features[new_mirna_id])  # add new miRNA to negative inputs
                del rule.neg_inputs[input_id]  # delete old miRNA

            if mutation_type == 1:  # remove NOT
                input_id = random.randrange(0, neg_inputs_len)  # choose input to remove NOT
                rule.pos_inputs.append(rule.neg_inputs[input_id])  # move input to positive inputs
                del rule.neg_inputs[input_id]  # delete old negative input

            if mutation_type == 2:  # add/remove miRNA

                if neg_inputs_len == 1:  # cannot remove miRNA as a rule cannot be empty
                    add_pos_or_neg = random.randrange(0, 2)  # choose which input to add

                    if add_pos_or_neg == 0:  # add positive input
                        new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                        rule.pos_inputs.append(temp_features[new_mirna_id])  # add new miRNA to positive

                    if add_pos_or_neg == 1:  # add negative input
                        new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                        rule.neg_inputs.append(temp_features[new_mirna_id])  # add new miRNA to positive

                if neg_inputs_len != 1:  # remove input (cannot add miRNAs)

                    input_id = random.randrange(0, neg_inputs_len)  # choose input to remove
                    del rule.neg_inputs[input_id]  # delete miRNA

            mutated = True

    # if rule consists of only positive inputs
    if mutated is False:  # if rule is unchanged
        if neg_inputs_len == 0:

            if mutation_type == 0:  # miRNA-id swap
                input_id = random.randrange(0, pos_inputs_len)  # choose input to swap
                new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                rule.pos_inputs.append(temp_features[new_mirna_id])  # add new miRNA to positive inputs
                del rule.pos_inputs[input_id]  # delete old miRNA

            if mutation_type == 1:  # add NOT
                input_id = random.randrange(0, pos_inputs_len)  # choose input to add NOT
                rule.neg_inputs.append(rule.pos_inputs[input_id])  # move input to negative inputs
                del rule.pos_inputs[input_id]  # delete old miRNA
                rule.gate = 1

            if mutation_type == 2:  # add/remove miRNA

                if pos_inputs_len == 1:  # cannot remove miRNA as a rule cannot be empty
                    add_pos_or_neg = random.randrange(0, 2)   # choose which input to add

                    if add_pos_or_neg == 0:  # add positive input
                        new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                        rule.pos_inputs.append(temp_features[new_mirna_id])  # add new miRNA to positive inputs
                        rule.gate = random.randint(0, 1)  # choose gate
                    if add_pos_or_neg == 1:  # add negative input
                        new_mirna_id = random.randrange(0, len(temp_features))  # choose new miRNA
                        rule.neg_inputs.append(temp_features[new_mirna_id])  # add new miRNA to negative inputs

                if pos_inputs_len != 1:  # remove input (cannot add miRNAs)
                    input_id = random.randrange(0, pos_inputs_len)  # choose input to remove
                    del rule.pos_inputs[input_id]  # delete miRNA
                    rule.gate = 1

            mutated = True


# mutate classifier
def mutate_classifier(population, classifier, features, evaluation_threshold):

    if evaluation_threshold is None:
        mutation_type = random.randrange(0, 3)  # choose mutation type
    else:
        mutation_type = random.randrange(0, 4)  # choose mutation type

    # copy rule from one classifier to another
    if mutation_type == 0:
        if len(classifier.rule_set) <= 4:  # if classifier is not too big
            classifier_to_copy = random.randrange(0, len(population))  # choose from which classifier to copy
            rule_to_copy = random.randrange(0, len(population[classifier_to_copy].rule_set))  # choose rule to copy
            classifier.rule_set.append(population[classifier_to_copy].rule_set[rule_to_copy].__copy__())  # copy rule

    # create temporary list of miRNAs
    temp_features = features.copy()

    # remove from miRNAs miRNAs existing in the classifier
    for rule in classifier.rule_set:
        for input in rule.pos_inputs:
            if input in temp_features:
                temp_features.remove(input)
        for input in rule.neg_inputs:
            if input in temp_features:
                temp_features.remove(input)

    # check how many features left
    if len(features) < 10 and len(temp_features) <= 3:
        temp_features = features.copy()

    # add rule
    if mutation_type == 1:
        if len(classifier.rule_set) <= 4:  # if classifier is not too big
            rule_to_add, temp_features = popinit.initialize_single_rule(temp_features)  # initialize new rule
            classifier.rule_set.append(rule_to_add.__copy__())  # add new rule to the classifier

    # remove rule
    if mutation_type == 2:
        if len(classifier.rule_set) >= 2:  # if classifier is not too small
            rule_to_remove = random.randrange(0, len(classifier.rule_set))  # choose rule to remove
            del classifier.rule_set[rule_to_remove]  # remove rule

    # mutate threshold
    if mutation_type == 3:
        thresholds = [0.25, 0.45, 0.50, 0.75, 1.0]
        thresholds.remove(classifier.evaluation_threshold)
        temp_evaluation_threshold = random.choice(thresholds)
        classifier.evaluation_threshold = temp_evaluation_threshold


# mutation
def mutate(population, features, mutation_probability, evaluation_threshold):

    # mutation for the population
    for classifier in population:
        mutation_rand = random.random()  # draw random number

        if mutation_rand <= mutation_probability:  # compare with mutation probability
            mutate_rule_or_classifier = random.randrange(0, 5)  # choose object to mutate: classifier or rule

            if mutate_rule_or_classifier == 0:  # mutate classifier
                mutate_classifier(population, classifier, features, evaluation_threshold)
            else:  # mutate rule
                rule_id = random.randrange(0, len(classifier.rule_set))  # choose rule for mutation
                rule = classifier.rule_set[rule_id]
                mutate_rule(rule, features)

    return population
