import popinit
import random

random.seed(1)


# mutate single rule
def mutate_rule(rule, mirnas):

    mutation_type = random.randrange(0, 3)  # choose mutation type
    temp_mirnas = mirnas.copy()  # copy mirnas to a temporary list

    # remove inputs that are in the rule
    for input in rule.pos_inputs:
        temp_mirnas.remove(input)

    for input in rule.neg_inputs:
        temp_mirnas.remove(input)

    mutated = False  # rule is unchanged

    pos_inputs_len = len(rule.pos_inputs)  # number of positive inputs in a rule
    neg_inputs_len = len(rule.neg_inputs)  # number of negative inputs in a rule

    # if rule consists of positive and negative inputs
    if mutated is False:  # if rule is unchanged
        if pos_inputs_len != 0 and neg_inputs_len != 0:

            if mutation_type == 0:  # miRNA-id swap
                input_type = random.randrange(0, 2)  # choose input type

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)  # choose input to swap
                    new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                    rule.pos_inputs.append(temp_mirnas[new_mirna_id])  # add new mirna to positive inputs
                    del rule.pos_inputs[input_id]  # delete old miRNA

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)  # choose input to swap
                    new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                    rule.neg_inputs.append(temp_mirnas[new_mirna_id])  # add new mirna to positive inputs
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
                new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                rule.neg_inputs.append(temp_mirnas[new_mirna_id])  # add new miRNA to negative inputs
                del rule.neg_inputs[input_id]  # delete old miRNA

            if mutation_type == 1:  # remove NOT
                input_id = random.randrange(0, neg_inputs_len)  # choose input to remove NOT
                rule.pos_inputs.append(rule.neg_inputs[input_id])  # move input to positive inputs
                del rule.neg_inputs[input_id]  # delete old negative input

            if mutation_type == 2:  # add/remove miRNA

                if neg_inputs_len == 1:  # cannot remove miRNA as a rule cannot be empty
                    add_pos_or_neg = random.randrange(0, 2)  # choose which input to add

                    if add_pos_or_neg == 0:  # add positive input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                        rule.pos_inputs.append(temp_mirnas[new_mirna_id])  # add new miRNA to positive

                    if add_pos_or_neg == 1:  # add negative input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                        rule.neg_inputs.append(temp_mirnas[new_mirna_id])  # add new miRNA to positive

                if neg_inputs_len != 1:  # remove input (cannot add miRNAs)

                    input_id = random.randrange(0, neg_inputs_len)  # choose input to remove
                    del rule.neg_inputs[input_id]  # delete miRNA

            mutated = True

    # if rule consists of only positive inputs
    if mutated is False:  # if rule is unchanged
        if neg_inputs_len == 0:

            if mutation_type == 0:  # miRNA-id swap
                input_id = random.randrange(0, pos_inputs_len)  # choose input to swap
                new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                rule.pos_inputs.append(temp_mirnas[new_mirna_id])  # add new miRNA to positive inputs
                del rule.pos_inputs[input_id]  # delete old miRNA

            if mutation_type == 1:  # add NOT
                input_id = random.randrange(0, pos_inputs_len)  # choose input to add NOT
                rule.neg_inputs.append(rule.pos_inputs[input_id])  # move input to negative inputs
                del rule.pos_inputs[input_id]  # delete old miRNA

            if mutation_type == 2:  # add/remove miRNA

                if pos_inputs_len == 1:  # cannot remove miRNA as a rule cannot be empty
                    add_pos_or_neg = random.randrange(0, 2)   # choose which input to add

                    if add_pos_or_neg == 0:  # add positive input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                        rule.pos_inputs.append(temp_mirnas[new_mirna_id])  # add new miRNA to positive inputs
                    if add_pos_or_neg == 1:  # add negative input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))  # choose new miRNA
                        rule.neg_inputs.append(temp_mirnas[new_mirna_id])  # add new miRNA to negative inputs

                if pos_inputs_len != 1:  # remove input (cannot add miRNAs)

                    input_id = random.randrange(0, pos_inputs_len)  # choose input to remove
                    del rule.pos_inputs[input_id]  # delete miRNA

            mutated = True


# mutate classifier
def mutate_classifier(population, classifier, mirnas):

    mutation_type = random.randrange(0, 4)

    # copy rule from one classifier to another
    if mutation_type == 0:
        if len(classifier.rule_set) <= 4:  # if classifier is not too big
            classifier_to_copy = random.randrange(0, len(population))  # choose from which classifier to copy
            rule_to_copy = random.randrange(0, len(population[classifier_to_copy].rule_set))  # choose rule to copy
            classifier.rule_set.append(population[classifier_to_copy].rule_set[rule_to_copy].__copy__())  # copy rule

    # create temporary list of miRNAs
    temp_mirnas = mirnas.copy()

    # remove from miRNAs miRNAs existing in the classifier
    for rule in classifier.rule_set:
        for input in rule.pos_inputs:
            if input in temp_mirnas:
                temp_mirnas.remove(input)
        for input in rule.neg_inputs:
            if input in temp_mirnas:
                temp_mirnas.remove(input)

    if len(mirnas) < 10 and len(temp_mirnas) <= 3:
        temp_mirnas = mirnas.copy()

    # add rule
    if mutation_type == 1:
        if len(classifier.rule_set) <= 4:  # if classifier is not too big
            rule_to_add, temp_mirnas = popinit.initialize_single_rule(temp_mirnas)  # initialize new rule
            classifier.rule_set.append(rule_to_add.__copy__())  # add new rule to the classifier

    # remove rule
    if mutation_type == 2:
        if len(classifier.rule_set) >= 2:  # if classifier is not too small
            rule_to_remove = random.randrange(0, len(classifier.rule_set))  # choose rule to remove
            del classifier.rule_set[rule_to_remove]  # remove rule

    if mutation_type == 3:
        thresholds = [0.25, 0.45, 0.50, 0.75, 1.0]
        thresholds.remove(classifier.evaluation_threshold)
        evaluation_threshold = random.choice(thresholds)


# mutation
def mutate(population, mirnas, mutation_probability):

    # mutation for the population
    for classifier in population:
        mutation_rand = random.random()
        if mutation_rand <= mutation_probability:
            mutate_rule_or_classifier = random.randrange(0, 5)  # choose object to mutate: classifier or rule

            if mutate_rule_or_classifier == 0:  # mutate classifier
                mutate_classifier(population, classifier, mirnas)
            else: # mutate rule
                rule_id = random.randrange(0, len(classifier.rule_set))  # choose rule for mutation
                rule = classifier.rule_set[rule_id]
                mutate_rule(rule, mirnas)

    return population
