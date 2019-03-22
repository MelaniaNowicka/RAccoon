import random

def mutate_rule(rule, mirnas):

    mutation_type = random.randrange(0, 3)  # choose mutation type
    temp_mirnas = mirnas.copy()

    # remove inputs that are in the rule
    for input in rule.pos_inputs:
        temp_mirnas.remove(input)

    for input in rule.neg_inputs:
        temp_mirnas.remove(input)

    mutated = False

    pos_inputs_len = len(rule.pos_inputs)
    neg_inputs_len = len(rule.neg_inputs)

    # if rule consists of positive and negative inputs
    if mutated is False:
        if pos_inputs_len != 0 and neg_inputs_len != 0:

            if mutation_type == 0:  # miRNA-id swap
                input_type = random.randrange(0, 2)  # choose input type

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)
                    new_mirna_id = random.randrange(0, len(temp_mirnas))
                    rule.pos_inputs.append(temp_mirnas[new_mirna_id])
                    del rule.pos_inputs[input_id]

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)
                    new_mirna_id = random.randrange(0, len(temp_mirnas))
                    rule.neg_inputs.append(temp_mirnas[new_mirna_id])
                    del rule.neg_inputs[input_id]

            if mutation_type == 1:  # add/remove NOT
                input_type = random.randrange(0, 2)

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)
                    rule.neg_inputs.append(rule.pos_inputs[input_id])
                    del rule.pos_inputs[input_id]

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)
                    rule.pos_inputs.append(rule.neg_inputs[input_id])
                    del rule.neg_inputs[input_id]

            if mutation_type == 2:  # remove or add miRNA
                input_type = random.randrange(0, 2)

                if input_type == 0:  # positive inputs
                    input_id = random.randrange(0, pos_inputs_len)
                    del rule.pos_inputs[input_id]

                if input_type == 1:  # negative inputs
                    input_id = random.randrange(0, neg_inputs_len)
                    del rule.neg_inputs[input_id]

            mutated = True

    # if rule consists of only negative inputs
    if mutated is False:
        if pos_inputs_len == 0:

            if mutation_type == 0:  # miRNA-id swap
                input_id = random.randrange(0, neg_inputs_len)
                new_mirna_id = random.randrange(0, len(temp_mirnas))
                rule.neg_inputs.append(temp_mirnas[new_mirna_id])
                del rule.neg_inputs[input_id]

            if mutation_type == 1:  # add/remove NOT
                input_id = random.randrange(0, neg_inputs_len)
                rule.pos_inputs.append(rule.neg_inputs[input_id])
                del rule.neg_inputs[input_id]

            if mutation_type == 2:  # remove or add miRNA

                if neg_inputs_len == 1:  # cannot remove miRNA as a rule cannot be empty

                    add_pos_or_neg = random.randrange(0, 2)
                    if add_pos_or_neg == 0:  # add positive input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))
                        rule.pos_inputs.append(temp_mirnas[new_mirna_id])
                    if add_pos_or_neg == 1:  # add negative input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))
                        rule.neg_inputs.append(temp_mirnas[new_mirna_id])

                if neg_inputs_len != 1:  # remove input

                    input_id = random.randrange(0, neg_inputs_len)
                    del rule.neg_inputs[input_id]

            mutated = True

    # if rule consists of only positive inputs
    if mutated is False:
        if neg_inputs_len == 0:

            if mutation_type == 0:  # miRNA-id swap
                input_id = random.randrange(0, pos_inputs_len)
                new_mirna_id = random.randrange(0, len(temp_mirnas))
                rule.pos_inputs.append(temp_mirnas[new_mirna_id])
                del rule.pos_inputs[input_id]

            if mutation_type == 1:  # add/remove NOT
                input_id = random.randrange(0, pos_inputs_len)
                rule.neg_inputs.append(rule.pos_inputs[input_id])
                del rule.pos_inputs[input_id]

            if mutation_type == 2:  # remove or add miRNA

                if pos_inputs_len == 1:  # cannot remove miRNA as a rule cannot be empty
                    add_pos_or_neg = random.randrange(0, 2)

                    if add_pos_or_neg == 0:  # add positive input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))
                        rule.pos_inputs.append(temp_mirnas[new_mirna_id])
                    if add_pos_or_neg == 1:  # add negative input
                        new_mirna_id = random.randrange(0, len(temp_mirnas))
                        rule.neg_inputs.append(temp_mirnas[new_mirna_id])

                if pos_inputs_len != 1:  # remove input

                    input_id = random.randrange(0, pos_inputs_len)
                    del rule.pos_inputs[input_id]

            mutated = True

# mutation
def mutate(population, mirnas, mutation_probability):

    # mutation for the population
    for classifier in population:
        mutation_rand = random.random()
        if mutation_rand <= mutation_probability:
            rule_id = random.randrange(0, len(classifier.rule_set))  # choose rule for mutation
            rule = classifier.rule_set[rule_id]
            mutate_rule(rule, mirnas)

    return population