import random
import popinit

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
    first_child = popinit.Classifier([], {}, 0.0)
    second_child = popinit.Classifier([], {}, 0.0)

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