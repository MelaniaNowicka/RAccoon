import popinit
import random


# crossover
def crossover(first_parent, second_parent):

    # checking sizes of parents and assigning rule sets
    # first parent - consists of more rules, second parents - less
    # if equal - assign first to first, second to second
    if len(first_parent.rule_set) < len(second_parent.rule_set):
        first_parent_rule_set = second_parent.rule_set
        second_parent_rule_set = first_parent.rule_set
    else:
        first_parent_rule_set = first_parent.rule_set
        second_parent_rule_set = second_parent.rule_set

    # creating empty offspring
    first_child = popinit.Classifier([], {}, {}, 0.0, 0.0, 0.0, {})
    second_child = popinit.Classifier([], {}, {}, 0.0, 0.0, 0.0, {})

    # if the first parent consists of more rules
    if len(first_parent_rule_set) > len(second_parent_rule_set):
        # difference between sizes of parents
        difference = len(first_parent_rule_set) - len(second_parent_rule_set)
        # crossover index specifies the position of the second parent in relation to the first one
        # e.g. (index = 1)
        # first parent:   [ ] [ ] [ ] [ ] [ ]
        # second parent:      [ ] [ ]
        crossover_index_second_parent = random.randrange(0, difference+1)

        for i in range(0, len(first_parent_rule_set)):  # iterating through the first parent
            swap_mask = random.randrange(0, 2)  # randomly choosing the mask
            if swap_mask == 1:  # if mask=1 swap elements

                # check the position of the second classifier
                # if the parents are not aligned in i move element i from the first parent to the second child
                if i < crossover_index_second_parent or \
                        i >= crossover_index_second_parent + len(second_parent_rule_set):
                    # move element i to the second child
                    second_child.rule_set.append(first_parent_rule_set[i].__copy__())
                else:  # else, swap elements
                    second_child.rule_set.append(first_parent_rule_set[i].__copy__())
                    first_child.rule_set.append(second_parent_rule_set[i-crossover_index_second_parent].__copy__())
            else:  # if mask=0 do not swap elements and copy elements from parents to offspring if possible
                if i < crossover_index_second_parent or \
                        i >= crossover_index_second_parent + len(second_parent_rule_set):
                    first_child.rule_set.append(first_parent_rule_set[i].__copy__())
                else:
                    second_child.rule_set.append(second_parent_rule_set[i-crossover_index_second_parent].__copy__())
                    first_child.rule_set.append(first_parent_rule_set[i].__copy__())

    else:  # if parents have the same length
        for i in range(0, len(first_parent_rule_set)):  # iterate over first parent
            swap_mask = random.randrange(0, 2)
            if swap_mask == 1:  # swap
                first_child.rule_set.append(second_parent_rule_set[i].__copy__())
                second_child.rule_set.append(first_parent_rule_set[i].__copy__())
            else:  # do not swap, just copy
                first_child.rule_set.append(first_parent_rule_set[i].__copy__())
                second_child.rule_set.append(second_parent_rule_set[i].__copy__())

    return first_child.__copy__(), second_child.__copy__()
