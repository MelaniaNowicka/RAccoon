import popinit
import random

random.seed(1)


# uniform crossover
def uniform_crossover(first_parent_rule_set, second_parent_rule_set, first_child, second_child):

    """

    Performs uniform crossover on two parents and returns two children.

    Parameters
    ----------
    first_parent_rule_set : list
        list of rules (SingleRule objects) of first parent
    second_parent_rule_set : list
        list of rules (SingleRule objects) of second parent
    first_child : Classifier object
        empty first child classifier
    second_child : Classifier object
        empty second child classifier

    Returns
    -------
    first_child : Classifier object
        first child classifier after crossover
    second_child : Classifier object
        second child classifier after crossover

    """

    for i in range(0, len(first_parent_rule_set)):  # iterate over first parent
        swap_mask = random.randint(0, 1)
        if swap_mask == 1:  # swap
            first_child.rule_set.append(second_parent_rule_set[i].__copy__())
            second_child.rule_set.append(first_parent_rule_set[i].__copy__())
        else:  # do not swap, just copy
            first_child.rule_set.append(first_parent_rule_set[i].__copy__())
            second_child.rule_set.append(second_parent_rule_set[i].__copy__())

    return first_child.__copy__(), second_child.__copy__()


# index-based crossover
# crossover index specifies the position of the second parent in relation to the first one
# e.g. (index = 1)
# first parent:   [ ] [ ] [ ] [ ] [ ]
# second parent:      [ ] [ ]
def index_based_crossover(first_parent_rule_set, second_parent_rule_set, first_child, second_child):

    """

    Performs index-based crossover on two parents and returns two children.

    Crossover index specifies the position of the second parent in relation to the first one and performs uniform
    crossover on overlapping rules. Other rules are randomly added to either the first or second child e.g. index = 1:

    first parent:   [ ] [ 1 ] [ ] [ ] [ ]

    second parent:      [ x ] [ ]

    Parameters
    ----------
    first_parent_rule_set : list
        list of rules (SingleRule objects) of first parent
    second_parent_rule_set : list
        list of rules (SingleRule objects) of second parent
    first_child : Classifier object
        empty first child classifier
    second_child : Classifier object
        empty second child classifier

    Returns
    -------
    first_child : Classifier object
        first child classifier after crossover
    second_child : Classifier object
        second child classifier after crossover

    """

    # difference between sizes of parents
    difference = len(first_parent_rule_set) - len(second_parent_rule_set)
    # crossover index specifies the position of the second parent in relation to the first one
    crossover_index_second_parent = random.randint(0, difference)

    for i in range(0, len(first_parent_rule_set)):  # iterating through the first parent

        swap_mask = random.randint(0, 1)  # randomly choosing the mask
        if swap_mask == 1:  # if mask=1 swap elements
            # check the position of the second classifier
            # if the parents are not aligned in i move element i from the first parent to the second child
            if i < crossover_index_second_parent or \
                    i >= crossover_index_second_parent + len(second_parent_rule_set):
                # move element i to the second child
                second_child.rule_set.append(first_parent_rule_set[i].__copy__())
            else:  # else, swap elements
                first_child.rule_set.append(second_parent_rule_set[i - crossover_index_second_parent].__copy__())
                second_child.rule_set.append(first_parent_rule_set[i].__copy__())

        else:  # if mask=0 do not swap elements and copy elements from parents to offspring if possible
            if i < crossover_index_second_parent or \
                    i >= crossover_index_second_parent + len(second_parent_rule_set):
                first_child.rule_set.append(first_parent_rule_set[i].__copy__())
            else:
                first_child.rule_set.append(first_parent_rule_set[i].__copy__())
                second_child.rule_set.append(second_parent_rule_set[i - crossover_index_second_parent].__copy__())

    return first_child.__copy__(), second_child.__copy__()


# crossover
def crossover_parents(first_parent, second_parent):

    """

    Performs crossover (uniform or index-based) on two parents and returns two children.

    Parameters
    ----------
    first_parent : Classifier object
        first parent classifier
    second_parent : Classifier object
        second parent classifier

    Returns
    -------
    first_child : Classifier object
        first child classifier after crossover
    second_child : Classifier object
        second child classifier after crossover

    """

    # checking size of parents and assigning rule sets
    # first parent - consists of more rules, second parents - less
    # if equal - assign first to first, second to second
    first_parent_rule_set = []
    second_parent_rule_set = []
    if len(first_parent.rule_set) < len(second_parent.rule_set):
        first_parent_rule_set = second_parent.rule_set.copy()
        second_parent_rule_set = first_parent.rule_set.copy()
    else:
        first_parent_rule_set = first_parent.rule_set.copy()
        second_parent_rule_set = second_parent.rule_set.copy()

    # initialization of empty offspring
    first_child = popinit.Classifier([], first_parent.evaluation_threshold, {}, {}, 0.0, 0.0, 0.0, {})
    second_child = popinit.Classifier([], second_parent.evaluation_threshold, {}, {}, 0.0, 0.0, 0.0, {})

    # if the first parent consists of more rules
    if len(first_parent_rule_set) > len(second_parent_rule_set):
        first_child, second_child = \
            index_based_crossover(first_parent_rule_set, second_parent_rule_set, first_child, second_child)

    else:  # if parents have the same length
        first_child, second_child = \
            uniform_crossover(first_parent_rule_set, second_parent_rule_set, first_child, second_child)

    return first_child.__copy__(), second_child.__copy__()
