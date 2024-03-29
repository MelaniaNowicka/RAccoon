from decimal import *
import random
import pandas
import sys

random.seed(1)


# single boolean function class
# inputs are connected with AND
class SingleRule:

    """

    Class representing a single rule.

    Attributes
    ----------
    pos_inputs : list
        list of positive inputs
    neg_inputs : list
        list of negative inputs
    gate : int
        0 - OR, 1 - AND

    Methods
    -------

    """

    def __init__(self, pos_inputs, neg_inputs, gate):
        self.pos_inputs = pos_inputs  # non-negated inputs
        self.neg_inputs = neg_inputs  # negated inputs
        self.gate = gate  # gate (0 - OR, 1 - AND)

    # copy object
    def __copy__(self):
        new_pos_inputs = []
        new_neg_inputs = []

        # copy positive inputs
        for inp in self.pos_inputs:
            new_pos_inputs.append(inp)

        # copy negative inputs
        for inp in self.neg_inputs:
            new_neg_inputs.append(inp)

        return SingleRule(new_pos_inputs, new_neg_inputs, self.gate)

    def to_string(self):

        rule_str = ""

        for inp in sorted(self.pos_inputs):
            rule_str = rule_str + "(" + inp + ")"
        for inp in sorted(self.neg_inputs):
            rule_str = rule_str + "(NOT " + inp + ")"

        rule_str = " [" + rule_str + "] "

        if self.gate == 0:
            rule_str = rule_str.replace(")(", ") OR (")
        elif self.gate == 1:
            rule_str = rule_str.replace(")(", ") AND (")

        return rule_str


# classifier (individual)
class Classifier:

    """

    Class representing a single classifier.

    Attributes
    ----------
    rule_set : list
        list of SingleRule objects
    evaluation_threshold : float
        classifier evaluation threshold
    errors : dict
        dictionary of errors (tp, tn, fp and fn)
    error_rates : dict
        dictionary of error rates (tpr, tnr, fpr and fnr)
    score : float
        classifier score
    bacc : float
        classifier balanced accuracy
    cdd_score : float
        classifier cdd score
    additional_scores : dict
        dictionary of additional scores (f1, mcc, ppv and fdr)

    """

    def __init__(self, rule_set, evaluation_threshold, theta, errors,
                 error_rates, score, bacc, cdd_score, additional_scores):
        self.rule_set = rule_set  # list of rules
        self.evaluation_threshold = evaluation_threshold  # evaluation threshold alpha
        self.theta = theta  # threshold in number of rules
        self.errors = errors  # dictionary of error (tp, tn, fp, fn)
        self.error_rates = error_rates  # dictionary of error rates (tpr, tnr, fpr, fnr)
        self.score = score  # classifier score (may be bacc, may be other)
        self.bacc = bacc  # balanced accuracy
        self.cdd_score = cdd_score  # classifier class distribution diversity score
        self.additional_scores = additional_scores  # dictionary of other scores (f1, mcc, precision, fdr)

    # copy object
    def __copy__(self):
        new_rule_set = []

        # copy rules
        for rule in self.rule_set:
            new_rule_set.append(rule.__copy__())

        return Classifier(new_rule_set, self.evaluation_threshold, self.theta, self.errors, self.error_rates,
                          self.score, self.bacc, self.cdd_score, self.additional_scores)

    # updates theta
    def update_theta(self):
        self.theta = int(Decimal(len(self.rule_set)*float(self.evaluation_threshold)).
                         to_integral_value(rounding=ROUND_HALF_UP))

    # get a list of inputs
    def get_input_list(self):

        """
        creates list of inputs in a classifier

        Returns
        _______
        inputs : list
            list of classifier inputs

        """

        inputs = []
        for rule in self.rule_set:
            for inp in rule.pos_inputs:  # add positive inputs
                inputs.append(inp)

            for inp in rule.neg_inputs:  # add negative inputs
                inputs.append(inp)

        return inputs

    # remove repeating rules in classifiers
    def remove_duplicates(self):

        """

        removes duplicate rules in a classifier

        """

        single_pos = []  # list of positive single-input rules and their ids
        single_neg = []  # list of negative single-input rules and their ids
        multi_input = []  # list of mixed rules with size > 1 and their ids

        to_del = []  # rules to be deleted

        # separating by rule type
        for i in range(0, len(self.rule_set)):
            # first, check the rule feasibility
            if not check_rule_feasibility(self.rule_set[i]):
                raise ValueError(f'Unfeasible rule found: {self.rule_set[i].to_string()}')
            # collecting negative single-input rules
            if len(self.rule_set[i].pos_inputs) == 0 and len(self.rule_set[i].neg_inputs) == 1:
                single_neg.append((self.rule_set[i], i))
            # collecting positive single-input rules
            elif len(self.rule_set[i].pos_inputs) == 1 and len(self.rule_set[i].neg_inputs) == 0:
                single_pos.append((self.rule_set[i], i))
            # collecting mixed rules
            else:
                multi_input.append((self.rule_set[i], i))

        # collecting ids of rules for removal
        # for positive single-input rules
        if len(single_pos) > 1:  # if more than one rule exists
            for i in range(0, len(single_pos)-1):  # iterate over rules
                for j in range(i+1, len(single_pos)):
                    if single_pos[i][0].pos_inputs == single_pos[j][0].pos_inputs:  # if inputs are identical
                        to_del.append(single_pos[j][1])  # add rule's id  for removal

        # for negative single-input rules
        if len(single_neg) > 1:  # if more than one rule exists
            for i in range(0, len(single_neg)-1):  # iterate over rules
                for j in range(i+1, len(single_neg)):
                    if single_neg[i][0].neg_inputs == single_neg[j][0].neg_inputs:  # if inputs are identical
                        to_del.append(single_neg[j][1])  # add rule's id for removal

        # for mixed rules
        if len(multi_input) > 1:  # if more than one rule exists
            for i in range(0, len(multi_input)-1):  # iterate over rules
                for j in range(i+1, len(multi_input)):

                    # for rules with two positive inputs
                    if len(multi_input[i][0].pos_inputs) == 2 and len(multi_input[j][0].pos_inputs) == 2:
                        pos_list1 = sorted(multi_input[i][0].pos_inputs)
                        pos_list2 = sorted(multi_input[j][0].pos_inputs)
                        # if inputs and gates are identical
                        if pos_list1 == pos_list2 and multi_input[i][0].gate == multi_input[j][0].gate:
                            to_del.append(multi_input[j][1])  # add rule's id for removal

                    # for rules with two negative inputs
                    elif len(multi_input[i][0].neg_inputs) == 2 and len(multi_input[j][0].neg_inputs) == 2:
                        neg_list1 = sorted(multi_input[i][0].neg_inputs)
                        neg_list2 = sorted(multi_input[j][0].neg_inputs)
                        # if inputs and gates are identical
                        if neg_list1 == neg_list2 and multi_input[i][0].gate == multi_input[j][0].gate:
                            to_del.append(multi_input[j][1])  # add rule's id for removal

                    # for rules with mixed inputs
                    elif len(multi_input[i][0].pos_inputs) == 1 and len(multi_input[i][0].neg_inputs) == 1:
                        if len(multi_input[j][0].pos_inputs) == 1 and len(multi_input[j][0].neg_inputs) == 1:
                            if multi_input[i][0].pos_inputs == multi_input[j][0].pos_inputs:  # if inputs are identical
                                if multi_input[i][0].neg_inputs == multi_input[j][0].neg_inputs:
                                    if multi_input[i][0].gate == multi_input[j][0].gate:
                                        to_del.append(multi_input[j][1])  # add rule's id for removal

        # sort indices in descending order for removal
        to_del = list(set(to_del))
        to_del.sort(reverse=True)
        # remove duplicates
        for ind in to_del:
            del self.rule_set[ind]


def check_rule_feasibility(rule):

    # check if a 1-input rule has an AND assigned as in initialize_single_rule()
    if (len(rule.pos_inputs) + len(rule.neg_inputs)) == 1 and rule.gate == 0:
        return False
    # check if rule is not empty
    elif (len(rule.pos_inputs) + len(rule.neg_inputs)) == 0:
        return False
    # check if a 2-input rule with at least one negative input has AND assigned as in biological restrictions
    # Note, OR cannot have negative inputs assigned!
    elif len(rule.neg_inputs) >= 1 and rule.gate == 0:
        return False
    # check if rule has at most 2 inputs
    elif (len(rule.pos_inputs) + len(rule.neg_inputs)) > 2:
        return False
    else:
        return True


# initialization of a single rule
# temp_features consists of features that were not used in the classifier yet
def initialize_single_rule(temp_features):

    """

    Initializes random single rule.

    Parameters
    ----------
    temp_features : list
        temporary list of features

    Returns
    -------
    single_rule : SingleRule objects
        single rule
    temp_features : list
        updated temporary list of features

    """

    # size of a single rule (between 1 and 2 inputs)
    size = random.randint(1, 2)

    if size == 2:  # if size is 2
        gate = random.randint(0, 1)  # randomly choose the assigned gate: 0 - OR, 1 - AND
    else:  # otherwise assign AND (1)
        gate = 1

    pos_inputs = []  # list of positive inputs
    neg_inputs = []  # list of negative inputs

    for i in range(0, size):  # drawing features for a rule (without replacement) respecting size

        if gate == 1:  # if AND gate - the input signs may differ
            input_sign = random.randint(0, 1)  # randomly choosing input sign (0 - negative, 1 - positive)
        else:  # if OR gate - only positive inputs may be assigned
            input_sign = 1

        feature_id = random.randrange(0, len(temp_features))  # randomly choosing feature ID

        # checking the input sign to assign inputs to a positive or negative group
        if input_sign == 0:
            neg_inputs.append(temp_features[feature_id])

        if input_sign == 1:
            pos_inputs.append(temp_features[feature_id])

        # removal of used features (rule must consist of i=unique features IDs)
        del temp_features[feature_id]

    # initialization of new single rule
    single_rule = SingleRule(pos_inputs, neg_inputs, gate)

    return single_rule, temp_features


# initialization of a new classifier
def initialize_classifier(classifier_size, evaluation_threshold, features):

    """

    Initializes random classifier.

    Parameters
    ----------
    classifier_size : int
        maximal classifier size
    evaluation_threshold : float
        classifier evaluation threshold
    features : list
        list of features


    Returns
    -------
    classifier : Classifier object
        classifier

    """

    # size of a classifier
    size = random.randint(1, classifier_size)

    # rules
    rule_set = []

    # copy of feature list
    temp_features = features.copy()

    # initialization of new rules
    for i in range(0, size):
        if len(features) < 10 and len(temp_features) <= 3:  # if there is not enough features to choose from
            temp_features = features.copy()

        rule, temp_features = initialize_single_rule(temp_features)  # initialize single rule and remove used features
        rule_set.append(rule)  # add rule to rule set

    # if no threshold is set
    if evaluation_threshold is None:
        thresholds = [0.25, 0.45, 0.50, 0.75, 1.0]
        temp_evaluation_threshold = random.choice(thresholds)  # randomly choose threshold

    # initialization of a new classifier
    classifier = Classifier(rule_set, evaluation_threshold=temp_evaluation_threshold, theta=0, errors={},
                            error_rates={}, score=0.0, bacc=0.0, additional_scores={}, cdd_score=0.0)

    return classifier


# population initialization
def initialize_population(population_size,
                          features,
                          evaluation_threshold,
                          classifier_size):

    """

    Initializes random population.

    Parameters
    ----------
    population_size : int
        population size
    features : list
        list of features
    evaluation_threshold : float
        classifier evaluation threshold
    classifier_size : int
        maximal classifier size

    Returns
    -------
    population : list
        list of Classifier objects

    """

    population = []  # empty population

    # initialization of n=population_size classifiers
    for i in range(0, population_size):
        classifier = initialize_classifier(classifier_size, evaluation_threshold, features)
        population.append(classifier)

    return population


# read rules from file
def read_rules_from_file(rule_file):

    """

    Reads pre-optimized rules from a file.

    Parameters
    ----------
    rule_file : str
        path to a file including pre-optimized rules

    Returns
    -------
    rules : list
        list of rules

    """

    # reading the data
    # throws an exception when datafile not found
    try:
        data = pandas.read_csv(rule_file, sep=';', header=0)
    except IOError:
        print("Error: No such file or directory.")
        sys.exit(0)

    rules = []  # list of rules

    for i in range(0, len(data.index)):
        new_rule = SingleRule([], [], 1)  # create empty rule
        rule = data.iloc[i]  # get rule from data

        # check rule's size
        if rule["size"] == 1:  # 1-input rules
            if rule["sign1"] == 0:  # check sign
                new_rule.neg_inputs.append(rule["feature1"])
            else:
                new_rule.pos_inputs.append(rule["feature1"])

        if rule["size"] == 2:  # 2-input rules
            if rule["sign1"] == 0:
                new_rule.neg_inputs.append(rule["feature1"])
            else:
                new_rule.pos_inputs.append(rule["feature1"])

            if rule["sign2"] == 0:
                new_rule.neg_inputs.append(rule["feature2"])
            else:
                new_rule.pos_inputs.append(rule["feature2"])

        if len(new_rule.pos_inputs) == 2:
            new_rule.gate = 0

        rules.append(new_rule)  # add rule to the list of rules

    return rules


# create population based on list of pre-optimized rules
def initialize_population_from_rules(population_size, features, evaluation_threshold, rule_list, popt_fraction,
                                     classifier_size):

    """

    Initializes a fraction of population based on pre-optimized rules (the rest is initialized randomly).

    Parameters
    ----------
    population_size : int
        population size
    features : list
        list of features
    evaluation_threshold : float
        classifier evaluation threshold
    rule_list : list
        list of pre-optimized rules
    popt_fraction : float
        fraction of population initialized with pre-optimized rules
    classifier_size : int
        maximal classifier size

    Returns
    -------
    population : list
        list of Classifier objects

    """

    population = []  # create empty population

    # initialization of population_size*fraction individuals built from pre-optimized rules
    fraction = int(population_size*popt_fraction)
    for i in range(0, fraction):
        # size of a classifier
        size = random.randint(1, classifier_size)
        rule_set = []
        for j in range(0, size):
            new_rule = random.randrange(0, len(rule_list))
            rule_set.append(rule_list[new_rule])

        # if no threshold is set
        if evaluation_threshold is None:
            thresholds = [0.25, 0.45, 0.50, 0.75, 1.0]
            temp_evaluation_threshold = random.choice(thresholds)  # randomly choose threshold
        else:
            temp_evaluation_threshold = evaluation_threshold

        # create a classifier
        classifier = Classifier(rule_set, evaluation_threshold=temp_evaluation_threshold, theta=0,
                                errors={}, error_rates={}, score={}, bacc={}, additional_scores={}, cdd_score={})

        population.append(classifier)  # add classifier to the population

    # initialization of random individuals
    for i in range(0, population_size - fraction):
        classifier = initialize_classifier(classifier_size, evaluation_threshold, features)
        population.append(classifier)

    return population
