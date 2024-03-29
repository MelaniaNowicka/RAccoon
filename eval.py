from decimal import *
import preproc
import numpy
import math
import sys
import log


class BestSolutions:

    """

    Class representing best solutions.

    BestSolutions class allows to store best found solutions as strings and as Classifier objects as well as it's
    performance score.

    Attributes
    ----------
    score : float
        solution score (may be multi-objective score, balanced accuracy or cdd score)
    solutions : list
        list of solutions as Classifier objects
    solutions_str : list
        list of solutions as strings

    """

    def __init__(self, score, solutions, solutions_str):
        self.score = score
        self.solutions = solutions
        self.solutions_str = solutions_str


# compare floats - is close with 1e-5 tolerance
def is_close(x, y, tol=1e-5):

    """

    Compares floats - checks whether y is close to x with 1e-5 tolerance.

    Parameters
    ----------
    x : float
        first float
    y : float
        second float
    tol : float
        tolerance

    Returns
    -------
    bool
        True if y is close to x

    """

    if abs(x - y) <= tol:
        return True
    else:
        return False


# compare floats - is y is higher than x with 1e-5 tolerance
def is_higher(x, y, tol=1e-5):

    """

    Compares floats - checks whether y is higher than x with 1e-5 tolerance.

    Parameters
    ----------
    x : float
        first float
    y : float
        second float
    tol : float
        tolerance

    Returns
    -------
    bool
        True if y is higher than x

    """

    if y - x >= tol:
        return True
    else:
        return False


# calculate balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    """

    Calculates balanced accuracy (bacc = (tp/p + tn/n)/2).

    Parameters
    ----------
    tp : int
        number of true positives
    tn : int
        number of true negatives
    p : int
        number of positives
    n : int
        number of negatives

    Returns
    -------
    float
        balanced accuracy

    """

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy


# calculate classifier cdd score
# class distribution diversity score
def calculate_cdd_score(inputs, feature_cdds, uniqueness):

    """

    Calculates CDD (Class Diversity Diversity) score.

    Parameters
    ----------
    inputs : list
        list of classifier inputs
    feature_cdds : dict
        dict of feature cdd scores
    uniqueness : bool
        if True only unique inputs in a classifier are counted, otherwise the input cdd score is multiplied by
        the number of input occurrences

    Returns
    -------
    float
        classifier cdd score

    """

    # sum up cdds for each miRNA in the classifier
    classifier_cdd_sum = 0

    if uniqueness is True:
        inputs = set(inputs)

    for i in inputs:
        try:
            classifier_cdd_sum += feature_cdds.get(i)
        except TypeError:
            print("Error: cdd score. Feature:", i, " - score not found.")
            sys.exit(0)

    # calculate cdd score
    try:
        classifier_cdd_score = classifier_cdd_sum / len(inputs)
    except ZeroDivisionError:
        print("Error: cdd score - division by zero! No inputs in the classifier.")
        sys.exit(0)

    return classifier_cdd_score


# calculate multi-objective score
# balanced accuracy and cdd score
def calculate_multi_objective_score(bacc, cdd_score, bacc_weight):

    """

    Calculates multi-objective classifier score (classifier_score = bacc * bacc_weight + cdd_score * (1-bacc_weight)).

    Parameters
    ----------
    bacc : float
        classifier balanced accuracy
    cdd_score : float
        classifier cdd score
    bacc_weight : float
        weight of balanced accuracy in the multi-objective score

    Returns
    -------
    float
        classifier multi objective score

    """

    classifier_score = bacc * bacc_weight + cdd_score * (1-bacc_weight)

    return classifier_score


# calculate error rates
def calculate_error_rates(tp, tn, p, n):

    """

    Calculates error rates (true positive rate, true negative rate, false positive rate and false negative rate).

    Parameters
    ----------
    tp : int
        number of true positives
    tn : int
        number of true negatives
    p : int
        number of positives
    n : int
        number of negatives

    Returns
    -------
    dict
        dictionary of error rates (tpr, tnr, fpr and fnr)

    """

    try:
        tpr = tp/p  # true positive rate
        tnr = tn/n  # true negative rate
        fpr = 1-tnr  # false positive rate
        fnr = 1-tpr  # false negative rate
    except ZeroDivisionError:
        print("Error: error rates - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    # create dictionary
    error_rates = {"tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr}

    return error_rates


# calculate other scores
def calculate_additional_scores(tp, tn, fp, fn):

    """

    Calculates additional scores (F1, MCC, PPV and FDR).

    Parameters
    ----------
    tp : int
        number of true positives
    tn : int
        number of true negatives
    fp : int
        number of false positives
    fn : int
        number of false negatives


    Returns
    -------
    dict
        dictionary of additional scores (f1, mcc, ppv and fdr)

    """

    try:
        f1 = 2*tp/(2*tp+fp+fn)  # f1 score
        mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        if Decimal(mcc_denom) > Decimal(0.0):
            mcc = (tp*tn-fp*fn)/mcc_denom
        else:
            mcc_denom = 1
            mcc = (tp*tn-fp*fn)/mcc_denom  # mcc score
        if tp+fp == 0:
            ppv = 0
        else:
            ppv = tp/(tp+fp)  # precision
        fdr = 1 - ppv  # false discovery rate
    except ZeroDivisionError:
        print("Error: additional scores - division by zero!")
        sys.exit(0)

    # create a dictionary for additional scores
    additional_scores = {"f1": f1, "mcc": mcc, "ppv": ppv, "fdr": fdr}

    return additional_scores


# evaluation of single classifier
def evaluate_classifier(classifier,
                        dataset,
                        annotation,
                        negatives,
                        positives,
                        feature_cdds,
                        uniqueness,
                        bacc_weight):

    """

    Evaluates classifier using data set and it's annotation and returns classifier scores.

    Parameters
    ----------
    classifier : Classifier object
        classifier
    dataset : Pandas DataFrame object
        data set
    annotation : list
        binary data annotation
    negatives : int
        number of negative samples
    positives : int
        number of positive samples
    feature_cdds : dict
        dict of feature cdd scores
    uniqueness : bool
        if True only unique inputs in a classifier are counted, otherwise the input cdd score is multiplied by
        the number of input occurrences
    bacc_weight : float
        weight of balanced accuracy in the multi-objective score

    Returns
    -------
    classifier_score : float
        classifier multi-objective score
    bacc : float
        classifier balanced accuracy
    errors : dict
        dictionary of errors (tp, tn, fp and fn)
    error_rates : dict
        dictionary of errors (tpr, tnr, fpr and fnr)
    additional_scores : dict
        dictionary of additional scores (f1, mcc, ppv and fdr)
    cdd_score : float
        classifier cdd score

    """

    # get data info
    dataset = dataset.__copy__()

    # classifier output
    classifier_output = [0] * len(annotation)

    # assign error numbers to 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for rule in classifier.rule_set:  # evaluate every rule in the classifier

        columns = []
        for inp in rule.pos_inputs:  # add positive inputs
            columns.append(dataset[inp].tolist())  # get all feature levels across samples

        for inp in rule.neg_inputs:  # add negative inputs
            columns.append([not x for x in dataset[inp].tolist()])  # get all feature levels across samples, negated

        rule_output = []

        if rule.gate == 0:  # if gate is OR
            # rule output across samples, fill with '0' as 0 OR 0 gives 0, 0 OR 1 gives 1
            rule_output = [0] * len(annotation)
            for column in columns:  # go through columns
                rule_output = [i or j for i, j in zip(rule_output, column)]  # perform OR
        elif rule.gate == 1:  # if gate is AND
            # rule output across samples, fill with '1' as 1 AND 1 gives 1, 1 AND 0 gives 0
            rule_output = [1] * len(annotation)
            for column in columns:  # go through columns
                rule_output = [i and j for i, j in zip(rule_output, column)]  # perform AND

        classifier_output = [i + j for i, j in zip(classifier_output, rule_output)]  # sum 'yes/1' outputs

    # calculate threshold
    classifier_size = len(classifier.rule_set)
    threshold = \
        Decimal(float(classifier.evaluation_threshold) * classifier_size).to_integral_value(rounding=ROUND_HALF_UP)

    # calculate outputs
    outputs = []
    for i in classifier_output:
        if i >= threshold:  # if the sum reaches threshold - yes/1
            outputs.append(1)
        else:  # otherwise no/0
            outputs.append(0)

    # count true positives, true negatives, false positives and false negatives
    for i in range(0, len(annotation)):
        if annotation[i] == 1 and outputs[i] == 1:
            true_positives = true_positives + 1
        if annotation[i] == 0 and outputs[i] == 0:
            true_negatives = true_negatives + 1
        if annotation[i] == 0 and outputs[i] == 1:
            false_positives = false_positives + 1
        if annotation[i] == 1 and outputs[i] == 0:
            false_negatives = false_negatives + 1

    errors = dict(zip(["tp", "tn", "fp", "fn"], [true_positives, true_negatives, false_positives, false_negatives]))

    # calculate and assign scores
    error_rates = calculate_error_rates(true_positives, true_negatives, positives, negatives)
    bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)
    additional_scores = calculate_additional_scores(true_positives, true_negatives, false_positives, false_negatives)

    # get the list of inputs in classifier
    inputs = classifier.get_input_list()

    if len(feature_cdds) == 0:  # if no feature cdds provided
        cdd_score = 0
    else:  # else calculate cdd_score
        cdd_score = calculate_cdd_score(inputs, feature_cdds, uniqueness)

    # calculate cdd score
    classifier_score = calculate_multi_objective_score(bacc, cdd_score, bacc_weight)

    return classifier_score, bacc, errors, error_rates, additional_scores, cdd_score


# comparing new score to the best score
def update_best_classifier(new_classifier, best_classifiers):

    """

    Updates list of best classifiers.

    Parameters
    ----------
    new_classifier : Classifier object
        new evaluated classifier
    best_classifiers : BestSolutions object
        includes best solutions

    Returns
    -------
    best_classifiers : BestSolutions object
        updated best solutions

    """

    if is_close(best_classifiers.score, new_classifier.score):  # if new score == the best
        classifier_str = log.convert_classifier_to_string(new_classifier.__copy__())
        if classifier_str not in best_classifiers.solutions_str:
            best_classifiers.solutions.append(new_classifier.__copy__())  # add new classifier to best classifiers
            best_classifiers.solutions_str.append(classifier_str)  # add new classifier to best classifiers

    if is_higher(best_classifiers.score, new_classifier.score):  # if new score is better
        best_classifiers.score = new_classifier.score  # assign new best score
        best_classifiers.solutions.clear()  # clear the list of best classifiers
        best_classifiers.solutions_str.clear()  # clear the list of best classifiers
        classifier_str = log.convert_classifier_to_string(new_classifier.__copy__())
        best_classifiers.solutions.append(new_classifier.__copy__())  # add new classifier to best classifiers
        best_classifiers.solutions_str.append(classifier_str)  # add new classifier to best classifiers

    # toolbox.remove_symmetric_solutions(best_classifiers)

    return best_classifiers


# evaluation of the population
def evaluate_individuals(population,
                         dataset,
                         bacc_weight,
                         feature_cdds,
                         uniqueness,
                         best_classifiers):

    """

    Evaluates all classifiers in the population and returns list of best solutions.

    Parameters
    ----------
    population : list
        list of classifiers (Classifier objects)
    dataset : Pandas DataFrame
        data set
    bacc_weight : float
        weight of balanced accuracy in the multi-objective score
    feature_cdds : dict
        dict of feature cdds
    uniqueness : bool
        if True only unique inputs in a classifier are counted, otherwise the input cdd score is multiplied by
        the number of input occurrences
    best_classifiers : BestSolutions object
        includes all best classifiers

    Returns
    -------
    avg_population_score : float
        average population classifier score
    best_classifiers : BestSolutions object
        includes all best classifiers

    """

    # get data info
    header = dataset.columns.values.tolist()
    samples, annotation, negatives, positives = preproc.get_data_info(dataset, header)

    individual_scores = []  # store scores of individuals

    # evaluate all classifiers in the population
    for classifier in population:

        classifier_score, bacc, errors, error_rates, additional_scores, cdd_score = \
            evaluate_classifier(classifier, dataset, annotation, negatives, positives, feature_cdds, uniqueness,
                                bacc_weight)

        # assigning classifier scores
        classifier.errors = errors
        classifier.error_rates = error_rates
        classifier.score = classifier_score
        classifier.bacc = bacc
        classifier.cdd_score = cdd_score
        classifier.additional_scores = additional_scores

        best_classifiers = update_best_classifier(classifier, best_classifiers)

        individual_scores.append(float(classifier_score))

    # calculate average score in population
    avg_population_score = numpy.average(individual_scores)

    return avg_population_score, best_classifiers
