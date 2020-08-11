from decimal import *
import preproc
import numpy
import math
import sys


# compare floats - is close with 1e-5 tolerance
def is_close(x, y, tol=1e-5):
    if abs(x - y) <= tol:
        return True
    else:
        return False


# compare floats - is y is higher than x with 1e-5 tolerance
def is_higher(x, y, tol=1e-5):
    if y - x >= tol:
        return True
    else:
        return False


# calculate balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy


# calculate classifier cdd score
# class distribution divergence score
def calculate_cdd_score(inputs, miRNA_cdds):

    # sum up cdds for each miRNA in the classifier
    classifier_cdd_sum = 0
    for input in inputs:
        try:
            classifier_cdd_sum += miRNA_cdds.get(input)
        except TypeError:
            print("Error: cdd score. Feature:", input, " - score not found.")
            sys.exit(0)

    # calculate cdd score
    classifier_cdd_score = 0
    try:
        classifier_cdd_score = classifier_cdd_sum / len(inputs)
    except ZeroDivisionError:
        print("Error: cdd score - division by zero! No inputs in the classifier.")
        sys.exit(0)

    return classifier_cdd_score


# calculate multi-objective score
# balanced accuracy and cdd score
def calculate_multi_objective_score(bacc, cdd_score, bacc_weight):

    classifier_score = bacc * bacc_weight + cdd_score * (1-bacc_weight)

    return classifier_score


# calculate error rates
def calculate_error_rates(tp, tn, p, n):

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
    additional_scores = {}
    additional_scores["f1"] = f1
    additional_scores["mcc"] = mcc
    additional_scores["ppv"] = ppv
    additional_scores["fdr"] = fdr

    return additional_scores


# evaluation of single classifier
def evaluate_classifier(classifier,
                        dataset,
                        annotation,
                        negatives,
                        positives,
                        feature_cdds,
                        bacc_weight):

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
        for input in rule.pos_inputs:  # add positive inputs
            columns.append(dataset[input].tolist())  # get all feature levels across samples

        for input in rule.neg_inputs:  # add negative inputs
            columns.append([not x for x in dataset[input].tolist()])  # get all feature levels across samples, negate

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
        cdd_score = calculate_cdd_score(inputs, feature_cdds)

    # calculate cdd score
    classifier_score = calculate_multi_objective_score(bacc, cdd_score, bacc_weight)

    return classifier_score, bacc, errors, error_rates, additional_scores, cdd_score


# comparing new score to the best score
def update_best_classifier(new_classifier, global_best_score, best_classifiers):

    if is_close(global_best_score, new_classifier.score):  # if new score == the best
        best_classifiers.append(new_classifier.__copy__())  # add new classifier to best classifiers

    if is_higher(global_best_score, new_classifier.score):  # if new score is better
        global_best_score = new_classifier.score  # assign new best score
        best_classifiers.clear()  # clear the list of best classifiers
        best_classifiers.append(new_classifier.__copy__())  # add new classifier

    return global_best_score, best_classifiers


# evaluation of the population
def evaluate_individuals(population,
                         dataset,
                         bacc_weight,
                         feature_cdds,
                         global_best_score,
                         best_classifiers):

    # sum of bacc for the population
    sum_bacc = 0.0

    # get data info
    header = dataset.columns.values.tolist()
    samples, annotation, negatives, positives = preproc.get_data_info(dataset, header)

    individual_scores = []  # store scores of individuals
    update = False  # check whether there was best score update

    # evaluate all classifiers in the population
    for classifier in population:

        classifier_score, bacc, errors, error_rates, additional_scores, cdd_score = \
            evaluate_classifier(classifier, dataset, annotation, negatives, positives, feature_cdds, bacc_weight)

        # assigning classifier scores
        classifier.errors = errors
        classifier.error_rates = error_rates
        classifier.score = classifier_score
        classifier.bacc = bacc
        classifier.cdd_score = cdd_score
        classifier.additional_scores = additional_scores

        global_best_score, best_classifiers = update_best_classifier(classifier, global_best_score, best_classifiers)

        individual_scores.append(classifier_score)

    # calculate average score in population
    avg_population_score = numpy.average(individual_scores)

    return global_best_score, avg_population_score, best_classifiers

