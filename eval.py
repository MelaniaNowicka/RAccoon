from decimal import Decimal, ROUND_HALF_UP
import toolbox
import pandas
import math
import sys

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

    classifier_cdds = 0.0
    # sum up cdds for each miRNA in the classifier
    for input in inputs:
        classifier_cdds = classifier_cdds + miRNA_cdds.get(input)

    try:
        classifier_cdd_score = classifier_cdds / len(inputs)
    except ZeroDivisionError:
        print("Error: cdd score - division by zero! No inputs in a classifier! ")

    return classifier_cdd_score

# calculate multi-objective score
# balanced accuracy with cdd score
def calculate_multi_objective_score(bacc, cdd_score, bacc_weight):

    classifier_score = bacc * bacc_weight + cdd_score * (1.0-bacc_weight)

    return classifier_score
    

# calculate error rates
def calculate_error_rates(tp, tn, p, n):

    try:
        tpr = tp/p  # true positive rate
        tnr = tn/n  # true negative rate
        fpr = 1-tnr  # false positive rate
        fnr = 1-tpr  # false negative rate
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    error_rates = {}
    error_rates["tpr"] = tpr
    error_rates["tnr"] = tnr
    error_rates["fpr"] = fpr
    error_rates["fnr"] = fnr

    return error_rates

# calculate other scores
def calculate_additional_scores(tp, tn, p, n):

    fp = p - tp  # false positives
    fn = n - tn  # false negatives

    try:
        f1 = 2*tp/(2*tp+fp+fn)  # f1 score
        mcc = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))  # mcc score
        ppv = tp/(tp+fp)  # precision
        fdr = 1 - ppv  # false discovery rate
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
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
                        annotation,
                        dataset,
                        evaluation_threshold,
                        miRNA_cdds,
                        bacc_weight):

    if isinstance(dataset, pandas.DataFrame):
        dataset = dataset.__copy__()
        samples = len(dataset.index)
        negatives = dataset[dataset["Annots"] == 0].count()["Annots"]
        positives = samples - negatives
    else:
        # read data
        dataset, negatives, positives = toolbox.read_data(dataset)

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
            columns.append(dataset[input].tolist())

        for input in rule.neg_inputs:  # add negative inputs
            columns.append([not x for x in dataset[input].tolist()])

        # rule output
        rule_output = [1] * len(annotation)

        for column in columns:
            rule_output = [i and j for i, j in zip(rule_output, column)]

        classifier_output = [i + j for i, j in zip(classifier_output, rule_output)]

    # calculate threshold
    threshold = Decimal(evaluation_threshold * len(classifier.rule_set)).to_integral_value(rounding=ROUND_HALF_UP)

    # calculate outputs
    outputs = []
    for i in classifier_output:
        if i >= threshold:
            outputs.append(1)
        else:
            outputs.append(0)

    # count true positives, true negatives, false positives and false negatives
    for i in range(0, len(annotation)):
        if annotation[i] == 1 and outputs[i] == 1:
            true_positives = true_positives + 1
        if annotation[i] == 0 and outputs[i] == 0:
            true_negatives = true_negatives + 1
        if annotation[i] == 1 and outputs[i] == 0:
            false_negatives = false_negatives + 1
        if annotation[i] == 0 and outputs[i] == 1:
            false_positives = false_positives + 1

    # list of inputs
    inputs = []

    # get the list of inputs in classifier
    inputs = classifier.get_input_list()

    # calculate and assign scores
    error_keys = ["tp", "tn", "fp", "fn"]
    error_values = [true_positives, true_negatives, false_negatives, false_positives]
    errors = dict(zip(error_keys, error_values))
    error_rates = calculate_error_rates(true_positives, true_negatives, positives, negatives)
    bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)
    additional_scores = calculate_additional_scores(true_positives, true_negatives, positives, negatives)
    cdd_score = calculate_cdd_score(inputs, miRNA_cdds)

    if (bacc_weight == 1):
        classifer_score = bacc
    else:
        classifer_score = calculate_multi_objective_score(bacc, cdd_score, bacc_weight)

    return bacc, errors, error_rates, additional_scores, cdd_score


# comparing new score to the best score
def update_best_classifier(new_classifier, best_bacc, best_classifiers):

    if best_bacc < new_classifier.bacc:  # if new score is better
        best_bacc = new_classifier.bacc  # assign new best score
        best_classifiers.clear()  # clear the list of best classifiers
        best_classifiers.append(new_classifier.__copy__())  # add new classifier

    if best_bacc == new_classifier.bacc:  # if new score == the best
        best_classifiers.append(new_classifier.__copy__())  # add new classifier to best classifiers

    return best_bacc, best_classifiers

# evaluation of the population
def evaluate_individuals(population,
                         evaluation_threshold,
                         miRNA_cdds,
                         dataset,
                         negatives,
                         positives,
                         best_bacc,
                         best_classifiers):

    # sum of bacc for the population
    sum_bacc = 0.0

    # get annotation
    annotation = dataset["Annots"].tolist()

    # evaluate all classifiers in the population
    for classifier in population:

        bacc, errors, error_rates, additional_scores, cdd_score = evaluate_classifier(classifier,
                                                                                              annotation,
                                                                                              dataset,
                                                                                              evaluation_threshold,
                                                                                              miRNA_cdds)

        # assigning classifier scores
        classifier.errors = errors
        classifier.error_rates = error_rates
        classifier.bacc = bacc
        classifier.other_scores = additional_scores
        classifier.cdd_score = cdd_score

        sum_bacc = sum_bacc + classifier.bacc

        best_bacc, best_classifiers = update_best_classifier(classifier, best_bacc, best_classifiers)

    # calculate average BACC
    avg_bacc = sum_bacc / len(population)

    return best_bacc, avg_bacc, best_classifiers

