from decimal import Decimal, ROUND_HALF_UP
import toolbox
import pandas
import math
import sys

# balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy

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


# evaluation of the population
def evaluate_individuals(population,
                         evaluation_threshold,
                         dataset,
                         negatives,
                         positives,
                         best_bacc,
                         best_classifiers):

    # sum of bacc for the population
    sum_bacc = 0.0

    annots = dataset["Annots"].tolist()

    for classifier in population:  # evaluating every classifier
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        classifier_output = [0] * len(annots)

        for rule in classifier.rule_set:  # evaluating every rule in the classifier
            columns = []
            for input in rule.pos_inputs:
                columns.append(dataset[input].tolist())
            for input in rule.neg_inputs:
                columns.append([not x for x in dataset[input].tolist()])

            rule_output = [1] * len(annots)

            for column in columns:
                rule_output = [i and j for i, j in zip(rule_output, column)]

            classifier_output = [i + j for i, j in zip(classifier_output, rule_output)]

        dec = Decimal(evaluation_threshold * len(classifier.rule_set)).to_integral_value(rounding=ROUND_HALF_UP)
        outputs = []
        for i in classifier_output:
            if i >= dec:
                outputs.append(1)
            else:
                outputs.append(0)

        for i in range(0, len(annots)):
            if annots[i] == 1 and outputs[i] == 1:
                true_positives = true_positives + 1
            if annots[i] == 0 and outputs[i] == 0:
                true_negatives = true_negatives + 1
            if annots[i] == 1 and outputs[i] == 0:
                false_negatives = false_negatives + 1
            if annots[i] == 0 and outputs[i] == 1:
                false_positives = false_positives + 1

        # assigning classifier scores
        classifier.errors["tp"] = true_positives
        classifier.errors["tn"] = true_negatives
        classifier.errors["fp"] = false_positives
        classifier.errors["fn"] = false_negatives
        classifier.error_rates["tpr"] = true_positives/positives
        classifier.error_rates["tnr"] = true_negatives/negatives
        classifier.error_rates["fpr"] = 1 - true_negatives/negatives
        classifier.error_rates["fnr"] = 1 - true_positives/positives
        classifier.bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)
        classifier.other_scores = calculate_additional_scores(true_positives, true_negatives, positives, negatives)

        sum_bacc = sum_bacc + classifier.bacc

        # comparing new score to the best score
        if best_bacc < classifier.bacc:  # if new score is better
            best_bacc = classifier.bacc  # assign new best score
            best_classifiers.clear()  # clear the list of best classifiers
            best_classifiers.append(classifier.__copy__())  # add new classifier

        if best_bacc == classifier.bacc:  # if new score == the best
            best_classifiers.append(classifier.__copy__())  # add new classifier to best classifiers

    # calculate average BACC
    avg_bacc = sum_bacc / len(population)

    return best_bacc, avg_bacc, best_classifiers


# evaluation of the population
def evaluate_classifier(classifier,
                        evaluation_threshold,
                        dataset_filename):

    if isinstance(dataset_filename, pandas.DataFrame):
        dataset = dataset_filename.__copy__()
        samples = len(dataset_filename.index)
        negatives = dataset_filename[dataset_filename["Annots"] == 0].count()["Annots"]
        positives = samples - negatives
    else:
        # read data
        dataset, negatives, positives = toolbox.read_data(dataset_filename)

    annots = dataset["Annots"].tolist()

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    classifier_output = [0] * len(annots)

    for rule in classifier.rule_set:  # evaluating every rule in the classifier
        columns = []
        for input in rule.pos_inputs:
            columns.append(dataset[input].tolist())
        for input in rule.neg_inputs:
            columns.append([not x for x in dataset[input].tolist()])

        rule_output = [1] * len(annots)

        for column in columns:
            rule_output = [i and j for i, j in zip(rule_output, column)]

        classifier_output = [i + j for i, j in zip(classifier_output, rule_output)]

    dec = Decimal(evaluation_threshold * len(classifier.rule_set)).to_integral_value(rounding=ROUND_HALF_UP)
    outputs = []
    for i in classifier_output:
        if i >= dec:
            outputs.append(1)
        else:
            outputs.append(0)

    for i in range(0, len(annots)):
        if annots[i] == 1 and outputs[i] == 1:
            true_positives = true_positives + 1
        if annots[i] == 0 and outputs[i] == 0:
            true_negatives = true_negatives + 1
        if annots[i] == 1 and outputs[i] == 0:
            false_negatives = false_negatives + 1
        if annots[i] == 0 and outputs[i] == 1:
            false_positives = false_positives + 1

    error_rates = calculate_error_rates(true_positives, true_negatives, positives, negatives)
    bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)
    additional_scores = calculate_additional_scores(true_positives, true_negatives, positives, negatives)

    print("TPR: ", error_rates["tpr"])
    print("TNR: ", error_rates["tnr"])
    print("FPR: ", error_rates["fpr"])
    print("FNR: ", error_rates["fnr"])
    print("BACC: ", bacc)
    print("F1: ", additional_scores["f1"])
    print("MCC: ", additional_scores["mcc"])
    print("PPV: ", additional_scores["ppv"])
    print("FDR: ", additional_scores["fdr"])

    return bacc, error_rates, additional_scores
