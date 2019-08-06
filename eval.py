from decimal import Decimal, ROUND_HALF_UP
import multiprocessing as mp
import sys

# balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy

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
        classifier.error_rates["tp"] = true_positives
        classifier.error_rates["tn"] = true_negatives
        classifier.error_rates["fp"] = false_positives
        classifier.error_rates["fn"] = false_negatives
        classifier.bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)

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
