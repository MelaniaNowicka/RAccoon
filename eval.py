from decimal import Decimal, ROUND_HALF_UP
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

    # a list of rule results
    rule_outputs = []

    error_rates = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for classifier in population:  # evaluating every classifier
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for sample_id in range(0, len(dataset.index)):  # iterate through dataset
            sample_output = 0  # single sample output
            rule_outputs.clear()  # clearing the rule outputs for a single sample

            for rule in classifier.rule_set:  # evaluating every rule in the classifier

                rule_output = 1

                for input in rule.pos_inputs:  # positive inputs
                    rule_output = rule_output and dataset.iloc[sample_id][input]
                for input in rule.neg_inputs:  # negative inputs
                    rule_output = rule_output and not dataset.iloc[sample_id][input]

                rule_outputs.append(rule_output)  # adding a single rule output to a list of rule outputs

            rule_positive_outputs = 0
            # count positive(1) outputs
            for result in rule_outputs:
                if result == 1:
                    rule_positive_outputs = rule_positive_outputs + 1

            # calculate the sample decision
            dec = Decimal(evaluation_threshold * len(rule_outputs)).to_integral_value(rounding=ROUND_HALF_UP)
            if rule_positive_outputs >= dec:
                sample_output = 1
            else:
                sample_output = 0

            # counting tps, tns, fps and fns
            if dataset.iloc[sample_id]['Annots'] == 1 and sample_output == 1:
                true_positives = true_positives + 1
            if dataset.iloc[sample_id]['Annots'] == 0 and sample_output == 0:
                true_negatives = true_negatives + 1
            if dataset.iloc[sample_id]['Annots'] == 1 and sample_output == 0:
                false_negatives = false_negatives + 1
            if dataset.iloc[sample_id]['Annots'] == 0 and sample_output == 1:
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
