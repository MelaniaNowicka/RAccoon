def write_generation_to_log(population, iteration, log_message):

    id = 1
    log_message = log_message + "\n\n ITERATION " + str(iteration) + "\n"
    for classifier in population:

        log_message = log_message + "C" + str(id) + ": "
        id = id + 1
        classifier_message = ""
        for rule in classifier.rule_set:
            rule_message = ""

            if len(rule.neg_inputs) == 0 and len(rule.pos_inputs) == 1:
                rule_message = "(" + str(rule.pos_inputs[0]) + ")"
                #print(rule_message)
            elif len(rule.pos_inputs) == 0 and len(rule.neg_inputs) == 1:
                rule_message = "(NOT " + str(rule.neg_inputs[0]) + ")"
                #print(rule_message)
            else:
                for input in rule.pos_inputs:
                    rule_message = rule_message + "(" + input + ")"
                    #print(rule_message)
                for input in rule.neg_inputs:
                    rule_message = rule_message + "(NOT " + input + ")"
                    #print(rule_message)

            rule_message = " [" + rule_message + "] "

            rule_message = rule_message.replace(")(", ") AND (")

            classifier_message = classifier_message + rule_message

        log_message = log_message + classifier_message + "\n"
        log_message = log_message + "BACC: " + str(classifier.bacc) + "; TP: " + str(classifier.error_rates["tp"]) \
                      + "; TN: " + str(classifier.error_rates["tn"]) + "\n"

    return log_message


def write_final_scores(best_bacc, best_classifier):

    # final score
    print("BEST BACC:", best_bacc)

    # final result
    classifier_message = ""

    for rule in best_classifier.rule_set:
        rule_message = ""

        if len(rule.neg_inputs) == 0 and len(rule.pos_inputs) == 1:
            rule_message = "(" + str(rule.pos_inputs[0]) + ")"
        elif len(rule.pos_inputs) == 0 and len(rule.neg_inputs) == 1:
            rule_message = "(NOT " + str(rule.neg_inputs[0]) + ")"
        else:
            for input in rule.pos_inputs:
                rule_message = rule_message + "(" + input + ")"
            for input in rule.neg_inputs:
                rule_message = rule_message + "(NOT " + input + ")"

        rule_message = " [" + rule_message + "] "

        rule_message = rule_message.replace(")(", ") AND (")

        classifier_message = classifier_message + rule_message

    print("BEST CLASSIFIER: ", classifier_message)

    log_message = "BEST SCORE: " + str(best_bacc) + "\n" + classifier_message

    return log_message