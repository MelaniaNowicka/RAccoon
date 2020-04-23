def write_generation_to_log(population, iteration, best_classifiers, log_message):

    id = 1
    log_message = log_message + "\n\n ITERATION " + str(iteration) + "\n"
    sum_len = 0
    for classifier in population:
        sum_len = sum_len + len(classifier.rule_set)

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

    log_message = log_message + "AVERAGE CLASSIFIER LENGTH: " + str(sum_len/len(population)) + "\n"
    print(log_message)

    return log_message


def write_final_scores(best_bacc, best_classifiers):

    # final score
    print("BEST DC SCORE:", best_bacc)

    best_classifiers_messages = []

    # final result
    for classifier in best_classifiers:
        classifier_message = ""
        for rule in classifier.rule_set:
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

        best_classifiers_messages.append(classifier_message)

    # removing duplicates
    best_classifiers_messages = set(best_classifiers_messages)

    for classifier in best_classifiers_messages:
        print("BEST CLASSIFIER: ", classifier)
        log_message = "\nBEST SCORE: " + str(best_bacc) + "| BEST CLASSIFIER: " + str(classifier)

    return log_message
