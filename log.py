def convert_classifier_to_string(classifier):

    """

    Converts Classifier object to a string.

    Parameters
    __________
    classifier : Classifier object
        classifier as Classifier object

    Returns
    _______
    classifier_str : str
        classifier as a string

    """

    classifier_str = ""
    rule_str_list = []
    for rule in classifier.rule_set:
        rule_str = ""

        for input in sorted(rule.pos_inputs):
            rule_str = rule_str + "(" + input + ")"
        for input in sorted(rule.neg_inputs):
            rule_str = rule_str + "(NOT " + input + ")"

        rule_str = " [" + rule_str + "] "

        if rule.gate == 0:
            rule_str = rule_str.replace(")(", ") OR (")
        elif rule.gate == 1:
            rule_str = rule_str.replace(")(", ") AND (")

        rule_str_list.append(rule_str)

    classifier_str = classifier_str.join(sorted(rule_str_list))

    classifier_str = classifier_str + " | THRESHOLD: " + str(classifier.evaluation_threshold)

    return classifier_str


def write_final_scores(best_bacc, best_classifiers):

    """

    Converts all best found classifiers to strings and returns a log message.

    Parameters
    __________
    best_bacc : float
        best balanced accuracy
    best_classifiers : BestSolutions object
        includes all best classifiers

    Returns
    _______
    log_message : str
        log message including best scores and classifiers as string

    """

    # final score
    print("BEST DC SCORE:", best_bacc)

    best_classifiers_messages = []

    # final result
    for classifier in best_classifiers:
        classifier_str = ""
        for rule in classifier.rule_set:
            rule_str = ""

            for input in rule.pos_inputs:
                rule_str = rule_str + "(" + input + ")"
            for input in rule.neg_inputs:
                rule_str = rule_str + "(NOT " + input + ")"

            rule_str = " [" + rule_str + "] "

            if rule.gate == 0:
                rule_str = rule_str.replace(")(", ") OR (")
            elif rule.gate == 1:
                rule_str = rule_str.replace(")(", ") AND (")

            classifier_str = classifier_str + rule_str

        classifier.update_theta()
        classifier_str = classifier_str + " | THRESHOLD: " + str(classifier.evaluation_threshold) \
                         + " | THETA: " + str(classifier.theta)

        best_classifiers_messages.append(classifier_str)

    # removing duplicates
    best_classifiers_messages = set(best_classifiers_messages)

    for classifier in best_classifiers_messages:
        print("BEST CLASSIFIER: ", classifier)

        log_message = "\nBEST SCORE: " + str(best_bacc) + "| BEST CLASSIFIER: " + str(classifier)

    return log_message
