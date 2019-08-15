from decimal import Decimal, ROUND_HALF_UP
import sys
import pandas as pd
import random
import popinit
import math

random.seed(0)

# reading binarized data set.
def read_data(dataset_filename):

    # reading the data
    # throws an exception when datafile not found
    try:
        dataset = pd.read_csv(dataset_filename, sep=';', header=0)
    except IOError:
        print("Error: No such file or directory.")
        sys.exit(0)

    # simple check whether data is in the right format
    # needs to be improved
    header = dataset.columns.values.tolist()

    if header[0] != 'ID' or header[1] != 'Annots':
        print("Error: wrong format. The first column must include sample IDs and the second "
              "- the annotation of samples.")
        sys.exit(0)

    # counting negative and positive samples
    samples = len(dataset.index)
    negatives = dataset[dataset["Annots"] == 0].count()["Annots"]
    positives = samples - negatives

    print("Number of samples: " + str(samples))
    print("Number of negative samples: " + str(negatives))
    print("Number of positive samples: " + str(positives))

    if negatives == 0 or positives == 0:
        print("Error: no negative or positive samples in the dataset!")
        sys.exit(0)

    return dataset, negatives, positives

# removal of irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset_filename):

    dataset, negatives, positives = read_data(dataset_filename)

    relevant_mirna = []
    irrelevant_mirna = []

    # sum of miRNA levels (0/1) in each column
    column_sum = dataset.sum(axis=0, skipna=True)

    dataset_out_filename = dataset_filename.replace(".csv", "_filtered.csv")

    # if miRNA levels sum up to 0 or the number of samples in the dataset - miRNA is irrelevant (non-regulated)
    # (in other words: the whole column is filled in with 0s or 1s)
    for id, sum in column_sum.items():
        if id not in ["ID", "Annots"]:
            if sum == 0 or sum == len(dataset.index):
                irrelevant_mirna.append(id)
            else:
                relevant_mirna.append(id)

    # removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    # creating log message
    print("Number of relevant miRNAs according to a given threshold: " + str(len(relevant_mirna)))
    print("Number of irrelevant miRNAs according to a given threshold: " + str(len(irrelevant_mirna)))

    print("Relevant miRNAs: ")

    for mirna in relevant_mirna:
        print(str(mirna), " ")

    dataset.to_csv(dataset_out_filename, sep=";", index=False)


def crossvalidation(dataset_filename, kfolds):

    dataset, negatives, positives = read_data(dataset_filename)
    header = dataset.columns.values.tolist()
    data_size = len(dataset.index)

    negative_samples = dataset.iloc[:negatives].copy()
    positive_samples = dataset.iloc[negatives:data_size].copy()

    negative_samples_to_draw = int(round(negatives/kfolds))

    positive_samples_to_draw = int(round(positives/kfolds))


    negative_folds = []
    positive_folds = []

    for fold in range(1, kfolds):

        data_fold = positive_samples.sample(n=positive_samples_to_draw)
        positive_folds.append(data_fold.sort_index())
        ids_to_drop = data_fold.index.values
        positive_samples.drop(ids_to_drop, inplace=True)

        data_fold = negative_samples.sample(n=negative_samples_to_draw)
        negative_folds.append(data_fold.sort_index())
        ids_to_drop = data_fold.index.values
        negative_samples.drop(ids_to_drop, inplace=True)

    positive_folds.append(positive_samples)
    negative_folds.append(negative_samples)
    test_folds = []

    for fold in range(0, kfolds):
        test_fold = negative_folds[fold].append(positive_folds[fold])
        test_folds.append(test_fold)
        print("TEST FOLD ", fold+1, len(test_fold.index.values))

        for index, row in dataset.iterrows():
            for index2, fold_row in test_fold.iterrows():
                if row["ID"] == fold_row["ID"]:
                    if not row.equals(fold_row):
                        print("WRONG ROW")

    for fold in range(0, kfolds):
        new_name = "_test_" + str(fold+1) + ".csv"
        filename = dataset_filename.replace(".csv", new_name)
        test_folds[fold].to_csv(filename, sep=";", index=False)


        train_folds_to_merge = []
        for i in range(0, kfolds):
            if i != fold:
                train_folds_to_merge.append(test_folds[i])

        train_fold = train_folds_to_merge[0].copy()
        for i in range(1, len(train_folds_to_merge)):
            train_fold = train_fold.append(train_folds_to_merge[i])

        for index, row in dataset.iterrows():
            for index2, fold_row in train_fold.iterrows():
                if row["ID"] == fold_row["ID"]:
                    if not row.equals(fold_row):
                        print("WRONG ROW")

        print("TRAIN FOLD ", fold + 1, len(train_fold.index.values))
        train_fold = train_fold.sort_index()
        new_name = "_train_" + str(fold + 1) + ".csv"
        filename = dataset_filename.replace(".csv", new_name)
        train_fold.to_csv(filename, sep=";", index=False)


# balanced accuracy score
def calculate_balanced_accuracy(tp, tn, p, n):

    try:
        balanced_accuracy = (tp/p + tn/n)/2
    except ZeroDivisionError:
        print("Error: balanced accuracy - division by zero! No negatives or positives in the dataset!")
        sys.exit(0)

    return balanced_accuracy

# evaluation of the population
def evaluate_classifier(classifier,
                        evaluation_threshold,
                        dataset_filename):

    # read data
    dataset, negatives, positives = read_data(dataset_filename)

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

    bacc = calculate_balanced_accuracy(true_positives, true_negatives, positives, negatives)

    print("BACC: ", bacc)
    print("TP: ", true_positives)
    print("TN: ", true_negatives)
    print("FP: ", false_positives)
    print("FN: ", false_negatives)

    return bacc

def generate_parameters(iter_lower, iter_upper, iter_step,
                        pop_lower, pop_upper, pop_step,
                        cp_lower, cp_upper, cp_step,
                        mp_lower, mp_upper, mp_step,
                        ts_lower, ts_upper, ts_step,
                        param_sets_numb):

    iterations = [i for i in range(iter_lower, iter_upper+1, iter_step)]
    population_size = [i for i in range(pop_lower, pop_upper+1, pop_step)]
    crossover_probability = [i/100 for i in range(cp_lower, cp_upper+1, cp_step)]
    mutation_probability = [i/100 for i in range(mp_lower, mp_upper+1, mp_step)]
    tournament_size = [i/100 for i in range(ts_lower, ts_upper+1, ts_step)]

    parameter_sets = []
    for i in range(0, param_sets_numb):
        iter = iterations[random.randrange(0, len(iterations))]
        pop = population_size[random.randrange(0, len(population_size))]
        cp = crossover_probability[random.randrange(0, len(crossover_probability))]
        mp = mutation_probability[random.randrange(0, len(mutation_probability))]
        ts = tournament_size[random.randrange(0, len(tournament_size))]
        while [iter, pop, cp, mp, ts] in parameter_sets:
            iter = iterations[random.randrange(0, len(iterations))]
            pop = population_size[random.randrange(0, len(population_size))]
            cp = crossover_probability[random.randrange(0, len(crossover_probability))]
            mp = mutation_probability[random.randrange(0, len(mutation_probability))]
            ts = tournament_size[random.randrange(0, len(tournament_size))]
        parameter_sets.append([iter, pop, cp, mp, ts])

    return parameter_sets

def generate_bash_scripts():

    parameter_sets = generate_parameters(25, 100, 25,
                                         50, 300, 50,
                                         10, 100, 10,
                                         10, 100, 10,
                                         10, 50, 10,
                                         300)

    train_data_1 = "C1_bt250_filtered_train_1.csv"
    train_data_2 = "C1_bt250_filtered_train_2.csv"
    train_data_3 = "C1_bt250_filtered_train_3.csv"

    test_data_1 = "C1_bt250_filtered_test_1.csv"
    test_data_2 = "C1_bt250_filtered_test_2.csv"
    test_data_3 = "C1_bt250_filtered_test_3.csv"

    run_commands = []

    for i in range(0, len(parameter_sets)):

        log_name = "log_" + train_data_1.replace(".csv", "") + "_" + str(parameter_sets[i][0]) + "_" \
                   + str(parameter_sets[i][1]) + "_" + str(parameter_sets[i][2]) + "_" \
                   + str(parameter_sets[i][3]) + "_" + str(parameter_sets[i][4])

        run_command = "python3 /home/mnowicka/run_GA.py" + \
                      " -train ./data/" + train_data_1 + " -test ./data/" + test_data_1 + \
                      " -iter " + str(parameter_sets[i][0]) + " -pop " + str(parameter_sets[i][1]) + \
                      " -size " + str(5) + " -thres " + str(0.5) + " -cp " + str(parameter_sets[i][2]) + \
                      " -mp " + str(parameter_sets[i][3]) + " -ts " + str(parameter_sets[i][4]) + \
                      " > " + log_name

        script = "#!/bin/bash\n#SBATCH -J " + log_name + "\n#SBATCH -D /data/scratch/mnowicka\n#SBATCH -o " \
                 + log_name + ".%j.out\n#SBATCH -n 1\n#SBATCH --time=5-00:00:00\n" \
                              "#SBATCH --partition=big\n#SBATCH --mem=500M\n#SBATCH --cpus-per-task=1\n\n" \
                              "source /home/mnowicka/venv/bin/activate\n" + run_command

        script_file = open("bash_scripts/script_" + log_name + ".sh", 'w')
        script_file.write(script)
        script_file.close()

        log_name = "log_" + train_data_2.replace(".csv", "") + "_" + str(parameter_sets[i][0]) + \
                   "_" + str(parameter_sets[i][1]) + "_" + str(parameter_sets[i][2]) + "_" \
                   + str(parameter_sets[i][3]) + "_" + str(parameter_sets[i][4])

        run_command = "python3 /home/mnowicka/run_GA.py" +\
                      " -train ./data/" + train_data_2 + " -test ./data/" + test_data_2 + \
                      " -iter " + str(parameter_sets[i][0]) + " -pop " + str(parameter_sets[i][1]) + \
                      " -size " + str(5) + " -thres " + str(0.5) + " -cp " + str(parameter_sets[i][2]) + \
                      " -mp " + str(parameter_sets[i][3]) + " -ts " + str(parameter_sets[i][4]) + \
                      " > " + log_name

        script = "#!/bin/bash\n#SBATCH -J " + log_name + "\n#SBATCH -D /data/scratch/mnowicka\n#SBATCH -o " \
                 + log_name + ".%j.out\n#SBATCH -n 1\n#SBATCH --time=5-00:00:00\n" \
                              "#SBATCH --partition=big\n#SBATCH --mem=500M\n#SBATCH --cpus-per-task=1\n\n" \
                              "source /home/mnowicka/venv/bin/activate\n" + run_command

        script_file = open("bash_scripts/script_" + log_name + ".sh", 'w')
        script_file.write(script)
        script_file.close()

        log_name = "log_" + train_data_3.replace(".csv", "") + "_" + str(parameter_sets[i][0]) + "_" \
                   + str(parameter_sets[i][1]) + "_" + str(parameter_sets[i][2]) + "_" \
                   + str(parameter_sets[i][3]) + "_" + str(parameter_sets[i][4])

        run_command = "python3 /home/mnowicka/run_GA.py" +\
                      " -train ./data/" + train_data_3 + " -test ./data/" + test_data_3 + \
                      " -iter " + str(parameter_sets[i][0]) + " -pop " + str(parameter_sets[i][1]) + \
                      " -size " + str(5) + " -thres " + str(0.5) + " -cp " + str(parameter_sets[i][2]) + \
                      " -mp " + str(parameter_sets[i][3]) + " -ts " + str(parameter_sets[i][4]) + \
                      " > " + log_name

        script = "#!/bin/bash\n#SBATCH -J " + log_name + "\n#SBATCH -D /data/scratch/mnowicka\n#SBATCH -o " \
                 + log_name + ".%j.out\n#SBATCH -n 1\n#SBATCH --time=5-00:00:00\n" \
                              "#SBATCH --partition=big\n#SBATCH --mem=500M\n#SBATCH --cpus-per-task=1\n\n" \
                              "source /home/mnowicka/venv/bin/activate\n" + run_command

        script_file = open("bash_scripts/script_" + log_name + ".sh", 'w')
        script_file.write(script)
        script_file.close()

def print_classifier(classifier):

    classifier_message = ""

    for rule in classifier.rule_set:
        rule_message = ""

        if len(rule.neg_inputs) == 0 and len(rule.pos_inputs) == 1:
            rule_message = "(" + str(rule.pos_inputs[0]) + ")"
            # print(rule_message)
        elif len(rule.pos_inputs) == 0 and len(rule.neg_inputs) == 1:
            rule_message = "(NOT " + str(rule.neg_inputs[0]) + ")"
            # print(rule_message)
        else:
            for input in rule.pos_inputs:
                rule_message = rule_message + "(" + input + ")"
                # print(rule_message)
            for input in rule.neg_inputs:
                rule_message = rule_message + "(NOT " + input + ")"
                # print(rule_message)

        rule_message = " [" + rule_message + "] "

        rule_message = rule_message.replace(")(", ") AND (")

        classifier_message = classifier_message + rule_message

    return classifier_message


def discretize_miRNA(miR_expr, annots, negatives, positives, m_segments, alpha_param, lambda_param):

    miR_expr_sorted, annots_sorted = zip(*sorted(zip(miR_expr, annots)))

    segment_step = (max(miR_expr) - min(miR_expr))/m_segments

    segment_threshold = min(miR_expr) + segment_step

    segments = []

    cdds = []

    for m in range(0, m_segments):

        segment = [i for i in miR_expr_sorted if i <= segment_threshold]

        segment_threshold += segment_step

        segments.append(segment)


    for segment in segments:

        neg_class = annots_sorted[0:len(segment)].count(0)
        pos_class = annots_sorted[0:len(segment)].count(1)

        cdd = neg_class/negatives - pos_class/positives

        cdds.append(cdd)

    cdd_max = max(cdds)
    cdd_min = min(cdds)

    cdd_max_abs = math.fabs(max(cdds))
    cdd_min_abs = math.fabs(min(cdds))

    difference = math.fabs(cdd_max-cdd_min)

    cutpoint = 0

    if difference < alpha_param or max(cdd_max_abs, cdd_min_abs) < lambda_param:
        print("ONE STATE")

    if difference >= alpha_param:
        if max(cdd_max_abs, cdd_min_abs) >= lambda_param:
            if min(cdd_max_abs, cdd_min_abs) < lambda_param:
                print("TWO STATES")

                if cdd_max_abs > cdd_min_abs:
                    index = cdds.index(cdd_max) + 1
                    cutpoint = min(miR_expr) + segment_step * index

                if cdd_max_abs <= cdd_min_abs:
                    index = cdds.index(cdd_min) + 1
                    cutpoint = min(miR_expr) + segment_step * index

        print("CUTPOINT: ", cutpoint)

    if difference >= alpha_param and min(cdd_max_abs, cdd_min_abs) >= lambda_param:
        print("THREE STATES")

    return cutpoint


def discretize_data(con_data_fname_train, m_segments, alpha_param, lambda_param):

    dataset, negatives, positives = read_data(con_data_fname_train)

    new_file = str(con_data_fname_train.replace(".csv", "")) + "_discretized_" + str(m_segments) \
               + "_" + str(alpha_param) + "_" + str(lambda_param)

    annots = dataset["Annots"].tolist()

    miRNAs = dataset.columns.values.tolist()[2:]

    thresholds = []

    data_discretized = dataset.drop(miRNAs, axis=1)

    for miRNA in miRNAs:

        threshold = 0

        print("miRNA ", miRNA)

        miR_expr = dataset[miRNA].tolist()

        threshold = discretize_miRNA(miR_expr, annots, negatives, positives, m_segments, alpha_param, lambda_param)

        thresholds.append(threshold)

        miR_discretized = [0 if i <= threshold else 1 for i in miR_expr]

        data_discretized[miRNA] = miR_discretized

    data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return miRNAs, thresholds


def discretize_with_thresholds(con_data_fname_test, miRNAs, thresholds):

    dataset, negatives, positives = read_data(con_data_fname_test)

    new_file = str(con_data_fname_test.replace(".csv", "")) + "_discretized"

    annots = dataset["Annots"].tolist()

    miRNAs = dataset.columns.values.tolist()[2:]

    data_discretized = dataset.drop(miRNAs, axis=1)

    miR_dict = dict(zip(miRNAs, thresholds))

    for miRNA in miRNAs:

        miR_expr = dataset[miRNA].tolist()

        miR_discretized = [0 if i <= miR_dict[miRNA] else 1 for i in miR_expr]

        data_discretized[miRNA] = miR_discretized

    data_discretized.to_csv(new_file+".csv", index=False, sep=";")


def discretize_data_for_tests(con_data_fname_train, con_data_fname_test, m_segments, alpha_param, lambda_param):

    miRNAs, thresholds = discretize_data(con_data_fname_train, m_segments, alpha_param, lambda_param)

    discretize_with_thresholds(con_data_fname_test, miRNAs, thresholds)

