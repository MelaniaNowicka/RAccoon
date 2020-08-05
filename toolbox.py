from decimal import Decimal, ROUND_HALF_UP
import sys
import preproc
import pandas
import random

random.seed(1)


def compare_folds(fold1, fold2):

    if isinstance(fold1, pandas.DataFrame) and isinstance(fold2, pandas.DataFrame):
        data1 = fold1.__copy__()
        data2 = fold2.__copy__()
    else:
        data1, negatives, positives = preproc.read_data(fold1)
        data2, negatives, positives = preproc.read_data(fold2)

    samples1 = list(data1["ID"])
    samples2 = list(data2["ID"])

    print("SAMPLES MATCH: ", samples1 == samples2)

    print("DATA MATCH: ", data1.equals(data2))


# removal of irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset_filename):

    dataset, negatives, positives = preproc.read_data(dataset_filename)

    relevant_mirna = []
    irrelevant_mirna = []

    # sum of miRNA levels (0/1) in each column
    column_sum = dataset.sum(axis=0, skipna=True)

    dataset_out_filename = dataset_filename.replace(".csv", "_filtered.csv")

    # if miRNA levels sum up to 0 or the number of samples in the dataset - miRNA is irrelevant (non-regulated)
    # (in other words: the whole column is filled in with 0s or 1s)
    for id, sum in column_sum.items():
        if id not in ["ID", "Annots"]:
            print("S", sum)
            print(len(dataset.index))
            if sum == 0 or sum == len(dataset.index):

                irrelevant_mirna.append(id)
            else:
                relevant_mirna.append(id)

    # removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    # creating log message
    print("Number of relevant miRNAs according to a given threshold: " + str(len(relevant_mirna)))
    print("Number of irrelevant miRNAs according to a given threshold: " + str(len(irrelevant_mirna)))

    dataset.to_csv(dataset_out_filename, sep=";", index=False)


def divide_into_train_test(dataset_filename, train_frac):

    dataset, negatives, positives = preproc.read_data(dataset_filename)
    header = dataset.columns.values.tolist()

    data_size = len(dataset.index)

    negative_samples = dataset.iloc[:negatives].copy()
    positive_samples = dataset.iloc[negatives:data_size].copy()

    # create training set
    negative_samples_to_draw = int(round(negatives*train_frac/100))
    positive_samples_to_draw = int(round(positives*train_frac/100))

    # draw positive training samples
    training_positives = positive_samples.sample(n=positive_samples_to_draw)  # draw n samples
    training_positives.sort_index()  # sort samples
    ids_to_drop = training_positives.index.values  # get the ids of positive training samples
    testing_positives = positive_samples.drop(ids_to_drop)

    # draw negative training samples
    training_negatives = negative_samples.sample(n=negative_samples_to_draw)  # draw n samples
    training_negatives.sort_index()  # sort samples
    ids_to_drop = training_negatives.index.values  # get the ids of negative training samples
    testing_negatives = negative_samples.drop(ids_to_drop)

    training_data = training_negatives.append(training_positives)
    testing_data = testing_negatives.append(testing_positives)

    new_name = "_train_" + str(train_frac) + ".csv"
    filename = dataset_filename.replace(".csv", new_name)
    training_data.to_csv(filename, sep=";", index=False)

    new_name = "_test_" + str(100-train_frac) + ".csv"
    filename = dataset_filename.replace(".csv", new_name)
    testing_data.to_csv(filename, sep=";", index=False)

    return training_data, testing_data


def crossvalidation(dataset_filename, kfolds):

    dataset, negatives, positives = preproc.read_data(dataset_filename)
    header = dataset.columns.values.tolist()
    data_size = len(dataset.index)

    negative_samples = dataset.iloc[:negatives].copy()
    positive_samples = dataset.iloc[negatives:data_size].copy()

    negative_samples_to_draw = int(round(negatives/kfolds))

    positive_samples_to_draw = int(round(positives/kfolds))

    negative_folds = []
    positive_folds = []

    train_datasets = []
    test_datasets = []

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

        train_datasets.append(train_fold)
        test_datasets.append(test_folds)

    return train_datasets, test_datasets


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


    if isinstance(dataset_filename, pandas.DataFrame):
        dataset = dataset_filename.__copy__()
        samples = len(dataset_filename.index)
        negatives = dataset_filename[dataset_filename["Annots"] == 0].count()["Annots"]
        positives = samples - negatives
    else:
        # read data
        dataset, negatives, positives = preproc.read_data(dataset_filename)


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


def generate_parameters(tune_weights, bacc_lower, bacc_upper, bacc_step,
                        iter_lower, iter_upper, iter_step,
                        pop_lower, pop_upper, pop_step,
                        cp_lower, cp_upper, cp_step,
                        mp_lower, mp_upper, mp_step,
                        ts_lower, ts_upper, ts_step,
                        param_sets_numb):

    if tune_weights:
        bacc_weights = [i for i in range(bacc_lower, bacc_upper+1, bacc_step)]
    iterations = [i for i in range(iter_lower, iter_upper+1, iter_step)]
    population_size = [i for i in range(pop_lower, pop_upper+1, pop_step)]
    crossover_probability = [i/100 for i in range(cp_lower, cp_upper+1, cp_step)]
    mutation_probability = [i/100 for i in range(mp_lower, mp_upper+1, mp_step)]
    tournament_size = [i/100 for i in range(ts_lower, ts_upper+1, ts_step)]

    parameter_sets = []
    for i in range(0, param_sets_numb):
        if tune_weights:
            weight = bacc_weights[random.randrange(0, len(bacc_weights))]/100
        else:
            weight = bacc_lower
        iter = iterations[random.randrange(0, len(iterations))]
        pop = population_size[random.randrange(0, len(population_size))]
        cp = crossover_probability[random.randrange(0, len(crossover_probability))]
        mp = mutation_probability[random.randrange(0, len(mutation_probability))]
        ts = tournament_size[random.randrange(0, len(tournament_size))]
        while [iter, pop, cp, mp, ts] in parameter_sets:
            if tune_weights == True:
                weight = bacc_weights[random.randrange(0, len(bacc_weights))]
            iter = iterations[random.randrange(0, len(iterations))]
            pop = population_size[random.randrange(0, len(population_size))]
            cp = crossover_probability[random.randrange(0, len(crossover_probability))]
            mp = mutation_probability[random.randrange(0, len(mutation_probability))]
            ts = tournament_size[random.randrange(0, len(tournament_size))]
        parameter_sets.append([weight, iter, pop, cp, mp, ts])

    return parameter_sets


def generate_bash_scripts(train_list, test_list):

    parameter_sets = generate_parameters(25, 100, 25,
                                         50, 300, 50,
                                         10, 100, 10,
                                         10, 100, 10,
                                         10, 50, 10,
                                         100)
    run_commands = []

    for train, test in zip(train_list, test_list):
        for i in range(0, len(parameter_sets)):

            log_name = "log_" + train.replace(".csv", "") + "_" + str(parameter_sets[i][0]) + "_" \
                       + str(parameter_sets[i][1]) + "_" + str(parameter_sets[i][2]) + "_" \
                       + str(parameter_sets[i][3]) + "_" + str(parameter_sets[i][4])

            run_command = "python3 /home/mnowicka/run_GA.py" + \
                          " --train ./data/" + train + " --test ./data/" + test + \
                          " -i " + str(parameter_sets[i][0]) + " -p " + str(parameter_sets[i][1]) + \
                          " -c " + str(5) + " -a " + str(0.5) + " -x " + str(parameter_sets[i][2]) + \
                          " -m " + str(parameter_sets[i][3]) + " -t " + str(parameter_sets[i][4]) + \
                          " > " + log_name

            script = "#!/bin/bash\n#SBATCH -J " + log_name + "\n#SBATCH -D /data/scratch/mnowicka\n#SBATCH -o " \
                     + log_name + ".%j.out\n#SBATCH -n 1\n#SBATCH --time=5-00:00:00\n" \
                                  "#SBATCH --partition=big\n#SBATCH --mem=2000M\n#SBATCH --cpus-per-task=1\n\n" \
                                  "source /home/mnowicka/venv/bin/activate\n" + run_command

            script_name = "C:/Projects/DISCRETIZATION/SIMDATA/D2/bash_scripts/script_" + log_name + ".sh"
            script_file = open(script_name, 'w')
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




