import random
import configparser
import numpy
import time
import argparse
import preproc
import sys

import toolbox
import run_GA
import eval

# divide data into train and test
def divide_into_train_test(dataset_filename, train_frac):

    dataset, negatives, positives = toolbox.read_data(dataset_filename)
    header = dataset.columns.values.tolist()

    data_size = len(dataset.index)

    negative_samples = dataset.iloc[:negatives].copy()
    positive_samples = dataset.iloc[negatives:data_size].copy()

    # create training set
    negative_samples_to_draw = int(round(negatives*train_frac/100))
    positive_samples_to_draw = int(round(positives*train_frac/100))

    print("TRAINING DATA SET: ", positive_samples_to_draw+negative_samples_to_draw, " samples")
    print("TESTING DATA SET: ", data_size - (positive_samples_to_draw + negative_samples_to_draw), " samples")

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

    return training_data, testing_data


def crossvalidation(dataset, dataset_filename, kfolds):

    samples = len(dataset.index)
    negatives = dataset[dataset["Annots"] == 0].count()["Annots"]
    positives = samples - negatives

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

        test_datasets.append(test_fold)

    for fold in range(0, kfolds):

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

        train_datasets.append(train_fold)

    return train_datasets, test_datasets


# train and test classifiers
def train_and_test(training_fold, testing_fold, parameter_set, evaluation_threshold, repeats):

    # parameter set
    iter, pop, cp, mp, ts = parameter_set

    train_bacc_avg = []
    train_tpr_avg = []
    train_tnr_avg = []
    train_fpr_avg = []
    train_fnr_avg = []
    train_f1_avg = []
    train_mcc_avg = []
    train_ppv_avg = []
    train_fdr_avg = []
    test_bacc_avg = []
    test_tpr_avg = []
    test_tnr_avg = []
    test_fpr_avg = []
    test_fnr_avg = []
    test_f1_avg = []
    test_mcc_avg = []
    test_ppv_avg = []
    test_fdr_avg = []

    inputs_avg = []
    rules_avg = []

    # repeat tests
    for i in range(0, repeats):

        print("REPEAT: ", i+1)
        start = time.time() # measure time

        # run the algorithm
        classifier, best_classifiers = run_GA.run_genetic_algorithm(training_fold, True, iter, pop, 5, 0.5, cp, mp, ts)

        # calculate best train BACC
        train_bacc, train_error_rates, train_additional_scores = \
            eval.evaluate_classifier(classifier, evaluation_threshold, training_fold)
        train_bacc_avg.append(train_bacc)
        train_tpr_avg.append(train_error_rates["tpr"])
        train_tnr_avg.append(train_error_rates["tnr"])
        train_fpr_avg.append(train_error_rates["fpr"])
        train_fnr_avg.append(train_error_rates["fnr"])
        train_f1_avg.append(train_additional_scores["f1"])
        train_mcc_avg.append(train_additional_scores["mcc"])
        train_ppv_avg.append(train_additional_scores["ppv"])
        train_fdr_avg.append(train_additional_scores["fdr"])

        # calculate best test BACC
        test_bacc, test_error_rates, test_additional_scores = \
            eval.evaluate_classifier(classifier, evaluation_threshold, testing_fold)

        test_bacc_avg.append(test_bacc)
        test_tpr_avg.append(test_error_rates["tpr"])
        test_tnr_avg.append(test_error_rates["tnr"])
        test_fpr_avg.append(test_error_rates["fpr"])
        test_fnr_avg.append(test_error_rates["fnr"])
        test_f1_avg.append(test_additional_scores["f1"])
        test_mcc_avg.append(test_additional_scores["mcc"])
        test_ppv_avg.append(test_additional_scores["ppv"])
        test_fdr_avg.append(test_additional_scores["fdr"])

        # calculate classifier size
        inputs = 0
        for rule in classifier.rule_set:
            for input in rule.pos_inputs:
                inputs = inputs + 1
            for input in rule.neg_inputs:
                inputs = inputs + 1
        rules = len(classifier.rule_set)

        inputs_avg.append(inputs)
        rules_avg.append(rules)

    # average scores
    print("###AVERAGE SCORES###")
    print("PARAMETER SET: ", parameter_set)
    print("EVALUATION THRESHOLD: ", evaluation_threshold)

    # calculate train average scores
    print("\nTRAIN AVERAGE RESULTS")
    print("TRAIN AVG BACC: ", numpy.average(train_bacc_avg))
    print("TRAIN AVG STDEV: ", numpy.std(train_bacc_avg))
    print("TRAIN AVG TPR: ", numpy.average(train_tpr_avg))
    print("TRAIN AVG TNR: ", numpy.average(train_tnr_avg))
    print("TRAIN AVG FPR: ", numpy.average(train_fpr_avg))
    print("TRAIN AVG FNR: ", numpy.average(train_fnr_avg))
    print("TRAIN AVG F1: ", numpy.average(train_f1_avg))
    print("TRAIN AVG MCC: ", numpy.average(train_mcc_avg))
    print("TRAIN AVG PPV: ", numpy.average(train_ppv_avg))
    print("TRAIN AVG FDR: ", numpy.average(train_fdr_avg))

    # calculate test average scores
    print("\nTEST AVERAGE RESULTS")
    print("TEST AVG BACC: ", numpy.average(test_bacc_avg))
    print("TEST AVG STDEV: ", numpy.std(test_bacc_avg))
    print("TEST AVG TPR: ", numpy.average(test_tpr_avg))
    print("TEST AVG TNR: ", numpy.average(test_tnr_avg))
    print("TEST AVG FPR: ", numpy.average(test_fpr_avg))
    print("TEST AVG FNR: ", numpy.average(test_fnr_avg))
    print("TEST AVG F1: ", numpy.average(test_f1_avg))
    print("TEST AVG MCC: ", numpy.average(test_mcc_avg))
    print("TEST AVG PV: ", numpy.average(test_ppv_avg))
    print("TEST AVG FDR: ", numpy.average(test_fdr_avg))

    # calculate size averages
    print("\nAVERAGE SIZE")
    print("AVERAGE NUMBER OF INPUTS: ", numpy.average(inputs_avg))
    print("AVERAGE NUMBER OF RULES: ", numpy.average(rules_avg))
    print("MEDIAN OF INPUTS: ", numpy.median(inputs_avg))
    print("MEDIAN OF RULES: ", numpy.median(rules_avg))

    test_bacc_avg = numpy.average(test_bacc_avg)
    test_std_avg = numpy.std(test_bacc_avg)

    return test_bacc_avg, test_std_avg

# parameter tuning
def tune_parameters(training_cv_datasets, testing_cv_datasets, config, evaluation_threshold, test_repeats):

    # get the parameters from configuration file
    iteration_lower_bound = int(config['PARAMETER TUNING']['IterationLowerBound'])
    iteration_upper_bound = int(config['PARAMETER TUNING']['IterationUpperBound'])
    iteration_step = int(config['PARAMETER TUNING']['IterationStep'])
    population_lower_bound = int(config['PARAMETER TUNING']['PopulationLowerBound'])
    population_upper_bound = int(config['PARAMETER TUNING']['PopulationUpperBound'])
    population_step = int(config['PARAMETER TUNING']['PopulationStep'])
    crossover_lower_bound = int(config['PARAMETER TUNING']['CrossoverLowerBound'])
    crossover_upper_bound = int(config['PARAMETER TUNING']['CrossoverUpperBound'])
    crossover_step = int(config['PARAMETER TUNING']['CrossoverStep'])
    mutation_lower_bound = int(config['PARAMETER TUNING']['MutationLowerBound'])
    mutation_upper_bound = int(config['PARAMETER TUNING']['MutationUpperBound'])
    mutation_step = int(config['PARAMETER TUNING']['MutationStep'])
    tournament_lower_bound = int(config['PARAMETER TUNING']['TournamentLowerBound'])
    tournament_upper_bound = int(config['PARAMETER TUNING']['TournamentUpperBound'])
    tournament_step = int(config['PARAMETER TUNING']['TournamentStep'])
    number_of_sets = int(config['PARAMETER TUNING']['NumberOfSets'])

    # generate parameter sets
    parameter_sets = toolbox.generate_parameters(iteration_lower_bound,
                                                 iteration_upper_bound,
                                                 iteration_step,
                                                 population_lower_bound,
                                                 population_upper_bound,
                                                 population_step,
                                                 crossover_lower_bound,
                                                 crossover_upper_bound,
                                                 crossover_step,
                                                 mutation_lower_bound,
                                                 mutation_upper_bound,
                                                 mutation_step,
                                                 tournament_lower_bound,
                                                 tournament_upper_bound,
                                                 tournament_step,
                                                 number_of_sets)

    # test parameter sets
    best_set = parameter_sets[0]
    best_avg_test_bacc = 0.0
    best_avg_test_std = 1.0

    # list of test bacc scores and standard deviation
    test_bacc_cv = []
    test_std_cv = []

    # iterate over parameter sets
    for parameter_set in parameter_sets:

        print("TESTING PARAMETER SET: ", parameter_set)

        fold = 1

        # iterate over folds
        for training_fold, testing_fold in zip(training_cv_datasets, testing_cv_datasets):

            print("\nTESTING FOLD: ", fold)
            fold = fold + 1

            # train and test classifiers
            test_bacc, test_std = train_and_test(training_fold, testing_fold, parameter_set,
                                             evaluation_threshold, test_repeats)

            test_bacc_cv.append(test_bacc)
            test_std_cv.append(test_std)

        # calculate average bacc scores
        test_bacc_avg = sum(test_bacc_cv)/len(test_bacc_cv)
        test_std_avg = sum(test_std_cv)/len(test_std_cv)

        # improvement check
        if test_bacc_avg > best_avg_test_bacc:
            best_set = parameter_set
            best_avg_test_bacc = test_bacc_avg
            best_avg_test_std = test_std_avg

            if test_bacc_avg == best_avg_test_bacc and test_std_avg > best_avg_test_std:
                best_set = parameter_set
                best_avg_test_bacc = test_bacc_avg
                best_avg_test_std = test_std_avg

    return best_set, best_avg_test_bacc, best_avg_test_std


def run_test(train_dataset_filename, test_dataset_filename):

    # parse configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    #train_frac = int(config['DATA DIVISION']['TrainingFraction'])

    # division into training and testing data
    #print("###########TESTS###########")
    #print("\n***DIVISION INTO TRAINING AND TESTING DATA SETS***")
    #training_data, testing_data = divide_into_train_test(dataset_filename, train_frac)

    # save to files
    #new_name = "_train_" + str(train_frac) + ".csv"
    #filename = dataset_filename.replace(".csv", new_name)
    #training_data.to_csv(filename, sep=";", index=False)

    #new_name = "_test_" + str(100 - train_frac) + ".csv"
    #filename = dataset_filename.replace(".csv", new_name)
    #testing_data.to_csv(filename, sep=";", index=False)

    print("###########DATA###########")
    #read the data
    print("\nTRAIN DATA")
    training_data, train_positives, train_negatives = toolbox.read_data(train_dataset_filename)
    print("\nTEST DATA")
    testing_data, test_positives, test_negatives = toolbox.read_data(test_dataset_filename)

    # 10-fold cross-validation on the training data
    print("\n###########PARAMETER TUNING###########")
    print("\n***CROSSVALIDATION DATA DIVISION***")
    cv_folds = int(config['DATA DIVISION']['CVFolds'])
    training_cv_datasets, testing_cv_datasets = crossvalidation(training_data, train_dataset_filename, cv_folds)

    # save to files
    fold = 1
    for train_set, test_set in zip(training_cv_datasets, testing_cv_datasets):

        new_name = "_train_" + str(fold) + ".csv"
        filename = train_dataset_filename.replace(".csv", new_name)
        train_set.to_csv(filename, sep=";", index=False)

        new_name = "_test_" + str(fold) + ".csv"
        filename = train_dataset_filename.replace(".csv", new_name)
        test_set.to_csv(filename, sep=";", index=False)

        fold = fold + 1

    print("\n***DATA DISCRETIZATION***")
    m_segments = int(config["BINARIZATION PARAMETERS"]["MSegments"])
    alpha_bin = float(config["BINARIZATION PARAMETERS"]["AlphaBin"])
    lambda_bin = float(config["BINARIZATION PARAMETERS"]["LambdaBin"])

    # binarize cv data sets
    training_cv_datasets_bin, testing_cv_datasets_bin = preproc.discretize_data_for_tests(training_cv_datasets,
                                                                                  testing_cv_datasets,
                                                                                  m_segments,
                                                                                  alpha_bin,
                                                                                  lambda_bin)

    # save to files
    fold = 1
    for train_set, test_set in zip(training_cv_datasets_bin, testing_cv_datasets_bin):

        new_name = "_train_" + str(fold) + "_bin.csv"
        filename = train_dataset_filename.replace(".csv", new_name)
        train_set.to_csv(filename, sep=";", index=False)

        new_name = "_test_" + str(fold) + "_bin.csv"
        filename = train_dataset_filename.replace(".csv", new_name)
        test_set.to_csv(filename, sep=";", index=False)

        fold = fold + 1

    evaluation_threshold = float(config["THRESHOLD"]["Alpha"])
    test_repeats = int(config["GENERAL"]["SingleTestRepeats"])

    # parameter tuning
    print("\n***PARAMETER TUNING***")
    best_parameters, best_bacc, best_std = tune_parameters(training_cv_datasets_bin,
                                                           testing_cv_datasets_bin,
                                                           config,
                                                           evaluation_threshold,
                                                           test_repeats)

    print("BEST PARAMETERS: ", best_parameters)
    print("BEST SCORE: ", best_bacc, " STD: ", best_std)

    print("\n###########FINAL TEST###########")
    print("\n***DATA DISCRETIZATION***")
    # binarize training and testing data sets
    discretized_train_data, discretized_test_data = preproc.discretize_data_for_tests([training_data],
                                                                              [testing_data],
                                                                              m_segments,
                                                                              alpha_bin,
                                                                              lambda_bin)
    # save to files
    new_name = "_train_bin.csv"
    filename_train = train_dataset_filename.replace(".csv", new_name)
    discretized_train_data[0].to_csv(filename_train, sep=";", index=False)

    new_name = "_test_bin.csv"
    filename_test = test_dataset_filename.replace(".csv", new_name)
    discretized_test_data[0].to_csv(filename_test, sep=";", index=False)

    # train and test
    print("\n***RUN ALGORITHM***")
    print("PARAMETERS: ", best_parameters)
    print("EVALUATION THRESHOLD: ", evaluation_threshold)
    print("SINGLE TEST REPEATS: ", test_repeats)
    # measure time
    start_test = time.time()
    #run test
    train_and_test(discretized_train_data[0], discretized_test_data[0], best_parameters, evaluation_threshold,
                   test_repeats)
    # measure time
    end_test = time.time()
    print("\nRUNTIME")
    print("TIME (AVG, ONE TEST): ", (end_test - start_test)/test_repeats)


if __name__ == "__main__":

    random.seed(1)

    start_global = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '--dataset-filename-train',
                        dest="dataset_filename_train", help='data set file name')
    parser.add_argument('--test', '--dataset-filename-test',
                        dest="dataset_filename_test", help='data set file name')

    params = parser.parse_args(sys.argv[1:])

    dataset_train = params.dataset_filename_train
    dataset_test = params.dataset_filename_test

    run_test(dataset_train, dataset_test)

    end_global = time.time()
    print("TIME (FULL TEST): ", end_global - start_global)

