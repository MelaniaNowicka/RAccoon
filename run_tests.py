from datetime import datetime
import configparser
import time
import argparse
import preproc
import sys
import random
import numpy
import os

import genetic_algorithm
import eval
import popinit
import tuner
import toolbox

numpy.random.seed(1)
random.seed(1)


# train and test classifiers
def train_and_test(data, parameter_set, classifier_size, evaluation_threshold, elitism, rules, uniqueness, repeats,
                   print_results):

    """

    Trains classifier on training data and tests on testing data.

    Parameters
    ----------
    data : list
        list including train data set, test data set and feature cdds list
    parameter_set : list
        list of genetic algorithm parameters (iterations, population size, crossover probability, mutation probability
        and tournament size)
    classifier_size : int
        maximal classifier size
    evaluation_threshold : float
        classifier evaluation threshold
    elitism : bool
        if True the best found solutions are added to the population in each selection operation
    rules : list
        list of pre-optimized rules
    uniqueness : bool
         if True only unique inputs in a classifier are counted, otherwise the input cdd score is multiplied by
         the number of input occurrences
    repeats : int
        number of single test repeats
    print_results : bool
        if True more information is shown

    Returns
    -------
    test_bacc_avg : float
        average test balanced accuracy

    """

    # parameter set
    weight, tc, pop, cp, mp, ts = parameter_set

    # unpack training data, testing data and feature cdds
    training_fold, testing_fold, feature_cdd_fold = data

    # lists of train scores
    train_score_avg = []
    train_bacc_avg = []
    train_tpr_avg = []
    train_tnr_avg = []
    train_fpr_avg = []
    train_fnr_avg = []
    train_f1_avg = []
    train_mcc_avg = []
    train_ppv_avg = []
    train_fdr_avg = []
    train_cdd_avg = []

    # lists of test scores
    test_bacc_avg = []
    test_tpr_avg = []
    test_tnr_avg = []
    test_fpr_avg = []
    test_fnr_avg = []
    test_f1_avg = []
    test_mcc_avg = []
    test_ppv_avg = []
    test_fdr_avg = []

    # lists of numbers of inputs and rules
    inputs_avg = []
    rules_avg = []

    print("\nTRAINING ON DATA FOLD...")

    train_runtimes = []  # training run-times
    update_number = []  # number of score updates

    classifier_list = []

    for i in range(0, repeats):  # repeat tests

        print("\nREPEAT: ", i+1)

        # measure time
        start_test = time.time()

        # run the algorithm
        classifier, best_classifiers, updates, first_global_best_score, first_avg_population_score \
            = genetic_algorithm.run_genetic_algorithm(train_data=training_fold,
                                                      filter_data=False, iterations=tc,
                                                      fixed_iterations=None,
                                                      population_size=pop,
                                                      elitism=elitism,
                                                      rules=rules,
                                                      popt_fraction=0,
                                                      classifier_size=classifier_size,
                                                      evaluation_threshold=evaluation_threshold,
                                                      feature_cdds=feature_cdd_fold,
                                                      crossover_probability=cp,
                                                      mutation_probability=mp,
                                                      tournament_size=ts,
                                                      bacc_weight=weight,
                                                      uniqueness=uniqueness,
                                                      print_results=print_results)

        # measure time
        end_test = time.time()

        classifier_list.append(classifier)

        train_runtimes.append(end_test-start_test)
        update_number.append(updates)

        # get annotation
        header = training_fold.columns.values.tolist()
        samples, annotation, negatives, positives = preproc.get_data_info(dataset=training_fold, header=header)

        # calculate best train BACC
        train_score, train_bacc, train_errors, train_error_rates, train_additional_scores, train_cdd = \
            eval.evaluate_classifier(classifier=classifier,
                                     dataset=training_fold,
                                     annotation=annotation,
                                     negatives=negatives,
                                     positives=positives,
                                     feature_cdds=feature_cdd_fold,
                                     uniqueness=uniqueness,
                                     bacc_weight=weight)

        print("TRAIN BACC: ", train_bacc)

        train_score_avg.append(train_score)
        train_bacc_avg.append(train_bacc)

        train_tpr_avg.append(train_error_rates["tpr"])
        train_tnr_avg.append(train_error_rates["tnr"])
        train_fpr_avg.append(train_error_rates["fpr"])
        train_fnr_avg.append(train_error_rates["fnr"])

        train_f1_avg.append(train_additional_scores["f1"])
        train_mcc_avg.append(train_additional_scores["mcc"])
        train_ppv_avg.append(train_additional_scores["ppv"])
        train_fdr_avg.append(train_additional_scores["fdr"])

        train_cdd_avg.append(train_cdd)

        # get annotation
        header = testing_fold.columns.values.tolist()
        samples, annotation, negatives, positives = preproc.get_data_info(testing_fold, header)

        # calculate best test BACC
        test_score, test_bacc, test_errors, test_error_rates, test_additional_scores, train_cdd = \
            eval.evaluate_classifier(classifier=classifier,
                                     dataset=testing_fold,
                                     annotation=annotation,
                                     negatives=negatives,
                                     positives=positives,
                                     feature_cdds=feature_cdd_fold,
                                     uniqueness=uniqueness,
                                     bacc_weight=weight)

        test_bacc_avg.append(test_bacc)

        test_tpr_avg.append(test_error_rates["tpr"])
        test_tnr_avg.append(test_error_rates["tnr"])
        test_fpr_avg.append(test_error_rates["fpr"])
        test_fnr_avg.append(test_error_rates["fnr"])

        test_f1_avg.append(test_additional_scores["f1"])
        test_mcc_avg.append(test_additional_scores["mcc"])
        test_ppv_avg.append(test_additional_scores["ppv"])
        test_fdr_avg.append(test_additional_scores["fdr"])

        print("TEST BACC: ", test_bacc)

        # show all found solutions
        if print_results is True:
            print("\n##ALL FOUND CLASSIFIERS##")
            for classifier_str in best_classifiers.solutions_str:
                print(classifier_str)

        # calculate classifier size
        number_of_inputs = len(classifier.get_input_list())
        number_of_rules = len(classifier.rule_set)

        inputs_avg.append(number_of_inputs)
        rules_avg.append(number_of_rules)

    if print_results:
        # rank features by frequency
        print("\n###FEATURE FREQUENCY ANALYSIS###")
        toolbox.rank_features_by_frequency(classifier_list)

        # average scores
        print("\n###AVERAGE SCORES###")

        # calculate train average scores
        print("\nTRAIN AVERAGE RESULTS")
        print("TRAIN AVG BACC: ", numpy.average(train_bacc_avg))
        print("TRAIN AVG STDEV: ", numpy.std(train_bacc_avg, ddof=1))
        print("TRAIN AVG CDD: ", numpy.average(train_cdd_avg))
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
        print("TEST AVG STDEV: ", numpy.std(test_bacc_avg, ddof=1))
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

        print("\nRUNTIME")
        print("RUN-TIME PER TRAINING: ", numpy.average(train_runtimes))
        print("UPDATES PER TRAINING:", numpy.average(update_number))

        print("CSV;", numpy.average(train_bacc_avg), ";", numpy.std(train_bacc_avg, ddof=1), ";",
              numpy.average(test_bacc_avg), ";", numpy.std(test_bacc_avg, ddof=1), ";",
              numpy.average(rules_avg), ";", numpy.average(inputs_avg))

    test_bacc_avg = numpy.average(test_bacc_avg)

    return test_bacc_avg


# run test
def run_test(train_data_file_name, test_data_file_name, rules, config_file_name):

    """

    Runs complex training including data pre-processing, parameter tuning and training.

    Parameters
    ----------
    train_data_file_name : str
        path to training data set
    test_data_file_name : str
        path to test data set
    rules : list
        list of pre-optimized rules
    config_file_name : str
        name of configuration file

    """

    # parse configuration file
    config_file = configparser.ConfigParser()
    config_file.read(config_file_name)

    # create new directory
    path_train = train_data_file_name
    head_tail = os.path.split(path_train)
    path_train = head_tail[0]
    file_name_train = head_tail[1]
    date = datetime.now()
    dir_name = date.strftime("%Y-%m-%d_%H-%M-%S")
    path = "/".join([path_train, dir_name])

    path_test = test_data_file_name
    head_tail = os.path.split(path_test)
    path_test = head_tail[0]
    file_name_test = head_tail[1]

    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " was created.\n")
    else:
        print("Directory ", path, " already exists.")

    toolbox.write_config_to_log(config_file)

    # READING/CREATING TRAINING AND TESTING DATA
    # create test data if not given
    if test_data_file_name is None:
        train_fraction = int(config_file['DATA DIVISION']['TrainingFraction'])
        set_seed = config_file.getboolean("DATA DIVISION", "SetSeed")

        # division into training and testing data
        print("###########READING DATA###########")
        print("\n***DIVISION INTO TRAINING AND TESTING DATA SETS***")
        training_data, testing_data = toolbox.divide_into_train_test(train_data_file_name, train_fraction, set_seed)

        # save to files
        new_name = "_train_" + str(train_fraction) + ".csv"
        new_name = file_name_train.replace(".csv", new_name)
        filename = "/".join([path, new_name])
        training_data.to_csv(filename, sep=";", index=False)

        new_name = "_test_" + str(100 - train_fraction) + ".csv"
        new_name = file_name_test.replace(".csv", new_name)
        filename = "/".join([path, new_name])
        testing_data.to_csv(filename, sep=";", index=False)

    else:
        print("###########READING DATA###########")
        # read data
        print("\nTRAIN DATA")
        training_data, train_annotation, train_positives, train_negatives, train_features = \
            preproc.read_data(train_data_file_name)
        print("\nTEST DATA")
        testing_data, test_annotation, test_positives, test_negatives, test_features = \
            preproc.read_data(test_data_file_name)

    # PARAMETER TUNING - CROSS-VALIDATION
    print("\n###########PARAMETER TUNING###########")

    print("\n***CROSSVALIDATION DATA DIVISION***")
    cv_folds = int(config_file['DATA DIVISION']['CVFolds'])
    pairing = config_file.getboolean("DATA DIVISION", "Pairing")
    set_seed = config_file.getboolean("DATA DIVISION", "SetSeed")

    training_cv_datasets, validation_cv_datasets = \
        toolbox.divide_into_cv_folds(dataset_file_name=train_data_file_name,
                                     path=path,
                                     dataset=training_data,
                                     k_folds=cv_folds,
                                     pairing=pairing,
                                     set_seed=set_seed)

    # discretize cv folds
    print("\n***DATA DISCRETIZATION***")
    m_segments = int(config_file["BINARIZATION PARAMETERS"]["MSegments"])
    alpha_bin = float(config_file["BINARIZATION PARAMETERS"]["AlphaBin"])
    lambda_bin = float(config_file["BINARIZATION PARAMETERS"]["LambdaBin"])

    training_cv_datasets_bin, validation_cv_datasets_bin, feature_cdds = \
        preproc.discretize_data_for_tests(training_fold_list=training_cv_datasets,
                                          validation_fold_list=validation_cv_datasets,
                                          m_segments=m_segments,
                                          alpha_param=alpha_bin,
                                          lambda_param=lambda_bin,
                                          print_results=False)

    classifier_size = int(config_file["CLASSIFIER PARAMETERS"]["ClassifierSize"])
    set_alpha = config_file.getboolean("CLASSIFIER PARAMETERS", "SetAlpha")

    if set_alpha is True:
        evaluation_threshold = float(config_file["CLASSIFIER PARAMETERS"]["Alpha"])
    else:
        evaluation_threshold = None

    elitism = config_file.getboolean("ALGORITHM PARAMETERS", "Elitism")
    uniqueness = config_file.getboolean("OBJECTIVE FUNCTION", "Uniqueness")
    test_repeats = int(config_file["RUN PARAMETERS"]["SingleTestRepeats"])

    # read rules from file
    if rules is not None:
        rules = popinit.read_rules_from_file(rules)

    # remove irrelevant features
    training_cv_datasets_bin_filtered = []
    for train_set in training_cv_datasets_bin:

        train_set_filtered, features = preproc.remove_irrelevant_features(train_set)
        training_cv_datasets_bin_filtered.append(train_set_filtered)

    # save to files
    fold = 1
    for train_set, val_set in zip(training_cv_datasets_bin_filtered, validation_cv_datasets_bin):

        new_name = "_cv_train_" + str(fold) + "_bin.csv"
        new_name = file_name_train.replace(".csv", new_name)
        filename = "/".join([path, new_name])
        train_set.to_csv(filename, sep=";", index=False)

        new_name = "_cv_val_" + str(fold) + "_bin.csv"
        new_name = file_name_test.replace(".csv", new_name)
        filename = "/".join([path, new_name])
        val_set.to_csv(filename, sep=";", index=False)

        fold = fold + 1

    # parameter tuning
    print("\n***PARAMETER TUNING***")
    best_parameters, best_bacc, best_std = tuner.tune_parameters(training_cv_datasets=training_cv_datasets_bin_filtered,
                                                                 validation_cv_datasets=validation_cv_datasets_bin,
                                                                 feature_cdds=feature_cdds,
                                                                 config_file=config_file,
                                                                 classifier_size=classifier_size,
                                                                 evaluation_threshold=evaluation_threshold,
                                                                 elitism=elitism,
                                                                 uniqueness=uniqueness,
                                                                 rules=rules,
                                                                 repeats=test_repeats)

    w, tc, ps, cp, mp, ts = best_parameters

    print("\n##BEST PARAMETERS##")
    print("WEIGHT: ", w, " TC: ", tc, " PS: ", ps, " CP: ", cp, " MP: ", mp, " TS: ", ts)
    print("BEST SCORE: ", best_bacc, " STD: ", best_std)

    print("\n###########FINAL TEST###########")
    print("\n***DATA DISCRETIZATION***")
    # binarize training and testing data sets
    discretized_train_data, discretized_test_data, feature_cdds = \
        preproc.discretize_data_for_tests(training_fold_list=[training_data],
                                          validation_fold_list=[testing_data],
                                          m_segments=m_segments,
                                          alpha_param=alpha_bin,
                                          lambda_param=lambda_bin,
                                          print_results=True)

    new_name = "_train_bin.csv"
    new_name = file_name_train.replace(".csv", new_name)
    filename_train = "/".join([path, new_name])
    discretized_train_data[0].to_csv(filename_train, sep=";", index=False)

    #remove irrelevant miRNAs
    discretized_train_data_filtered, relevant_features = preproc.remove_irrelevant_features(discretized_train_data[0])

    # save to files
    new_name = "_train_bin_filtered.csv"
    new_name = file_name_train.replace(".csv", new_name)
    filename_train = "/".join([path, new_name])
    discretized_train_data_filtered.to_csv(filename_train, sep=";", index=False)

    new_name = "_test_bin.csv"
    new_name = file_name_test.replace(".csv", new_name)
    filename_test = "/".join([path, new_name])
    discretized_test_data[0].to_csv(filename_test, sep=";", index=False)

    # train and test
    print("\n***RUN ALGORITHM***")
    w, tc, ps, cp, mp, ts = best_parameters
    print("PARAMETERS:")
    print("WEIGHT: ", w, ", TC: ", tc, ", PS: ", ps, ", CP: ", cp, ", MP: ", mp, ", TS: ", ts)
    print("EVALUATION THRESHOLD: ", evaluation_threshold)
    print("SINGLE TEST REPEATS: ", test_repeats, "\n")

    #run test
    train_and_test(data=(discretized_train_data_filtered, discretized_test_data[0], feature_cdds[0]),
                   parameter_set=best_parameters,
                   classifier_size=classifier_size,
                   evaluation_threshold=evaluation_threshold,
                   elitism=elitism,
                   rules=rules,
                   uniqueness=uniqueness,
                   repeats=test_repeats,
                   print_results=True)


if __name__ == "__main__":

    start_global = time.time()

    print('A genetic algorithm (GA) optimizing a set of miRNA-based distributed cell classifiers \n'
          'for in situ cancer classification. Written by Melania Nowicka, FU Berlin, 2019.\n')

    print("Log date: ", datetime.now().isoformat(), "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '--dataset-filename-train',
                        dest="dataset_filename_train", help='train data set file name')
    parser.add_argument('--test', '--dataset-filename-test', default=None,
                        dest="dataset_filename_test", help='test data set file name')
    parser.add_argument('--rules', '--rule-file', type=str, default=None,
                        dest="rule_file", help='rules file name')
    parser.add_argument('--config', '--config_filename',
                        dest="config_filename", help='config file name')

    params = parser.parse_args(sys.argv[1:])

    dataset_train = params.dataset_filename_train
    dataset_test = params.dataset_filename_test
    rule_list = params.rule_file
    config_filename = params.config_filename

    run_test(dataset_train, dataset_test, rule_list, config_filename)

    end_global = time.time()
    print("TIME (FULL TEST): ", end_global - start_global)

