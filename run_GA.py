'''
A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import sys
import random
import time
import argparse
import numpy

import genetic_algorithm
import preproc
import eval

numpy.random.seed(1)
random.seed(1)


# algorithm parameters read from the command line
def parse_parameters(args):

    """

    Parses command line parameters.

    Parameters
    ----------
    args : list
        list of command line arguments

    """

    # parameter parser
    parser = argparse.ArgumentParser(description='A genetic algorithm (GA) optimizing a set of miRNA-based cell '
                                                 'classifiers for in situ cancer classification. Written by Melania '
                                                 'Nowicka, FU Berlin, 2019.\n\n')

    # adding arguments
    parser.add_argument('--train', '--dataset-filename-train', dest="dataset_filename_train",
                        help='train data set file name')
    parser.add_argument('--test', '--dataset-filename-test', dest="dataset_filename_test", default=None,
                        help=' test data set file name')
    parser.add_argument('--filter', '--filter-data', dest="filter_data", type=str, default='t',
                        help='filter data')
    parser.add_argument('--discretize', '--discretize-data', dest="discretize_data", type=str, default='t',
                        help='discretize data')
    parser.add_argument('--mbin', '--m-bin', dest="m_bin", type=int, default=50,
                        help='m segments')
    parser.add_argument('--abin', '--a-bin', dest="a_bin", type=float, default=0.5,
                        help='binarization alpha')
    parser.add_argument('--lbin', '--l-bin', dest="l_bin", type=float, default=0.1,
                        help='binarization lambda')
    parser.add_argument('-c', '--classifier-size', dest="classifier_size", type=int, default=5,
                        help='classifier size')
    parser.add_argument('-a', '--evaluation-threshold', dest="evaluation_threshold", default=None,
                        help='evaluation threshold alpha')
    parser.add_argument('-w', '--bacc-weight', dest="bacc_weight", default=0.5, type=float,
                        help='multi-objective function weight')
    parser.add_argument('-u', '--uniqueness', dest="uniqueness", default='t', type=str,
                        help='uniqueness of inputs')
    parser.add_argument('-i', '--iterations', dest="iterations", type=int, default=30,
                        help='number of iterations without improvement')
    parser.add_argument('-f', '--fixed-iterations', dest="fixed_iterations", type=int, default=None,
                        help='fixed number of iterations')
    parser.add_argument('-p', '--population-size', dest="population_size", type=int, default=300,
                        help='population size')
    parser.add_argument('--elitism', '--elitism', dest="elitism", type=float, default=1,
                        help='copy fraction of current best solutions to the population')
    parser.add_argument('--rules', '--rules', dest="rule_list", type=str, default=None,
                        help='list of pre-optimized rules')
    parser.add_argument('--poptfrac', '--p-opt-frac', dest="p_opt_frac", type=float, default=0.5,
                        help='pre-optimized fraction of population')
    parser.add_argument('-x', '--crossover-probability', dest="crossover_probability", default=0.8, type=float,
                        help='probability of crossover')
    parser.add_argument('-m', '--mutation-probability', dest="mutation_probability", default=0.1, type=float,
                        help='probability of mutation')
    parser.add_argument('-t', '--tournament-size', dest="tournament_size", default=0.2, type=float,
                        help='tournament size')

    # parse arguments
    params = parser.parse_args(args)

    parameters = [params.dataset_filename_train, params.dataset_filename_test, params.filter_data,
                  params.discretize_data, params.m_bin, params.a_bin, params.l_bin,
                  params.classifier_size, params.evaluation_threshold, params.bacc_weight,
                  params.uniqueness, params.iterations, params.fixed_iterations, params.population_size,
                  params.elitism, params.rule_list, params.p_opt_frac, params.crossover_probability,
                  params.mutation_probability, params.tournament_size]

    return parameters


# process parameters and data and run algorithm
def process_and_run(args):

    """

    Processes data and parameters and runs the algorithm.

    Parameters
    ----------
    args : list
        list of command line arguments

    Returns
    -------
    train_bacc : float
        training balanced accuracy
    test_bacc : float
        test balanced accuracy
    updates : int
        number of best score updates
    training_time : float
        training time
    first_global : float
        first global best score
    first_avg_pop : float
        first population average score

    """

    parameters = parse_parameters(args)

    # process parameters
    train_datafile, test_datafile, filter_data, discretize_data, m_bin, a_bin, l_bin, classifier_size, \
        evaluation_threshold, bacc_weight, uniqueness, iterations, fixed_iterations, population_size, elitism, \
        rule_list, popt_fraction, crossover_probability, mutation_probability, tournament_size = parameters

    print("##PARAMETERS##")
    if filter_data == 't':
        print("FILTERING: ", "on")
        filter_data = True
    else:
        print("FILTERING: ", "off")
        filter_data = False
    if discretize_data == 't':
        print("DISCRETIZE: ", "on")
        print("DISCRETIZATION M: ", m_bin)
        print("DISCRETIZATION ALPHA: ", a_bin)
        print("DISCRETIZATION LAMBDA: ", l_bin)
    else:
        print("DISCRETIZE: ", "off")
    print("EVALUATION THRESHOLD: ", evaluation_threshold)
    print("MAX SIZE: ", classifier_size)
    print("WEIGHT: ", bacc_weight)
    if uniqueness == 't':
        print("UNIQUENESS: ", "on")
        uniqueness = True
    else:
        print("UNIQUENESS: ", "off")
        uniqueness = False

    if rule_list is not None:
        print("POPULATION PRE-OPTIMIZATION: ", "on")
        print("POPULATION PRE-OPTIMIZED FRACTION: ", popt_fraction)
    print("GA PARAMETERS: ", "TC: ", iterations, ", PS: ", population_size, ", CP: ", crossover_probability, ", MP: ", \
          mutation_probability, ", TS: ", tournament_size)

    print("\n##TRAIN DATA##")
    # read the data
    train_dataset, annotation, negatives, positives, features = preproc.read_data(train_datafile)
    annotation = train_dataset["Annots"]

    # discretize data
    if discretize_data == 't':
        print("\n##DISCRETIZATION##")
        data_discretized, features, thresholds, feature_cdds = \
            preproc.discretize_train_data(train_dataset, m_bin, a_bin, l_bin, True)
    else:
        data_discretized = train_dataset
        feature_cdds = {}
        bacc_weight = 1.0

    print("\nTRAINING...")
    start_train = time.time()
    classifier, best_classifiers, updates, first_best_score, first_avg_pop = \
        genetic_algorithm.run_genetic_algorithm(data_discretized, filter_data, iterations, fixed_iterations,
                                                population_size, elitism, rule_list, popt_fraction, classifier_size,
                                                evaluation_threshold, feature_cdds, crossover_probability,
                                                mutation_probability, tournament_size, bacc_weight, uniqueness, True)

    end_train = time.time()
    training_time = end_train - start_train
    print("TRAINING TIME: ", end_train - start_train)

    # evaluate best classifier
    classifier_score, train_bacc, errors, train_error_rates, train_additional_scores, cdd_score = \
        eval.evaluate_classifier(classifier, data_discretized, annotation, positives, negatives, feature_cdds,
                                 uniqueness, bacc_weight)

    print("\n##TRAIN DATA SCORES##")
    print("BACC: ", train_bacc)
    print("CDD SCORE: ", cdd_score)
    print("TPR: ", train_error_rates["tpr"])
    print("TNR: ", train_error_rates["tnr"])
    print("FNR: ", train_error_rates["fpr"])
    print("FPR: ", train_error_rates["fnr"])

    if test_datafile is not None:

        print("\n##TEST DATA##")
        # read test data
        test_dataset, annotation, negatives, positives, features = preproc.read_data(test_datafile)
        annotation = test_dataset["Annots"]

        # discretize data
        if discretize_data == 't':
            print("\n##DISCRETIZATION##")
            data_discretized = preproc.discretize_test_data(test_dataset, thresholds)
        else:
            data_discretized = test_dataset
            feature_cdds = {}
            bacc_weight = 1.0

        # evaluate classifier
        classifier_score, test_bacc, errors, test_error_rates, test_additional_scores, cdd_score = \
            eval.evaluate_classifier(classifier, data_discretized, annotation, positives, negatives, feature_cdds,
                                     uniqueness, bacc_weight)

        print("\n##TEST DATA SCORES##")
        print("BACC: ", test_bacc)
        print("CDD SCORE: ", cdd_score)
        print("TPR: ", test_error_rates["tpr"])
        print("TNR: ", test_error_rates["tnr"])
        print("FNR: ", test_error_rates["fpr"])
        print("FPR: ", test_error_rates["fnr"])

    else:
        test_bacc = None

    return train_bacc, test_bacc, updates, training_time, first_best_score, first_avg_pop


# repeat GA run
def repeat(repeats, args):

    """

    Processes data and parameters and runs the algorithm.

    Parameters
    ----------
    repeats : int
        number of single test repeats
    args : list
        list of command line arguments

    """

    train_scores = []
    test_scores = []
    time = []
    updates_list = []
    first_scores = []
    first_avg_population_scores = []

    test_bacc = None

    for i in range(0, repeats):
        print("\nREPEAT ", i+1)
        train_bacc, test_bacc, updates, train_time, first_best_score, first_avg_pop = process_and_run(args)
        train_scores.append(train_bacc)
        test_scores.append(test_bacc)
        time.append(train_time)
        updates_list.append(updates)
        first_scores.append(first_best_score)
        first_avg_population_scores.append(first_avg_pop)

    print("\nRESULTS")
    print("AVG TRAIN: ", numpy.average(train_scores), " STDEV: ", numpy.std(train_scores, ddof=1))
    if test_bacc is not None:
        print("AVG TEST: ", numpy.average(test_scores), " STDEV: ", numpy.std(test_scores, ddof=1))
    print("AVG UPDATES: ", numpy.average(updates_list), " STDEV: ", numpy.std(updates_list, ddof=1))
    print("AVG TRAINING TIME: ", numpy.average(time), " STDEV: ", numpy.std(time, ddof=1))
    print("AVG FIRST BEST SCORE: ", numpy.average(first_scores), " STDEV: ", numpy.std(first_scores, ddof=1))
    print("AVG INITIAL POPULATION SCORE: ", numpy.average(first_avg_population_scores), " STDEV: ",
          numpy.std(first_avg_population_scores, ddof=1))

    if test_bacc is not None:
        print("CSV;", numpy.average(train_scores), ";", numpy.std(train_scores, ddof=1), ";",
              numpy.average(test_scores), ";", numpy.std(test_scores, ddof=1), ";",
              numpy.average(updates_list), ";", numpy.std(updates_list, ddof=1), ";",
              numpy.average(time), ";", numpy.std(time, ddof=1), ";",
              numpy.average(first_scores), ";", numpy.std(first_scores, ddof=1), ";",
              numpy.average(first_avg_population_scores), ";", numpy.std(first_avg_population_scores, ddof=1))
    else:
        print("CSV;", numpy.average(train_scores), ";", numpy.std(train_scores, ddof=1), ";",
              numpy.average(updates_list), ";", numpy.std(updates_list, ddof=1), ";",
              numpy.average(time), ";", numpy.std(time, ddof=1), ";",
              numpy.average(first_scores), ";", numpy.std(first_scores, ddof=1), ";",
              numpy.average(first_avg_population_scores), ";", numpy.std(first_avg_population_scores, ddof=1))


if __name__ == "__main__":

    start = time.time()

    print('A genetic algorithm (GA) optimizing a set of miRNA-based distributed cell classifiers \n'
          'for in situ cancer classification. Written by Melania Nowicka, FU Berlin, 2019.\n')

    repeat(10, sys.argv[1:])

    end = time.time()
    print("TIME: ", end - start)
