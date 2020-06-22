'''
A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import sys
import random
import pandas
import preproc
import popinit
import eval
import selection
import crossover
import mutation
import log
import time
import argparse
import numpy

numpy.random.seed(1)
random.seed(1)

# algorithm parameters read from the command line
def check_params(args):

    # parameter parser
    parser = argparse.ArgumentParser(description='A genetic algorithm (GA) optimizing a set of miRNA-based cell '
                                                 'classifiers for in situ cancer classification. Written by Melania '
                                                 'Nowicka, FU Berlin, 2019.\n\n')

    # adding arguments
    parser.add_argument('--train', '--dataset-filename-train', dest="dataset_filename_train", help='data set file name')
    parser.add_argument('--test', '--dataset-filename-test', dest="dataset_filename_test", default=None, help='data set file name')
    parser.add_argument('--filter', '--filter-data', dest="filter_data", type=str, default='t', help='filter data')
    parser.add_argument('--discretize', '--discretize-data', dest="discretize_data", type=str, default='t', help='discretize data')
    parser.add_argument('--mbin', '--m-bin', dest="m_bin", type=int, default=50, help='m segments')
    parser.add_argument('--abin', '--a-bin', dest="a_bin", type=float, default=0.5, help='binarization alpha')
    parser.add_argument('--lbin', '--l-bin', dest="l_bin", type=float, default=0.1, help='binarization lambda')
    parser.add_argument('-c', '--classifier-size', dest="classifier_size", type=int, default=5, help='classifier size')
    parser.add_argument('-a', '--evaluation-threshold', dest="evaluation_threshold", default=0.45, type=float, help='evaluation threshold alpha')
    parser.add_argument('-w', '--bacc-weight', dest="bacc_weight", default=0.5, type=float, help='bacc_weight')
    parser.add_argument('-i', '--iterations', dest="iterations", type=int, default=30, help='number of iterations without improvement')
    parser.add_argument('-f', '--fixed-iterations', dest="fixed_iterations", type=int, default=None, help='fixed number of iterations')
    parser.add_argument('-p', '--population-size', dest="population_size", type=int, default=300, help='population size')
    parser.add_argument('--rules', '--rules', dest="rule_list", type=str, default=None,
                        help='List of pre-optimized rules')
    parser.add_argument('--poptfrac', '--p-opt-frac', dest="p_opt_frac", type=float, default=0.5,
                        help='List of pre-optimized rules')
    parser.add_argument('-x', '--crossover-probability', dest="crossover_probability", default=0.8, type=float, help='probability of crossover')
    parser.add_argument('-m', '--mutation-probability', dest="mutation_probability", default=0.1, type=float, help='probability of mutation')
    parser.add_argument('-t', '--tournament-size', dest="tournament_size", default=0.2, type=float, help='tournament size')

    # parse arguments
    params = parser.parse_args(args)

    return params.dataset_filename_train, params.dataset_filename_test, params.filter_data, \
           params.discretize_data, params.m_bin, params.a_bin, params.l_bin, \
           params.classifier_size, params.evaluation_threshold, params.bacc_weight, \
           params.iterations, params.fixed_iterations, params.population_size, params.rule_list, params.p_opt_frac, \
           params.crossover_probability, params.mutation_probability, params.tournament_size


def run_iteration(dataset, population, evaluation_threshold, bacc_weight, miRNA_cdds, global_best_score,
                  best_classifiers, mirnas, population_size, crossover_probability, mutation_probability,
                  tournament_size, print_results):

    # SELECTION
    selected_parents = []

    for i in range(0, int(population_size / 2)):  # iterate through population

        first_parent_id, second_parent_id = selection.select(population,
                                                             tournament_size)
        # add new parents to the selected parents
        selected_parents.append(population[first_parent_id].__copy__())
        selected_parents.append(population[second_parent_id].__copy__())

    population.clear()  # empty population

    # CROSSOVER
    for i in range(0, int(population_size / 2)):  # iterate through parents

        crossover_rand = random.random()  # randomly choose probability for crossover

        first_parent_id = random.randrange(0, len(selected_parents))  # randomly choose first parent id
        first_parent = selected_parents[first_parent_id].__copy__()  # copy first parent

        del selected_parents[first_parent_id]  # remove parent from available parents

        second_parent_id = random.randrange(0, len(selected_parents))  # randomly choose second parent id
        second_parent = selected_parents[second_parent_id].__copy__()  # copy first parent

        del selected_parents[second_parent_id]  # remove parent from available parents

        # if the crossover_rand is lower than probability - apply crossover
        if crossover_rand <= crossover_probability:

            # crossover
            first_child, second_child = crossover.crossover(first_parent, second_parent)

            population.append(first_child)  # add children to the new population
            population.append(second_child)

        else:
            population.append(first_parent.__copy__())  # if crossover not allowed - copy parents
            population.append(second_parent.__copy__())

    # MUTATION
    population = mutation.mutate(population, mirnas, mutation_probability)

    # REMOVE RULE DUPLICATES
    for classifier in population:
        classifier.remove_duplicates()

    # EVALUATION OF THE POPULATION
    new_global_best_score, avg_population_score, best_classifiers = eval.evaluate_individuals(population, dataset,
                                                                                              evaluation_threshold,
                                                                                              bacc_weight, miRNA_cdds,
                                                                                              global_best_score,
                                                                                              best_classifiers)
    if print_results:
        print("average population score: ", avg_population_score)

    return new_global_best_score


# run genetic algorithm
def run_genetic_algorithm(train_data,  # name of train datafile
                          filter_data,  # a flag whether data should be filtered or not
                          iterations,  # number of iterations without improvement
                          fixed_iterations,  # fixed number of iterations
                          population_size,  # size of a population
                          rule_list,  # list of pre-optimized rules
                          popt_fraction,  # fraction of population that is pre-optimized
                          classifier_size,  # max size of a classifier
                          evaluation_threshold,  # evaluation function
                          miRNA_cdds,  # miRNA cdds
                          crossover_probability,  # probability of crossover
                          mutation_probability,  # probability of mutation
                          tournament_size,  # size of a tournament
                          bacc_weight,  # bacc weight
                          print_results):

    # initialize best classifier (empty)
    global_best_score = 0.0  # best classifier BACC

    # first best classifier
    best_classifier = popinit.Classifier(rule_set=[], errors={}, error_rates={}, score=0.0, bacc=0.0, cdd_score=0.0,
                                         additional_scores={})
    best_classifiers = [best_classifier.__copy__()]  # list of best classifiers

    # check if the data comes from file or data frame
    if isinstance(train_data, pandas.DataFrame):
        dataset = train_data.__copy__()
        header = dataset.columns.values.tolist()
        mirnas = header[2:]
    else:
        # read data
        dataset, negatives, positives, mirnas = preproc.read_data(train_data)

    # REMOVE IRRELEVANT miRNAs
    if filter_data is True:
        dataset, mirnas = preproc.remove_irrelevant_mirna(dataset)

    # INITIALIZE POPULATION
    if rule_list is None:
        population = popinit.initialize_population(population_size, mirnas, classifier_size)
    else:
        rule_list = popinit.read_rules_from_file(rule_list)
        population = popinit.initialize_population_from_rules(population_size, mirnas, rule_list, popt_fraction,
                                                              classifier_size)

    # REMOVE RULE DUPLICATES
    for classifier in population:
        classifier.remove_duplicates()

    # EVALUATE INDIVIDUALS
    first_global_best_score, first_avg_population_score, best_classifiers = eval.evaluate_individuals(population, dataset,
                                                                       evaluation_threshold, bacc_weight, miRNA_cdds,
                                                                       global_best_score, best_classifiers)
    if print_results:
        print("first global best score: ", first_global_best_score)
        print("first average population score: ", first_avg_population_score)

    global_best_score = first_global_best_score

    # count iterations without a change of scores
    iteration_counter = 0
    run_algorithm = True
    updates = 0

    # ITERATE OVER GENERATIONS
    # run as long as there is score change
    while run_algorithm:

        new_global_best_score = run_iteration(dataset, population, evaluation_threshold, bacc_weight, miRNA_cdds,
                                              global_best_score, best_classifiers, mirnas, population_size,
                                              crossover_probability, mutation_probability, tournament_size,
                                              print_results)

        # CHECK IMPROVEMENT
        if eval.is_higher(global_best_score, new_global_best_score):  # if there was improvement
            updates += 1
            if fixed_iterations is None:
                iteration_counter = 0
            else:
                iteration_counter = iteration_counter + 1

            global_best_score = new_global_best_score

            if print_results:
                print("new best score: ", global_best_score)
        else:  # if there is no improvement increase the number of updates and reset the iteration counter
            iteration_counter = iteration_counter + 1

        # if the iteration_counter reaches the maximal number of allowed iterations stop the algorithm
        if fixed_iterations is None:
            if iteration_counter == iterations:
                run_algorithm = False
        else:
            if iteration_counter == fixed_iterations:
                run_algorithm = False

    if print_results:
        print("Number of score updates: ", updates)

    # check classifer sizes
    classifier_sizes = []
    for classifier in best_classifiers:
        classifier_sizes.append(len(classifier.get_input_list()))

    #log.write_final_scores(global_best_score, best_classifiers)

    # show best scores
    print("\n##TRAINED CLASSIFIER## ")
    shortest_classifier = classifier_sizes.index(min(classifier_sizes))  # find shortest classifier
    log.write_final_scores(global_best_score, [best_classifiers[shortest_classifier]])  # shortest classifier

    return best_classifiers[shortest_classifier], best_classifiers, updates, first_global_best_score, \
           first_avg_population_score


def repeat(repeats, args):

    train_scores = []
    test_scores = []
    time = []
    updates_list = []
    first_scores = []
    first_avg_population_scores = []

    test_bacc = None

    for i in range(0, repeats):
        print("\nREPEAT ", i+1)
        train_bacc, test_bacc, updates, train_time, first_score, first_avg_pop = process_and_run(args)
        train_scores.append(train_bacc)
        test_scores.append(test_bacc)
        time.append(train_time)
        updates_list.append(updates)
        first_scores.append(first_score)
        first_avg_population_scores.append(first_avg_pop)

    print("\nRESULTS")
    print("AVG TRAIN: ", numpy.average(train_scores), " STDEV: ", numpy.std(train_scores))
    if test_bacc is not None:
        print("AVG TEST: ", numpy.average(test_scores), " STDEV: ", numpy.std(test_scores))
    print("AVG UPDATES: ", numpy.average(updates_list), " STDEV: ", numpy.std(updates_list))
    print("AVG TRAINING TIME: ", numpy.average(time), " STDEV: ", numpy.std(time))
    print("AVG FIRST BEST SCORE: ", numpy.average(first_scores), " STDEV: ", numpy.std(first_scores))
    print("AVG INITIAL POPULATION SCORE: ", numpy.average(first_avg_population_scores), " STDEV: ",
          numpy.std(first_avg_population_scores))

    if test_bacc is not None:
        print("CSV;", numpy.average(train_scores), ";", numpy.std(train_scores), ";",
          numpy.average(test_scores), ";", numpy.std(test_scores), ";",
          numpy.average(updates_list), ";", numpy.std(updates_list), ";",
          numpy.average(time), ";", numpy.std(time), ";",
          numpy.average(first_scores), ";", numpy.std(first_scores), ";",
          numpy.average(first_avg_population_scores), ";", numpy.std(first_avg_population_scores))
    else:
        print("CSV;", numpy.average(train_scores), ";", numpy.std(train_scores), ";",
              numpy.average(updates_list), ";", numpy.std(updates_list), ";",
              numpy.average(time), ";", numpy.std(time), ";",
              numpy.average(first_scores), ";", numpy.std(first_scores), ";",
              numpy.average(first_avg_population_scores), ";", numpy.std(first_avg_population_scores))


# process parameters and data and run algorithm
def process_and_run(args):

    # process parameters
    train_datafile, test_datafile, filter_data, discretize_data, m_bin, a_bin, l_bin, classifier_size, \
    evaluation_threshold, bacc_weight, iterations, fixed_iterations, population_size, rule_list, popt_fraction, \
    crossover_probability, mutation_probability, tournament_size = check_params(args)


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

    if rule_list is not None:
        print("POPULATION PRE-OPTIMIZATION: ", "on")
        print("POPULATION PRE-OPTIMIZED FRACTION: ", popt_fraction)
    print("GA PARAMETERS: ", "TC: ", iterations, ", PS: ", population_size, ", CP: ", crossover_probability, ", MP: ", \
          mutation_probability, ", TS: ", tournament_size)

    print("\n##TRAIN DATA##")
    # read the data
    train_dataset, negatives, positives, mirnas = preproc.read_data(train_datafile)
    annotation = train_dataset["Annots"]

    # discretize data
    if discretize_data == 't':
        print("\n##DISCRETIZATION##")
        data_discretized, miRNAs, thresholds, miRNA_cdds = \
            preproc.discretize_train_data(train_dataset, m_bin, a_bin, l_bin, True)
    else:
        data_discretized = train_dataset
        miRNA_cdds = {}
        bacc_weight = 1.0

    print("\nTRAINING...")
    start_train = time.time()
    classifier, best_classifiers, updates, first_global, first_avg_pop = \
        run_genetic_algorithm(data_discretized, filter_data, iterations, fixed_iterations, population_size,
                              rule_list, popt_fraction, classifier_size, evaluation_threshold, miRNA_cdds,
                              crossover_probability, mutation_probability, tournament_size,
                              bacc_weight, True)

    end_train = time.time()
    training_time = end_train - start_train
    print("TRAINING TIME: ", end_train - start_train)

    # evaluate best classifier
    classifier_score, train_bacc, errors, train_error_rates, train_additional_scores, cdd_score = \
        eval.evaluate_classifier(classifier, annotation, data_discretized, evaluation_threshold, miRNA_cdds,
                                 bacc_weight)

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
        test_dataset, negatives, positives, mirnas = preproc.read_data(test_datafile)
        annotation = test_dataset["Annots"]

        # discretize data
        if discretize_data == 't':
            print("\n##DISCRETIZATION##")
            data_discretized = preproc.discretize_test_data(test_dataset, thresholds)
        else:
            data_discretized = test_dataset
            miRNA_cdds = {}
            bacc_weight = 1.0

        # evaluate classifier
        classifier_score, test_bacc, errors, test_error_rates, test_additional_scores, cdd_score = \
            eval.evaluate_classifier(classifier, annotation, data_discretized, evaluation_threshold, miRNA_cdds,
                                     bacc_weight)

        print("\n##TEST DATA SCORES##")
        print("BACC: ", test_bacc)
        print("CDD SCORE: ", cdd_score)
        print("TPR: ", test_error_rates["tpr"])
        print("TNR: ", test_error_rates["tnr"])
        print("FNR: ", test_error_rates["fpr"])
        print("FPR: ", test_error_rates["fnr"])

    else:
        test_bacc = None

    return train_bacc, test_bacc, updates, training_time, first_global, first_avg_pop


if __name__ == "__main__":

    start = time.time()

    print('A genetic algorithm (GA) optimizing a set of miRNA-based distributed cell classifiers \n'
          'for in situ cancer classification. Written by Melania Nowicka, FU Berlin, 2019.\n')

    #process_and_run(sys.argv[1:])
    repeat(100, sys.argv[1:])

    end = time.time()
    print("TIME: ", end - start)
