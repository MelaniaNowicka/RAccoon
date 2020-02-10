'''
A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import datetime
import sys
import random
import numpy
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
import toolbox


# algorithm parameters read from the command line
def check_params(args):

    # parameter parser
    parser = argparse.ArgumentParser(description='A genetic algorithm (GA) optimizing a set of miRNA-based cell '
                                                 'classifiers for in situ cancer classification. Written by Melania '
                                                 'Nowicka, FU Berlin, 2019.\n\n')

    # adding arguments
    parser.add_argument('--train', '--dataset-filename-train', dest="dataset_filename_train", help='data set file name')
    parser.add_argument('--test', '--dataset-filename-test', dest="dataset_filename_test", help='data set file name')
    parser.add_argument('-f', '--filter-data', dest="filter_data", type=bool, default=False, help='filter data of not')
    parser.add_argument('-i', '--iterations', dest="iterations", type=int, default=100, help='number of iterations')
    parser.add_argument('-p', '--population-size', dest="population_size", type=int, default=300, help='population size')
    parser.add_argument('-c', '--classifier-size', dest="classifier_size", type=int, default=5, help='classifier size')
    parser.add_argument('-a', '--evaluation-threshold', dest="evaluation_threshold", default=0.75, type=float, help='evaluation threshold alpha')
    parser.add_argument('-x', '--crossover-probability', dest="crossover_probability", default=0.9, type=float, help='probability of crossover')
    parser.add_argument('-m', '--mutation-probability', dest="mutation_probability", default=0.1, type=float, help='probability of mutation')
    parser.add_argument('-t', '--tournament-size', dest="tournament_size", default=0.2, type=float, help='tournament size')

    # parse arguments
    params = parser.parse_args(args)

    return params.dataset_filename_train, params.dataset_filename_test, params.filter_data, params.iterations, \
           params.population_size, params.classifier_size, params.evaluation_threshold, params.crossover_probability, \
           params.mutation_probability, params.tournament_size


# run genetic algorithm
def run_genetic_algorithm(train_data,  # name of train datafile
                          filter_data,  # a flag whether data should be filtered or not
                          iterations,  # number of iterations
                          population_size,  # size of a population
                          classifier_size,  # max size of a classifier
                          evaluation_threshold,  # evaluation function
                          miRNA_cdds,  # miRNA cdds
                          crossover_probability,  # probability of crossover
                          mutation_probability,  # probability of mutation
                          tournament_size,  # size of a tournament
                          bacc_weight):  # bacc weight

    # initialize of best classifier
    best_score = 0.0  # best classifier BACC

    # first best classifier
    best_classifier = popinit.Classifier(rule_set=[], errors={}, error_rates={}, score=0.0, bacc=0.0, cdd_score=0.0, additional_scores={})
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
    if filter_data == True:
        dataset, mirnas = preproc.remove_irrelevant_mirna(dataset)

    # INITIALIZE POPULATION
    population = popinit.initialize_population(population_size, mirnas, classifier_size)

    # REMOVE RULE DUPLICATES
    for classifier in population:
        classifier.remove_duplicates()

    # EVALUATE INDIVIDUALS
    best_score, avg_bacc, best_classifiers = eval.evaluate_individuals(population, dataset,
                                                                       evaluation_threshold, bacc_weight, miRNA_cdds,
                                                                       best_score, best_classifiers)
    # count iterations without a change of scores
    iteration_counter = 0

    # ITERATE OVER GENERATIONS
    # run as long as there is score change
    run_algorithm = True
    while run_algorithm == True:

        # show progress, the current best score and average score
        #print(int(iteration/iterations*100), "% | BACC: ", best_score, "|  AVG BACC: ", avg_bacc)

        # SELECTION
        selected_parents = []

        for i in range(0, int(population_size/2)):  # iterate through population

            first_parent_id, second_parent_id = selection.select(population,
                                                                 tournament_size)
            # add new parents to the selected parents
            selected_parents.append(population[first_parent_id].__copy__())
            selected_parents.append(population[second_parent_id].__copy__())

        population.clear()  # empty population

        # CROSSOVER
        for i in range(0, int(population_size/2)):  # iterate through parents

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

        global_best_score = best_score

        # evaluation of population
        best_score, avg_bacc, best_classifiers = \
            eval.evaluate_individuals(population, dataset, evaluation_threshold, bacc_weight, miRNA_cdds, best_score,
                                      best_classifiers)

        if best_score == global_best_score:
            iteration_counter = iteration_counter + 1
        else:
            iteration_counter = 0

        if iteration_counter == iterations:
            run_algorithm = False

    classifier_sizes = []
    for classifier in best_classifiers:
        inputs = 0
        for rule in classifier.rule_set:
            for input in rule.pos_inputs:
                inputs = inputs + 1
            for input in rule.neg_inputs:
                inputs = inputs + 1
        classifier_sizes.append(inputs)

    # show best scores
    shortest_classifier = classifier_sizes.index(min(classifier_sizes))
    print("SHORTEST CLASSIFIER: ")
    log.write_final_scores(best_score, [best_classifiers[shortest_classifier]])
    return best_classifiers[shortest_classifier], best_classifiers


if __name__ == "__main__":

    start = time.time()

    train_datafile, test_datafile, filter_data, iterations, population_size, classifier_size, evaluation_threshold, \
    crossover_probability, mutation_probability, tournament_size = check_params(sys.argv[1:])

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

    for i in range(0, 50):
        print("TEST ", i+1)
        print("TRAINING DATA")
        classifier, best_classifiers = run_genetic_algorithm(train_datafile,
                                           filter_data,
                                           iterations,
                                           population_size,
                                           classifier_size,
                                           evaluation_threshold,
                                           crossover_probability,
                                           mutation_probability,
                                           tournament_size)

        print("TRAIN DATA")
        train_bacc, train_error_rates, train_additional_scores = \
            toolbox.evaluate_classifier(classifier, evaluation_threshold, train_datafile)
        train_bacc_avg.append(train_bacc)
        train_tpr_avg.append(train_error_rates["tpr"])
        train_tnr_avg.append(train_error_rates["tnr"])
        train_fpr_avg.append(train_error_rates["fpr"])
        train_fnr_avg.append(train_error_rates["fnr"])
        train_f1_avg.append(train_additional_scores["f1"])
        train_mcc_avg.append(train_additional_scores["mcc"])
        train_ppv_avg.append(train_additional_scores["ppv"])
        train_fdr_avg.append(train_additional_scores["fdr"])

        print("TEST DATA")
        test_bacc, test_error_rates, test_additional_scores = \
            toolbox.evaluate_classifier(classifier, evaluation_threshold, test_datafile)
        test_bacc_avg.append(test_bacc)

    print("TRAIN AVG BACC: ", numpy.average(train_bacc_avg))
    print("TRAIN AVG STDEV: ", numpy.std(train_bacc_avg))
    print("TRAIN AVG TPR: ", numpy.average(train_tpr_avg))
    print("TRAIN AVG TNR: ", numpy.average(train_tnr_avg))
    print("TRAIN AVG FPR: ", numpy.average(train_fpr_avg))
    print("TRAIN AVG FNR: ", numpy.average(train_fnr_avg))
    print("TRAIN AVG TPR: ", numpy.average(train_f1_avg))
    print("TRAIN AVG TNR: ", numpy.average(train_mcc_avg))
    print("TRAIN AVG FPR: ", numpy.average(train_ppv_avg))
    print("TRAIN AVG FNR: ", numpy.average(train_fdr_avg))

    print("TEST AVG BACC: ", numpy.average(test_bacc_avg))
    print("TEST AVG STDEV: ", numpy.std(test_bacc_avg))
    print("TEST AVG TPR: ", numpy.average(test_tpr_avg))
    print("TEST AVG TNR: ", numpy.average(test_tnr_avg))
    print("TEST AVG FPR: ", numpy.average(test_fpr_avg))
    print("TEST AVG FNR: ", numpy.average(test_fnr_avg))
    print("TEST AVG TPR: ", numpy.average(test_f1_avg))
    print("TEST AVG TNR: ", numpy.average(test_mcc_avg))
    print("TEST AVG FPR: ", numpy.average(test_ppv_avg))
    print("TEST AVG FNR: ", numpy.average(test_fdr_avg))

    end = time.time()
    print("TIME: ", end - start)
