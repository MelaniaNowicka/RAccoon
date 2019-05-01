'''
A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer classification.
Written by Melania Nowicka, FU Berlin, 2019.
'''

import datetime
import sys
import random
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

random.seed(0)

# algorithm parameters read from the command line
def check_params(args):

    # parameter parser
    parser = argparse.ArgumentParser(description='A genetic algorithm (GA) optimizing a set of miRNA-based cell '
                                                 'classifiers for in situ cancer classification. Written by Melania '
                                                 'Nowicka, FU Berlin, 2019.\n\n')

    # adding arguments
    parser.add_argument('-train', '--dataset_filename_train', help='data set file name')
    parser.add_argument('-test', '--dataset_filename_test', help='data set file name')
    parser.add_argument('-f', '--filter_data', type=bool, default=True, help='filter data of not')
    parser.add_argument('-iter', '--iterations', type=int, default=100, help='number of iterations')
    parser.add_argument('-pop', '--population_size', type=int, default=300, help='population size')
    parser.add_argument('-size', '--classifier_size', type=int, default=5, help='classifier size')
    parser.add_argument('-thres', '--evaluation_threshold', default=0.75, type=float, help='evaluation threshold')
    parser.add_argument('-cp', '--crossover_probability', default=0.9, type=float, help='probability of crossover')
    parser.add_argument('-mp', '--mutation_probability', default=0.1, type=float, help='probability of mutation')
    parser.add_argument('-ts', '--tournament_size', default=0.2, type=float, help='tournament size')

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
                          crossover_probability,  # probability of crossover
                          mutation_probability,  # probability of mutation
                          tournament_size):  # size of a tournament

    # start log message for GA run
    log_message = "A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer " \
                  "classification. Written by Melania Nowicka, FU Berlin, 2019.\n\n"
    log_file_name = "log_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".txt"  # log name

    # initialize of best classifier
    best_bacc = 0.0  # best classifier BACC
    avg_bacc = 0.0  # average BACC in the population
    best_classifier = popinit.Classifier(rule_set=[], error_rates={}, bacc=0.0)  # first best classifier
    best_classifiers = [best_classifier.__copy__()]  # list of best classifiers

    # read data
    dataset, negatives, positives, mirnas, log_message = preproc.read_data(train_data, log_message)

    # remove irrelevant miRNAs
    if filter_data == True:
        dataset, mirnas, log_message = preproc.remove_irrelevant_mirna(dataset, log_message)

    # population initialization
    population, log_message = popinit.initialize_population(population_size, mirnas, classifier_size, log_message)

    # remove rule duplicates
    for classifier in population:
        classifier.remove_duplicates()

    # first evaluation of individuals
    best_bacc, avg_bacc, best_classifiers = eval.evaluate_individuals(population, evaluation_threshold, dataset,
                                                                     negatives, positives, best_bacc, best_classifiers)
    # write first population to log
    #log_message = log_message + "***FIRST POPULATION***"
    #log_message = log.write_generation_to_log(population, 0, best_classifiers, log_message)
    #log_message = log_message + "\n***FIRST POPULATION***"
    #with open(log_file_name, "a+") as log_file: # write log to a file
        #log_file.write(log_message)

    # create a new empty population for selection
    selected_parents = []

    # iterate over generations
    for iteration in range(0, iterations):
        # show progress, the current best score and average score
        print(int(iteration/iterations*100), "% | BACC: ", best_bacc, "|  AVG BACC: ", avg_bacc)

        selected_parents.clear()  # empty new population for selection

        # selection of individuals for crossover
        for i in range(0, int(population_size/2)):  # iterate through population
            first_parent_id, second_parent_id = selection.select(population,
                                                                 tournament_size)
            # add new parents to the selected parents
            selected_parents.append(population[first_parent_id].__copy__())
            selected_parents.append(population[second_parent_id].__copy__())

        population.clear()  # empty population

        # crossover of selected individuals
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
                first_child, second_child = crossover.crossover(selected_parents, first_parent, second_parent)

                population.append(first_child)  # add children to the new population
                population.append(second_child)
            else:
                population.append(first_parent.__copy__())  # if crossover not allowed - copy parents
                population.append(second_parent.__copy__())

        # mutation
        population = mutation.mutate(population, mirnas, mutation_probability)

        # remove rule duplicates
        for classifier in population:
            classifier.remove_duplicates()

        # evaluation of population
        best_bacc, avg_bacc, best_classifiers = eval.evaluate_individuals(population,
                                                                         evaluation_threshold,
                                                                         dataset,
                                                                         negatives,
                                                                         positives,
                                                                         best_bacc,
                                                                         best_classifiers)

        # writing log message to a file
        #log_message = ""
        #log_message = log.write_generation_to_log(population, iteration, best_classifiers, log_message)
        #with open(log_file_name, "a+") as log_file:
            #log_file.write(log_message)
    #log_message = "***LAST POPULATION***"
    #log_message = log.write_generation_to_log(population, iteration, best_classifiers, log_message)
    #log_message = "***LAST POPULATION***"
    # show best scores
    #log_message = log.write_final_scores(best_bacc, best_classifiers)
    #with open(log_file_name, "a+") as log_file:
        #log_file.write(log_message)

    #print("BEST CLASSIFIERS: ")
    # show best scores
    #log_message = log.write_final_scores(best_bacc, best_classifiers)

    classifier_sizes = []
    for classifier in best_classifiers:
        inputs = 0
        for rule in classifier.rule_set:
            for input in rule.pos_inputs:
                inputs = inputs + 1
            for input in rule.neg_inputs:
                inputs = inputs + 1
        classifier_sizes.append(inputs)

    shortest_classifier = classifier_sizes.index(min(classifier_sizes))
    print("SHORTEST CLASSIFIER: ")
    # show best scores
    log_message = log.write_final_scores(best_bacc, [best_classifiers[shortest_classifier]])
    #with open(log_file_name, "a+") as log_file:
        #log_file.write(log_message)

    return best_classifiers[shortest_classifier], best_classifiers

if __name__ == "__main__":

    start = time.time()

    train_datafile, test_datafile, filter_data, iterations, population_size, classifier_size, evaluation_threshold, \
    crossover_probability, mutation_probability, tournament_size = check_params(sys.argv[1:])

    classifier_performances = []

    for i in range(0, 1):
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
        print("TEST DATA")
        test_bacc = toolbox.evaluate_classifier(classifier, evaluation_threshold, test_datafile)
        classifier_performances.append(test_bacc)

    bacc_avg = sum(classifier_performances) / len(classifier_performances)

    print("AVERAGE BACC: ", bacc_avg)

    end = time.time()
    print("TIME: ", end - start)
