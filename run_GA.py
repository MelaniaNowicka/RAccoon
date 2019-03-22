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

random.seed(669)


def run_genetic_algorithm(dataset_filename,  # name of the dataset file
                          iterations,  # number of iterations
                          population_size,  # size of a population
                          classifier_size,  # max size of a classifier
                          evaluation_function,  # evaluation function
                          crossover_probability,  # probability of crossover
                          mutation_probability,  # probability of mutation
                          tournament_size):  # size of a tournament

    # start log message
    log_message = "A genetic algorithm (GA) optimizing a set of miRNA-based cell classifiers for in situ cancer " \
                  "classification. Written by Melania Nowicka, FU Berlin, 2019.\n\n"
    log_file_name = "log_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".txt"  # log name

    # initialize of best classifier
    best_bacc = 0.0
    avg_bacc = 0.0
    best_classifier = popinit.Classifier(rule_set=[], error_rates={}, bacc=0.0)

    # read data
    data, negatives, positives, log_message = preproc.read_data(dataset_filename, log_message)

    # remove irrelevant miRNAs
    datasetR, mirnas, log_message = preproc.remove_irrelevant_mirna(data, log_message)

    # population initialization
    population, log_message = popinit.initialize_population(population_size, mirnas, classifier_size, log_message)

    # first evaluation of individuals
    best_bacc, avg_bacc, best_classifier = eval.evaluate_individuals(population, ('majority', 0.5), datasetR, negatives,
                                                                positives, best_bacc, best_classifier)

    # write first population to log
    log_message = log_message + "***FIRST POPULATION***"
    log_message = log.write_generation_to_log(population, 0, log_message)
    log_message = log_message + "\n***FIRST POPULATION***"
    # write log to a file
    with open(log_file_name, "a+") as log_file:
        log_file.write(log_message)

    # create a new empty population
    new_population = []

    # iterate over generations
    for iteration in range(0, iterations):
        # show progress, the current best score and average score
        print(int(iteration/iterations*100), "% | BACC: ", best_bacc, "|  AVG BACC: ", avg_bacc)
        new_population.clear()  # empty new population

        # create a list of available for selection individuals
        available_for_selection = list(range(0, len(population)))
        for i in range(0, int(population_size/2)):  # iterate through population

            crossover_rand = random.random()  # randomly choose a number for crossover
            # if the crossover_rand is lower than probability - apply crossover
            if crossover_rand <= crossover_probability:

                # selection of parents for crossover
                first_parent_id, second_parent_id = selection.select(population,
                                                                     available_for_selection,
                                                                     tournament_size)

                # selected parents are removed from a list of available parents
                available_for_selection.remove(first_parent_id)
                available_for_selection.remove(second_parent_id)

                # crossover
                first_child, second_child = crossover.crossover(population, first_parent_id, second_parent_id)

                # add offspring to the new population
                new_population.append(first_child)
                new_population.append(second_child)

        # add parents that were not used for crossover to the new population
        for j in range(0, population_size):
            if j in available_for_selection:
                new_population.append(population[j])

        # assign a new population to the current generation
        population = new_population.copy()

        # mutation
        population = mutation.mutate(population, mirnas, mutation_probability)

        # evaluation of population
        best_bacc, avg_bacc, best_classifier = eval.evaluate_individuals(population,
                                                                         evaluation_function,
                                                                         data,
                                                                         negatives,
                                                                         positives,
                                                                         best_bacc,
                                                                         best_classifier)

        # writing log message to a file
        log_message = ""
        log_message = log.write_generation_to_log(population, iteration, log_message)
        with open(log_file_name, "a+") as log_file:
            log_file.write(log_message)

    # show best scores
    log_message = log.write_final_scores(best_bacc, best_classifier)
    with open(log_file_name, "a+") as log_file:
        log_file.write(log_message)


if __name__ == "__main__":

    dataset_filename = sys.argv[1]
    run_genetic_algorithm(dataset_filename=dataset_filename,
                          iterations=50,
                          population_size=25,
                          classifier_size=5,
                          evaluation_function=('threshold', 0.75),
                          crossover_probability=0.3,
                          mutation_probability=0.5,
                          tournament_size=20)
