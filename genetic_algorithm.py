import pandas
import random
import numpy

import popinit
import preproc
import selection
import crossover
import mutation
import eval
import log

numpy.random.seed(1)
random.seed(1)


def run_iteration(dataset, features, feature_cdds, population, population_size, elitism,
                  evaluation_threshold, bacc_weight, uniqueness, best_classifiers,
                  crossover_probability, mutation_probability, tournament_size, print_results):

    """

    Runs a single genetic algorithm iteration.

    Parameters
    ----------
    dataset : Pandas DataFrame
        data set
    features : list
        list of features
    feature_cdds : list
        list of feature cdds dicts
    population : list
        list of classifiers (Classifier objects)
    population_size : int
        population size
    elitism : bool
        if True the best found solutions are added to the population in each selection operation
    evaluation_threshold : float
        classifier evaluation threshold
    bacc_weight : float
        weight of balanced accuracy in the multi-objective score
    uniqueness : bool
         if True only unique inputs in a classifier are counted, otherwise the input cdd score is multiplied by
         the number of input
    best_classifiers : BestSolutions object
        includes best solutions
    crossover_probability : float
        crossover probability
    mutation_probability : float
        mutation probability
    tournament_size : float
        tournament size
    print_results : bool
         if True more information is shown, otherwise not all results are printed

    Returns
    -------
    best_classifiers : BestSolutions object
        includes all best classifiers

    """

    # SELECTION
    selected_parents = []
    temp_population = []

    if elitism is True:
        temp_population = population.copy()
        classifier_id = random.randrange(0, len(best_classifiers.solutions))
        temp_population.append(best_classifiers.solutions[classifier_id])

    for i in range(0, int(population_size / 2)):  # iterate through population

        # select two parents
        if elitism is True:
            first_parent_id, second_parent_id = selection.select(temp_population, tournament_size)
            # add new parents to selected parents
            selected_parents.append(temp_population[first_parent_id].__copy__())
            selected_parents.append(temp_population[second_parent_id].__copy__())
        else:
            first_parent_id, second_parent_id = selection.select(population, tournament_size)
            # add new parents to selected parents
            selected_parents.append(population[first_parent_id].__copy__())
            selected_parents.append(population[second_parent_id].__copy__())

    population.clear()  # empty population

    # CROSSOVER
    for i in range(0, int(population_size / 2)):  # iterate through selected parents

        crossover_rand = random.random()  # randomly choose probability for crossover

        first_parent_id = random.randrange(0, len(selected_parents))  # randomly choose first parent id
        first_parent = selected_parents[first_parent_id].__copy__()  # copy first parent

        del selected_parents[first_parent_id]  # remove parent from available parents

        second_parent_id = random.randrange(0, len(selected_parents))  # randomly choose second parent id
        second_parent = selected_parents[second_parent_id].__copy__()  # copy first parent

        del selected_parents[second_parent_id]  # remove parent from available parents

        # if the crossover_rand is lower than or equal to probability - apply crossover
        if crossover_rand <= crossover_probability:

            # crossover
            first_child, second_child = crossover.crossover_parents(first_parent, second_parent)

            population.append(first_child.__copy__())  # add children to the new population
            population.append(second_child.__copy__())

        else:
            population.append(first_parent.__copy__())  # if crossover not allowed - copy parents
            population.append(second_parent.__copy__())

    # MUTATION
    population = mutation.mutate(population, features, mutation_probability, evaluation_threshold)

    # UPDATE THETA AND REMOVE RULE DUPLICATES
    for classifier in population:
        classifier.update_theta()
        classifier.remove_duplicates()

    # EVALUATION OF THE POPULATION
    avg_population_score, best_classifiers = \
        eval.evaluate_individuals(population=population,
                                  dataset=dataset,
                                  bacc_weight=bacc_weight,
                                  feature_cdds=feature_cdds,
                                  uniqueness=uniqueness,
                                  best_classifiers=best_classifiers)

    if print_results:
        print("average population score: ", avg_population_score)

    return best_classifiers


# run genetic algorithm
def run_genetic_algorithm(train_data,  # name of train datafile
                          filter_data,  # a flag whether data should be filtered or not
                          iterations,  # number of iterations without improvement till termination
                          fixed_iterations,  # fixed number of iterations
                          population_size,  # size of a population
                          elitism,  # fraction of elite solutions
                          rules,  # list of pre-optimized rules
                          popt_fraction,  # fraction of population that is pre-optimized
                          classifier_size,  # max size of a classifier
                          evaluation_threshold,  # evaluation threshold
                          feature_cdds,  # feature cdds
                          crossover_probability,  # probability of crossover
                          mutation_probability,  # probability of mutation
                          tournament_size,  # size of a tournament
                          bacc_weight,  # bacc weight
                          uniqueness,  # whether only unique inputs are taken into account in cdd score
                          print_results):

    """

    Runs the genetic algorithm.

    Parameters
    ----------
    train_data : str/Pandas DataFrame
        path to train data file (str) or already read data set (Pandas DataFrame)
    filter_data : bool
        if True non-relevant features are filtered out from the data
    iterations : int
        number of iterations without improvement until termination
    fixed_iterations : int
        fixed number of iterations until termination
    population_size : int
        population size
    elitism : bool
         if True the best found solutions are added to the population in each selection operation
    rules : list
        name of rule file or None
    popt_fraction : float
        fraction of population that is pre-optimized
    classifier_size : int
        maximal classifier size
    evaluation_threshold : float
        classifier evaluation threshold
    feature_cdds : dict
         dict of feature cdds
    crossover_probability : float
        crossover probability
    mutation_probability : float
        mutation probability
    tournament_size : float
        tournament size
    bacc_weight : float
        weight of balanced accuracy in the multi-objective score
    uniqueness : bool
        if True only unique inputs in a classifier are counted, otherwise the input cdd score is multiplied by
        the number of input occurrences
    print_results : bool
        if True more information is shown, otherwise not all results are printed

    Returns
    -------
    best_classifier : Classifier object
        best classifier
    best_classifiers : BestSolutions object
        includes all best classifiers
    updates : int
        number of best score updates
    first_avg_population_score : float
        first population average score

    """

    # initialize best solutions object
    best_classifiers = eval.BestSolutions(0.0, [], [])
    global_best_score = best_classifiers.score

    # check if data comes from file or data frame
    if isinstance(train_data, pandas.DataFrame):
        dataset = train_data.__copy__()
        header = dataset.columns.values.tolist()
        features = header[2:]
        samples, annotation, negatives, positives = preproc.get_data_info(dataset, header)
    else:
        # read data
        dataset, annotation, negatives, positives, features = preproc.read_data(train_data)

    # REMOVE IRRELEVANT features
    if filter_data:
        dataset, features = preproc.remove_irrelevant_features(dataset)

    # INITIALIZE POPULATION
    if rules is None:
        population = popinit.initialize_population(population_size, features, evaluation_threshold, classifier_size)
    else:
        rules = popinit.read_rules_from_file(rules)
        population = popinit.initialize_population_from_rules(population_size, features, evaluation_threshold,
                                                              rules, popt_fraction, classifier_size)

    # UPDATE THETA AND REMOVE RULE DUPLICATES
    for classifier in population:
        classifier.update_theta()
        classifier.remove_duplicates()

    # EVALUATE INDIVIDUALS
    first_avg_population_score, best_classifiers = \
        eval.evaluate_individuals(population, dataset, bacc_weight, feature_cdds,
                                  uniqueness, best_classifiers)

    if print_results:
        print("first global best score: ", best_classifiers.score)
        print("first average population score: ", first_avg_population_score)

    first_global_best_score = best_classifiers.score
    global_best_score = best_classifiers.score

    iteration_counter = 0  # count iterations without change of scores
    updates = 0  # count number of score updates
    run_algorithm = True

    # ITERATE OVER GENERATIONS
    # run as long as there is score change
    while run_algorithm:

        best_classifiers = run_iteration(dataset=dataset,
                                         features=features,
                                         feature_cdds=feature_cdds,
                                         population=population,
                                         population_size=population_size,
                                         elitism=elitism,
                                         evaluation_threshold=evaluation_threshold,
                                         bacc_weight=bacc_weight,
                                         uniqueness=uniqueness,
                                         best_classifiers=best_classifiers,
                                         crossover_probability=crossover_probability,
                                         mutation_probability=mutation_probability,
                                         tournament_size=tournament_size,
                                         print_results=print_results)

        # CHECK IMPROVEMENT
        if eval.is_higher(global_best_score, best_classifiers.score):  # if there was improvement
            updates += 1  # add new update
            if fixed_iterations == 0:  # if there is no number of fixed iterations
                iteration_counter = 0  # reset iteration without improvement counter
            else:
                iteration_counter = iteration_counter + 1  # else add iteration

            global_best_score = best_classifiers.score  # assign new global best score

            if print_results:
                print("new best score: ", global_best_score)

        else:  # if there is no improvement increase the number of updates
            iteration_counter = iteration_counter + 1

        # if the iteration_counter reaches the maximal number of allowed iterations stop the algorithm
        if fixed_iterations == 0:
            if iteration_counter == iterations:
                run_algorithm = False
        else:
            if iteration_counter == fixed_iterations:
                run_algorithm = False

    if print_results:
        print("Number of score updates: ", updates)

    # check classifer sizes
    classifier_sizes = []
    for classifier in best_classifiers.solutions:
        classifier_sizes.append(len(classifier.get_input_list()))

    # show best scores
    print("\n##TRAINED CLASSIFIER## ")
    shortest_classifier = classifier_sizes.index(min(classifier_sizes))  # find shortest classifier
    log.write_final_scores(global_best_score, [best_classifiers.solutions[shortest_classifier]])  # shortest classifier

    best_classifier = best_classifiers.solutions[shortest_classifier]
    return best_classifier, best_classifiers, updates, first_global_best_score, first_avg_population_score
