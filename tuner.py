import random
import numpy
import eval
import run_tests
from multiprocessing import Pool
from functools import partial


# generate parameters
def generate_parameters(tune_weights, weight_lower, weight_upper, weight_step,
                        tc_lower, tc_upper, tc_step,
                        pop_lower, pop_upper, pop_step,
                        cp_lower, cp_upper, cp_step,
                        mp_lower, mp_upper, mp_step,
                        ts_lower, ts_upper, ts_step,
                        param_sets_numb):

    parameter_sets = []  # list of param_sets_numb parameter sets
    for i in range(0, param_sets_numb):
        if tune_weights:  # if weight tuning is on
            weight = random.randrange(weight_lower, weight_upper + 1, weight_step) / 100  # generate weight
        else:
            weight = weight_lower
        tc = random.randrange(tc_lower, tc_upper + 1, tc_step)  # iterations
        pop = random.randrange(pop_lower, pop_upper+1, pop_step)  # population size
        cp = random.randrange(cp_lower, cp_upper+1, cp_step) / 100  # crossover probability
        mp = random.randrange(mp_lower, mp_upper+1, mp_step) / 100  # mutation probability
        ts = random.randrange(ts_lower, ts_upper+1, ts_step) / 100  # tournament size
        while [tc, pop, cp, mp, ts] in parameter_sets:  # if parameter set is already listed
            if tune_weights:
                weight = random.randrange(weight_lower, weight_upper + 1, weight_step) / 100
            tc = random.randrange(tc_lower, tc_upper + 1, tc_step)
            pop = random.randrange(pop_lower, pop_upper + 1, pop_step)
            cp = random.randrange(cp_lower, cp_upper + 1, cp_step) / 100
            mp = random.randrange(mp_lower, mp_upper + 1, mp_step) / 100
            ts = random.randrange(ts_lower, ts_upper + 1, ts_step) / 100
        parameter_sets.append([weight, tc, pop, cp, mp, ts])  # add set to list

    return parameter_sets


# parameter tuning
def tune_parameters(training_cv_datasets, validation_cv_datasets, feature_cdds, config, classifier_size,
                    evaluation_threshold, elite_fraction, uniqueness, rule_list, test_repeats):

    # get the parameters from configuration file
    tune_weights = config.getboolean("PARAMETER TUNING", "TuneWeights")

    if tune_weights:  # assign bounds if weight tuning is on
        weight_lower_bound = int(config['PARAMETER TUNING']['WeightLowerBound'])
        weight_upper_bound = int(config['PARAMETER TUNING']['WeightUpperBound'])
        weight_step = int(config['PARAMETER TUNING']['WeightStep'])
    else:  # otherwise assign weight from config file
        weight_lower_bound = float(config['OBJECTIVE FUNCTION']['Weight'])
        weight_upper_bound = float(config['OBJECTIVE FUNCTION']['Weight'])
        weight_step = float(config['OBJECTIVE FUNCTION']['Weight'])

    # iteration bounds
    iteration_lower_bound = int(config['PARAMETER TUNING']['IterationLowerBound'])
    iteration_upper_bound = int(config['PARAMETER TUNING']['IterationUpperBound'])
    iteration_step = int(config['PARAMETER TUNING']['IterationStep'])

    # population size range
    population_lower_bound = int(config['PARAMETER TUNING']['PopulationLowerBound'])
    population_upper_bound = int(config['PARAMETER TUNING']['PopulationUpperBound'])
    population_step = int(config['PARAMETER TUNING']['PopulationStep'])

    # crossover probability bounds
    crossover_lower_bound = int(config['PARAMETER TUNING']['CrossoverLowerBound'])
    crossover_upper_bound = int(config['PARAMETER TUNING']['CrossoverUpperBound'])
    crossover_step = int(config['PARAMETER TUNING']['CrossoverStep'])

    # mutation probability bounds
    mutation_lower_bound = int(config['PARAMETER TUNING']['MutationLowerBound'])
    mutation_upper_bound = int(config['PARAMETER TUNING']['MutationUpperBound'])
    mutation_step = int(config['PARAMETER TUNING']['MutationStep'])

    # tournament size bounds
    tournament_lower_bound = int(config['PARAMETER TUNING']['TournamentLowerBound'])
    tournament_upper_bound = int(config['PARAMETER TUNING']['TournamentUpperBound'])
    tournament_step = int(config['PARAMETER TUNING']['TournamentStep'])

    # number of parameter sets
    number_of_sets = int(config['PARAMETER TUNING']['NumberOfSets'])

    # number of available processors to parallelize cross-validation
    processes = int(config['PARALELIZATION']['ProccessorNumb'])

    # generate parameter sets
    parameter_sets = generate_parameters(tune_weights,
                                         weight_lower_bound,
                                         weight_upper_bound,
                                         weight_step,
                                         iteration_lower_bound,
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
    best_parameter_set = parameter_sets[0]  # assign first set to best parameter set
    best_avg_val_bacc = 0.0  # assign best bacc
    best_avg_val_std = 1.0  # assign best std

    parameter_set_number = 0  # parameter set counter

    # iterate over parameter sets
    for parameter_set in parameter_sets:

        parameter_set_number += 1
        print("\nTESTING PARAMETER SET ", parameter_set_number, ": ", parameter_set)

        # zip training data sets, testing data sets and cdd scores
        cv_datasets = zip(training_cv_datasets, validation_cv_datasets, feature_cdds)

        # train and test on each fold
        if processes > 1:  # if more than one processor is available
            with Pool(processes) as p:
                val_bacc_cv = p.map(partial(run_tests.train_and_test, parameter_set=parameter_set,
                                            classifier_size=classifier_size, evaluation_threshold=evaluation_threshold,
                                            elite=elite_fraction, rules=rule_list, uniqueness=uniqueness,
                                            repeats=test_repeats, print_results=False), cv_datasets)
        else:
            val_bacc_cv = list(map(partial(run_tests.train_and_test, parameter_set=parameter_set,
                                           classifier_size=classifier_size, evaluation_threshold=evaluation_threshold,
                                           elite=elite_fraction, rules=rule_list, uniqueness=uniqueness,
                                           repeats=test_repeats, print_results=False), cv_datasets))


        # calculate average bacc scores for folds and std
        val_bacc_avg = numpy.average(val_bacc_cv)
        val_std_avg = numpy.std(val_bacc_cv, ddof=1)

        print("\nRESULTS PARAMETER SET ", parameter_set_number, ": ", parameter_set)
        print("VALIDATION AVG BACC: ", val_bacc_avg, ", STD: ", val_std_avg)

        # check for improvement
        if eval.is_higher(best_avg_val_bacc, val_bacc_avg):
            best_parameter_set = parameter_set
            best_avg_val_bacc = val_bacc_avg
            best_avg_val_std = val_std_avg

        # if new score is not higher but std is lower assign new parameter set as best
        elif eval.is_close(best_avg_val_bacc, val_bacc_avg) and eval.is_higher(val_std_avg, best_avg_val_std):
            best_parameter_set = parameter_set
            best_avg_val_bacc = val_bacc_avg
            best_avg_val_std = val_std_avg

    return best_parameter_set, best_avg_val_bacc, best_avg_val_std
