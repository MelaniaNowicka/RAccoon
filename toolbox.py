import preproc
import popinit
import eval
import pandas
import numpy
import random
from matplotlib import pyplot
import seaborn
import os
import csv

random.seed(1)


def write_config_to_log(config_file):

    """

    Writes all parameter values to log.

    Parameters
    ----------
    config_file : ConfigParser object
        configuration file

    """

    print("###########READING CONFIG###########\n")
    print("CLASSIFIER PARAMETERS")
    print("Classifier Size: ", int(config_file['CLASSIFIER PARAMETERS']['ClassifierSize']))
    print("Set Alpha: ", config_file.getboolean("CLASSIFIER PARAMETERS", "SetAlpha"))
    print("Alpha: ", float(config_file['CLASSIFIER PARAMETERS']['Alpha']))

    print("\nDATA DIVISION")
    print("Training Fraction: ", int(config_file['DATA DIVISION']['TrainingFraction']))
    print("CV Folds: ", int(config_file['DATA DIVISION']['CVFolds']))
    print("Pairing: ", config_file.getboolean("DATA DIVISION", "Pairing"))
    print("Set Seed: ", config_file.getboolean("DATA DIVISION", "SetSeed"))

    print("\nBINARIZATION PARAMETERS")
    print("M Segments: ", int(config_file['BINARIZATION PARAMETERS']['MSegments']))
    print("Alpha Bin: ", float(config_file['BINARIZATION PARAMETERS']['AlphaBin']))
    print("Lambda Bin: ", float(config_file['BINARIZATION PARAMETERS']['LambdaBin']))

    print("\nPARAMETER TUNING")
    print("Tune parameters: ", config_file.getboolean("PARAMETER TUNING", "Tuning"))
    print("Number Of Sets: ", int(config_file['PARAMETER TUNING']['NumberOfSets']))
    print("Tune Weights: ", config_file.getboolean("PARAMETER TUNING", "TuneWeights"))
    print("Iteration Lower Bound: ", int(config_file['PARAMETER TUNING']['IterationLowerBound']))
    print("Iteration Upper Bound: ", int(config_file['PARAMETER TUNING']['IterationUpperBound']))
    print("Iteration Step: ", int(config_file['PARAMETER TUNING']['IterationStep']))
    print("Population Lower Bound: ", int(config_file['PARAMETER TUNING']['PopulationLowerBound']))
    print("Population Upper Bound: ", int(config_file['PARAMETER TUNING']['PopulationUpperBound']))
    print("Population Step: ", int(config_file['PARAMETER TUNING']['PopulationStep']))
    print("Crossover Lower Bound: ", int(config_file['PARAMETER TUNING']['CrossoverLowerBound']))
    print("Crossover Upper Bound: ", int(config_file['PARAMETER TUNING']['CrossoverUpperBound']))
    print("Crossover Step: ", int(config_file['PARAMETER TUNING']['CrossoverStep']))
    print("Mutation Lower Bound: ", int(config_file['PARAMETER TUNING']['MutationLowerBound']))
    print("Mutation Upper Bound: ", int(config_file['PARAMETER TUNING']['MutationUpperBound']))
    print("Mutation Step: ", int(config_file['PARAMETER TUNING']['MutationStep']))
    print("Tournament Lower Bound: ", int(config_file['PARAMETER TUNING']['TournamentLowerBound']))
    print("Tournament Upper Bound: ", int(config_file['PARAMETER TUNING']['TournamentUpperBound']))
    print("Tournament Step: ", int(config_file['PARAMETER TUNING']['TournamentStep']))
    print("Weight Lower Bound: ", int(config_file['PARAMETER TUNING']['WeightLowerBound']))
    print("Weight Upper Bound: ", int(config_file['PARAMETER TUNING']['WeightUpperBound']))
    print("Weight Step: ", int(config_file['PARAMETER TUNING']['WeightStep']))

    print("\nGA PARAMETERS")
    print("Iterations: ", int(config_file['GA PARAMETERS']['Iterations']))
    print("Population Size: ", int(config_file['GA PARAMETERS']['PopulationSize']))
    print("Crossover Probability: ", float(config_file['GA PARAMETERS']['CrossoverProbability']))
    print("Mutation Probability: ", float(config_file['GA PARAMETERS']['MutationProbability']))
    print("Tournament Size: ", float(config_file['GA PARAMETERS']['TournamentSize']))

    print("\nRUN PARAMETERS")
    print("Single Test Repeats: ", int(config_file['RUN PARAMETERS']['SingleTestRepeats']))

    print("\nALGORITHM PARAMETERS")
    print("Elitism: ", config_file.getboolean("ALGORITHM PARAMETERS", "Elitism"))

    print("\nOBJECTIVE FUNCTION")
    print("Weight: ", float(config_file['OBJECTIVE FUNCTION']['Weight']))
    print("Uniqueness: ", config_file.getboolean("OBJECTIVE FUNCTION", "Uniqueness"))

    print("\nPARALELIZATION")
    print("Proccessor Numb: ", int(config_file['PARALELIZATION']['ProccessorNumb']))
    print("\n")


def compare_folds(fold1, fold2):

    """

    Compares two data folds by checking the sample ids and data points.

    Parameters
    ----------
    fold1 : str
        path to first data fold
    fold2 : str
        path to second data fold

    """

    if isinstance(fold1, pandas.DataFrame) and isinstance(fold2, pandas.DataFrame):
        data1 = fold1.__copy__()
        data2 = fold2.__copy__()
    else:
        data1, annotation, negatives, positives, features = preproc.read_data(fold1)
        data2, annotation, negatives, positives, features = preproc.read_data(fold2)

    samples1 = list(data1["ID"])
    samples2 = list(data2["ID"])

    print("SAMPLES MATCH: ", samples1 == samples2)
    print("DATA MATCH: ", data1.equals(data2))


def compare_ids(train_fold, test_fold):

    """

    Compares training and test data folds for sample id uniqueness

    Parameters
    ----------
    train_fold : str
        path to train data fold
    test_fold : str
        path to test data fold

    """

    dataset_train, annotation, negatives, positives, features = preproc.read_data(train_fold)
    dataset_val, annotation, negatives, positives, features = preproc.read_data(test_fold)

    train = dataset_train["ID"].to_list()
    val = dataset_val["ID"].to_list()

    intersection = set(train) & set(val)
    if len(intersection) == 0:
        print("COMPARISON RESULT: samples are unique.")
    else:
        print("COMPARISON RESULT: wrong data division, samples are not unique!")


# divide data into train and test
def divide_into_train_test(dataset_file_name, train_fraction, set_seed):

    """

    Divides a data set into training and test fraction.

    Parameters
    ----------
    dataset_file_name : Pandas DataFrame
        data set
    train_fraction : float
        fraction of samples in training data
    set_seed : bool
        if True sample() seed is set to 1

    Returns
    -------
    training_data : dataframe
        training data set
    testing_data : dataframe
        test data set

    """

    dataset, annotation, negatives, positives, features = preproc.read_data(dataset_file_name)

    data_size = len(dataset.index)

    # copy samples
    negative_samples = dataset.iloc[:negatives].copy()
    positive_samples = dataset.iloc[negatives:data_size].copy()

    # calculate how many samples go to training set
    negative_samples_to_draw = int(round(negatives * train_fraction / 100))
    positive_samples_to_draw = int(round(positives * train_fraction / 100))

    print("TRAINING DATA SET: ", positive_samples_to_draw + negative_samples_to_draw, " samples")
    print("TESTING DATA SET: ", data_size - (positive_samples_to_draw + negative_samples_to_draw), " samples")

    # draw positive training samples
    if set_seed is True:
        training_positives = positive_samples.sample(n=positive_samples_to_draw, random_state=1)  # draw n samples
    else:
        training_positives = positive_samples.sample(n=positive_samples_to_draw)  # draw n samples
    training_positives.sort_index()  # sort samples
    ids_to_drop = training_positives.index.values  # get the ids of positive training samples
    testing_positives = positive_samples.drop(ids_to_drop)  # add rest as test samples

    # draw negative training samples
    if set_seed is True:
        training_negatives = negative_samples.sample(n=negative_samples_to_draw, random_state=1)  # draw n samples
    else:
        training_negatives = negative_samples.sample(n=negative_samples_to_draw)  # draw n samples
    training_negatives.sort_index()  # sort samples
    ids_to_drop = training_negatives.index.values  # get the ids of negative training samples
    testing_negatives = negative_samples.drop(ids_to_drop)  # add rest as test samples

    training_data = training_negatives.append(training_positives)  # merge negative and positive samples
    testing_data = testing_negatives.append(testing_positives)

    return training_data, testing_data


# division of the data set into k folds
def divide_into_cv_folds(dataset_file_name, path, dataset, k_folds, pairing, set_seed):

    """

    Divides a data set into k training and validation data folds.

    Parameters
    ----------
    dataset_file_name : str
        data set filename
    path : str
        path to data set file
    dataset : Pandas DataFrame
        data set
    k_folds : int
        number of cv folds
    pairing : bool
        if True samples are divided into training and validation folds by pairs (first negative and first positive
        sample is treated as a pair)
    set_seed : bool
        if True sample() seed is set to 1

    Returns
    -------
    training_data : list
        list of train data folds
    testing_data : list
        list of validation data folds

    """

    header = dataset.columns.values.tolist()
    samples, annotation, negatives, positives = preproc.get_data_info(dataset, header)

    negative_samples = dataset.iloc[:negatives].copy()  # copy negative samples
    positive_samples = dataset.iloc[negatives:samples].copy()  # copy positive samples
    positive_samples_temp = []  # used only if pairing is True, allows to leave positive_samples untouched

    negative_samples_to_draw = int(round(negatives / k_folds))  # number of neg samples to draw
    positive_samples_to_draw = int(round(positives / k_folds))  # number of pos samples to draw

    negative_folds = []  # list of negative folds
    positive_folds = []  # list of positive folds

    train_datasets = []  # list of train folds
    val_datasets = []   # list of test folds

    if pairing is True:  # create a copy of positive samples
        positive_samples_temp = dataset.iloc[negatives:samples].copy()

    for fold in range(1, k_folds):  # k-1 times draw positive and negative samples

        if set_seed is True:
            # draw n samples
            negative_data_fold = negative_samples.sample(n=negative_samples_to_draw, random_state=1)
        else:
            negative_data_fold = negative_samples.sample(n=negative_samples_to_draw)

        negative_folds.append(negative_data_fold.sort_index())  # add sorted fold to negative folds
        neg_used_ids = negative_data_fold.index.values  # list used ids

        if pairing is True:  # if samples are paired draw by pair (depends on order of samples in file!)
            positive_data_fold = positive_samples.iloc[neg_used_ids]  # draw paired samples (same ids as for negatives)
            positive_folds.append(positive_data_fold.sort_index())  # add sorted fold to positive folds
            pos_used_ids = [x + negatives for x in neg_used_ids]  # calculate paired ids based on neg used ids to drop
            positive_samples_temp.drop(pos_used_ids, inplace=True)  # drop used samples by pos_used_ids
        else:  # if samples are not paired draw positive samples randomly
            if set_seed is True:
                # draw n samples
                positive_data_fold = positive_samples.sample(n=positive_samples_to_draw, random_state=1)
            else:
                positive_data_fold = positive_samples.sample(n=positive_samples_to_draw)
            positive_folds.append(positive_data_fold.sort_index())  # add sorted fold to positive folds
            pos_used_ids = positive_data_fold.index.values  # used ids
            positive_samples.drop(pos_used_ids, inplace=True)  # drop used samples

        negative_samples.drop(neg_used_ids, inplace=True)  # drop used samples

    # add remaining samples as last folds
    if pairing is True:
        positive_folds.append(positive_samples_temp)  # if pairing is True use positive_samples_temp
    else:
        positive_folds.append(positive_samples)  # otherwise use positive_samples
    negative_folds.append(negative_samples)

    for fold in range(0, k_folds):  # create k validation folds

        val_fold = negative_folds[fold].append(positive_folds[fold])  # merge negative and positive samples
        val_datasets.append(val_fold)  # add to list of validation folds

        print("VALIDATION FOLD ", fold+1, len(val_fold.index.values))

    for fold in range(0, k_folds):  # create k train folds

        train_folds_to_merge = []  # list of folds to merge
        for i in range(0, k_folds):
            if i != fold:  # ommit one fold to become validation fold
                train_folds_to_merge.append(val_datasets[i])  # add rest of folds to train fold

        train_fold = train_folds_to_merge[0].copy()  # merge folds to create train fold
        for i in range(1, len(train_folds_to_merge)):
            train_fold = train_fold.append(train_folds_to_merge[i])

        print("TRAIN FOLD ", fold + 1, len(train_fold.index.values))
        train_fold = train_fold.sort_index()

        train_datasets.append(train_fold)

    # save to files
    fold = 1

    for train_set, val_set in zip(train_datasets, val_datasets):  # iterate over pairs of sets

        new_name = "_cv_train_" + str(fold) + ".csv"  # train fold name
        new_name = dataset_file_name.replace(".csv", new_name)
        filename = os.path.join(path, new_name)
        train_set.to_csv(filename, sep=";", index=False)

        new_name = "_cv_val_" + str(fold) + ".csv"  # validation fold name
        new_name = dataset_file_name.replace(".csv", new_name)
        filename = os.path.join(path, new_name)
        val_set.to_csv(filename, sep=";", index=False)

        fold = fold + 1

    return train_datasets, val_datasets


# watch out, may be slow!
# removes symmetric solutions (solutions differing only in the order of gates and inputs)
def remove_symmetric_solutions(best_classifiers):

    """

    Removes symmetric solutions (solutions differing only in the order of gates and inputs).

    Parameters
    ----------
    best_classifiers : BestSolutions object
         includes all best solutions

    """

    to_del = []  # solution ids to delete

    for i in range(0, len(best_classifiers.solutions)-1):  # iterate over solutions
        for j in range(i+1, len(best_classifiers.solutions)):
            # check if classifiers have same thresholds
            if best_classifiers.solutions[i].evaluation_threshold == best_classifiers.solutions[j].evaluation_threshold:
                # check if solutions have same size comparing length of rule_sets
                if len(best_classifiers.solutions[i].rule_set) == len(best_classifiers.solutions[j].rule_set):
                    identical_rule_counter = 0  # set counter of identical rules
                    for rule1 in best_classifiers.solutions[i].rule_set:  # iterate over rules in two compared solutions
                        for rule2 in best_classifiers.solutions[j].rule_set:
                            if sorted(rule1.neg_inputs) == sorted(rule2.neg_inputs):  # if neg inputs are equal
                                if sorted(rule1.pos_inputs) == sorted(rule2.pos_inputs):  # if pos inputs are equal
                                    if rule1.gate == rule2.gate:  # if gates are equal
                                        identical_rule_counter += 1  # add identical rule
                                        break
                    # if all rules are identical
                    if identical_rule_counter == len(best_classifiers.solutions[i].rule_set):
                        to_del.append(j)  # add id to delete

    if len(to_del) != 0:

        # sort indices in descending order for removal
        to_del = list(set(to_del))
        to_del.sort(reverse=True)
        # remove duplicates
        for i in to_del:
            del best_classifiers.solutions[i]
            del best_classifiers.solutions_str[i]


# ranks features in classifiers
# ranks features in classifiers
def rank_features_by_frequency(solutions, path, file_name):

    """

    Ranks features in classifiers.

    Parameters
    ----------
    solutions : list
        list of classifiers

    """

    frequency_general = {}  # count occurrences in total
    frequency_pos = {}  # count occurrences as positive inputs
    frequency_neg = {}  # count occurrences as negative inputs

    features_total = 0

    for solution in solutions:  # for all solutions
        for rule in solution.rule_set:  # for all rules
            for inp in rule.pos_inputs:  # for all positive inputs
                if inp not in frequency_pos.keys():  # check whether the input key is already in the dict
                    frequency_pos[inp] = 1  # add key
                    frequency_general[inp] = 1
                else:
                    frequency_pos[inp] += 1
                    frequency_general[inp] += 1

            for inp in rule.neg_inputs:  # for all negative inputs
                if inp not in frequency_neg.keys():
                    frequency_neg[inp] = 1
                    frequency_general[inp] = 1
                else:
                    frequency_neg[inp] += 1
                    frequency_general[inp] += 1

        features_total += len(solution.get_input_list())  # total number of features in all solutions

    print("TOTAL VALUES")
    print("NUMBER OF FEATURES IN ALL SOLUTIONS IN TOTAL: ", features_total)
    print("POSITIVE FEATURES: ")
    for feature in sorted(frequency_pos, key=frequency_pos.get, reverse=True):
        print(feature, ": ", frequency_pos[feature])
    print("NEGATIVE FEATURES: ")
    for feature in sorted(frequency_neg, key=frequency_neg.get, reverse=True):
        print(feature, ": ", frequency_neg[feature])

    header = {'feature': [], 'level': [], 'relative_frequency': []}
    frequency_data = pandas.DataFrame(data=header)

    print("\nRELATIVE FREQUENCY")
    for feature in sorted(frequency_general, key=frequency_general.get, reverse=True):
        print(feature, ": ", frequency_general[feature]/features_total)
    print("POSITIVE FEATURES: ")
    for feature in sorted(frequency_pos, key=frequency_pos.get, reverse=True):
        print(feature, ": ", frequency_pos[feature]/features_total)
        row = [feature, "high", frequency_pos[feature]/features_total]
        frequency_data.loc[len(frequency_data)] = row
    print("NEGATIVE FEATURES: ")
    for feature in sorted(frequency_neg, key=frequency_neg.get, reverse=True):
        print(feature, ": ", frequency_neg[feature]/features_total)
        row = [feature, "low", frequency_neg[feature]/features_total]
        frequency_data.loc[len(frequency_data)] = row

    # plot relative frequencies
    colors = ["#14645aff", "#bfcd53ff"]
    seaborn.set_palette(seaborn.color_palette(colors))
    seaborn.color_palette("pastel")
    ax = seaborn.barplot(x=frequency_data.index, y="relative_frequency", hue="level", data=frequency_data, dodge=False)
    ax.set_xticklabels(frequency_data["feature"])
    pyplot.xticks(rotation=90)
    pyplot.tight_layout()
    pyplot.xlabel("")
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyplot.savefig("/".join([path, "_".join([file_name.replace(".csv", ""), "frequency_plot.png"])]),
                   bbox_inches="tight")


def preproc_data(train_names, val_names, m_segments, bin_alpha, bin_lambda):

    """

    Pre-processes train and validation data.

    Pre-processing includes reading, discretizing and filtering data according to binarization parameters.
    Pre-processed data sets are saved to files.

    Parameters
    ----------
    train_names : list
        list of paths to train data sets

    val_names : list
        list of paths to validation data sets

    m_segments : int
        number of segments for discretization

    bin_alpha : float
        alpha parameter for discretization

    bin_lambda : float
        lambda parameter for discretization

    """

    train_datasets = []  # train data sets
    val_datasets = []  # test data sets

    for i in range(0, len(train_names)):  # iterate over data sets

        # read data sets
        dataset_t, annotation, negatives, positives, features = preproc.read_data(train_names[i])  # read train data
        train_datasets.append(dataset_t)  # add train data to list of train data sets
        dataset_v, annotation, negatives, positives, features = preproc.read_data(val_names[i])  # read val data
        val_datasets.append(dataset_v)  # add val data to list of val data sets

    # discretize train and validation data sets
    train_datasets_bin, val_datasets_bin, feature_cdds \
        = preproc.discretize_data_for_tests(train_datasets, val_datasets, m_segments, bin_alpha, bin_lambda, False)

    # filter train data sets
    train_datasets_bin_f = []

    # data filtering
    [train_datasets_bin_f.append(preproc.remove_irrelevant_features(dataset)[0]) for dataset in train_datasets_bin]

    # save data sets to files
    for i in range(0, len(train_names)):
        # create name for train data
        name = \
            str(train_names[i].replace(".csv", "_"+str(m_segments)+"_"+str(bin_alpha)+"_"+str(bin_lambda)+"_bin.csv"))
        header_t = list(train_datasets_bin_f[i].columns)
        # remove -, . and : from the header (ASP solver cannot parse such symbols)
        header_t = list([colname.replace("-", "x") for colname in header_t])
        header_t = list([colname.replace(".", "y") for colname in header_t])
        header_t = list([colname.replace(":", "z") for colname in header_t])
        train_datasets_bin_f[i].columns = header_t
        dataset_t = train_datasets_bin_f[i]
        dataset_t.to_csv(name, sep=";", index=False)  # save discretized and filtered train data to file

        # create name for validation data
        name = str(val_names[i].replace(".csv", "_"+str(m_segments)+"_"+str(bin_alpha)+"_"+str(bin_lambda)+"_bin.csv"))
        header_v = list(val_datasets_bin[i].columns)
        # remove -, . and : from the header (ASP solver cannot parse such symbols)
        header_v = list([colname.replace("-", "x") for colname in header_v])
        header_v = list([colname.replace(".", "y") for colname in header_v])
        header_v = list([colname.replace(":", "z") for colname in header_v])
        val_datasets_bin[i].columns = header_v
        val_datasets_bin[i].to_csv(name, sep=";", index=False)  # save discretized validation data to file


def read_classifier(classifier, threshold):

    rules = classifier.split('  ')

    rule_set = []

    for r in rules:

        neg_inputs = []
        pos_inputs = []

        temp_rule = r.replace('[', '').replace(']', '')
        temp_rule = temp_rule.replace('(', '').replace(')', '')

        if 'AND' in temp_rule:
            temp_inputs = temp_rule.split('AND')
            for i in temp_inputs:
                if 'NOT' in i:
                    temp_input = i.replace('NOT', '')
                    neg_inputs.append(temp_input.strip())
                else:
                    temp_input = i
                    pos_inputs.append(temp_input.strip())

            rule = popinit.SingleRule(pos_inputs, neg_inputs, 1)
            rule_set.append(rule)

        elif 'OR' in temp_rule:
            temp_inputs = temp_rule.split('OR')
            pos_inputs = [i.strip() for i in temp_inputs]
            neg_inputs = []
            rule = popinit.SingleRule(pos_inputs, neg_inputs, 0)
            rule_set.append(rule)

        else:
            if 'NOT' in temp_rule:
                temp_rule = temp_rule.replace('NOT', '')
                pos_inputs = []
                neg_inputs = [temp_rule.strip()]
                rule = popinit.SingleRule(pos_inputs, neg_inputs, 1)
                rule_set.append(rule)
            else:
                pos_inputs = [temp_rule]
                neg_inputs = []
                rule = popinit.SingleRule(pos_inputs, neg_inputs, 1)
                rule_set.append(rule)

    classifier = popinit.Classifier(rule_set, evaluation_threshold=threshold, theta=0, errors={}, error_rates={},
                                    score=0, bacc=0, cdd_score=0, additional_scores={})
    return classifier


def read_classifiers_from_file(path):

    file_reader = csv.reader(open(path, 'r'), delimiter=';')

    classifier_population = []

    for row in file_reader:
        classifier, threshold = row
        classifier_formatted = read_classifier(classifier, threshold)
        classifier_population.append(classifier_formatted)

    return classifier_population


def evaluate_classifiers_from_external_file(path_to_classifiers, path_to_data):

    classifier_population = read_classifiers_from_file(path_to_classifiers)
    dataset, annotation, negatives, positives, features = preproc.read_data(path_to_data)

    test_bacc_avg = []
    test_tpr_avg = []
    test_tnr_avg = []
    test_fpr_avg = []
    test_fnr_avg = []
    test_f1_avg = []
    test_mcc_avg = []
    test_ppv_avg = []
    test_fdr_avg = []

    for classifier in classifier_population:
        classifier_score, bacc, errors, error_rates, additional_scores, cdd_score = \
            eval.evaluate_classifier(classifier, dataset, annotation, negatives, positives, [], True, 1.0)

        classifier.bacc = bacc
        classifier.errors = errors
        classifier.error_rates = error_rates
        classifier.additional_scores = additional_scores
        classifier.cdd_score = cdd_score
        classifier.score = classifier_score

        test_bacc_avg.append(bacc)

        test_tpr_avg.append(error_rates["tpr"])
        test_tnr_avg.append(error_rates["tnr"])
        test_fpr_avg.append(error_rates["fpr"])
        test_fnr_avg.append(error_rates["fnr"])

        test_f1_avg.append(additional_scores["f1"])
        test_mcc_avg.append(additional_scores["mcc"])
        test_ppv_avg.append(additional_scores["ppv"])
        test_fdr_avg.append(additional_scores["fdr"])

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

