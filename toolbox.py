import preproc
import tuner
import pandas
import random

random.seed(1)


def write_config_to_log(config_file):

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
def divide_into_train_test(dataset_filename, train_fraction, set_seed):

    dataset, annotation, negatives, positives, features = preproc.read_data(dataset_filename)

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
def divide_into_cv_folds(dataset_file_name, dataset, k_folds, pairing, set_seed):

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

    if pairing is True:
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

        new_name = "_train_" + str(fold) + ".csv"  # train fold name
        filename = dataset_file_name.replace(".csv", new_name)
        train_set.to_csv(filename, sep=";", index=False)

        new_name = "_val_" + str(fold) + ".csv"  # validation fold name
        filename = dataset_file_name.replace(".csv", new_name)
        val_set.to_csv(filename, sep=";", index=False)

        fold = fold + 1

    return train_datasets, val_datasets


def remove_symmetric_solutions(best_classifiers):

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
        print("DELETING")
        for str in best_classifiers.solutions_str:
            print(str)
        print(to_del)

        # sort indices in descending order for removal
        to_del = list(set(to_del))
        to_del.sort(reverse=True)
        # remove duplicates
        for i in to_del:
            del best_classifiers.solutions[i]
            del best_classifiers.solutions_str[i]

        for str in best_classifiers.solutions_str:
            print(str)


def rank_features_by_frequency(solutions):

    frequency_general = {}  # count occurences in total
    frequency_pos = {}  # count occurences as positive inputs
    frequency_neg = {}  # count occurences as negative inputs

    features_total = 0

    for solution in solutions:  # for all solutions
        for rule in solution.rule_set:
            for i in rule.pos_inputs:
                if i not in frequency_pos.keys():
                    frequency_pos[i] = 1
                    frequency_general[i] = 1
                else:
                    frequency_pos[i] = frequency_pos[i] + 1
                    frequency_general[i] = frequency_general[i] + 1

            for i in rule.neg_inputs:
                if i not in frequency_neg.keys():
                    frequency_neg[i] = 1
                    frequency_general[i] = 1
                else:
                    frequency_neg[i] = frequency_neg[i] + 1
                    frequency_general[i] = frequency_general[i] + 1

        features_total = features_total + len(solution.get_input_list())  # total number of features in all solutions

    print("\n###FEATURE FREQUENCY ANALYSIS###")
    print("TOTAL VALUES")
    print("NUMBER OF FEATURES IN ALL SOLUTIONS IN TOTAL: ", features_total)
    print("POSITIVE FEATURES: ")
    for feature in sorted(frequency_pos, key=frequency_pos.get, reverse=True):
        print(feature, ": ", frequency_pos[feature])
    print("NEGATIVE FEATURES: ")
    for feature in sorted(frequency_neg, key=frequency_neg.get, reverse=True):
        print(feature, ": ", frequency_neg[feature])

    print("\nRELATIVE FREQUENCY")
    for feature in sorted(frequency_general, key=frequency_general.get, reverse=True):
        print(feature, ": ", frequency_general[feature]/features_total)
    print("POSITIVE FEATURES: ")
    for feature in sorted(frequency_pos, key=frequency_pos.get, reverse=True):
        print(feature, ": ", frequency_pos[feature]/features_total)
    print("NEGATIVE FEATURES: ")
    for feature in sorted(frequency_neg, key=frequency_neg.get, reverse=True):
        print(feature, ": ", frequency_neg[feature]/features_total)


