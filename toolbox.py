import preproc
import tuner
import pandas
import random

random.seed(1)


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
def divide_into_train_test(dataset_filename, train_frac):

    dataset, annotation, negatives, positives, features = preproc.read_data(dataset_filename)

    data_size = len(dataset.index)

    # copy samples
    negative_samples = dataset.iloc[:negatives].copy()
    positive_samples = dataset.iloc[negatives:data_size].copy()

    # calculate how many samples go to training set
    negative_samples_to_draw = int(round(negatives*train_frac/100))
    positive_samples_to_draw = int(round(positives*train_frac/100))

    print("TRAINING DATA SET: ", positive_samples_to_draw + negative_samples_to_draw, " samples")
    print("TESTING DATA SET: ", data_size - (positive_samples_to_draw + negative_samples_to_draw), " samples")

    # draw positive training samples
    training_positives = positive_samples.sample(n=positive_samples_to_draw)  # draw n samples
    training_positives.sort_index()  # sort samples
    ids_to_drop = training_positives.index.values  # get the ids of positive training samples
    testing_positives = positive_samples.drop(ids_to_drop)  # add rest as test samples

    # draw negative training samples
    training_negatives = negative_samples.sample(n=negative_samples_to_draw)  # draw n samples
    training_negatives.sort_index()  # sort samples
    ids_to_drop = training_negatives.index.values  # get the ids of negative training samples
    testing_negatives = negative_samples.drop(ids_to_drop)  # add rest as test samples

    training_data = training_negatives.append(training_positives)  # merge negative and positive samples
    testing_data = testing_negatives.append(testing_positives)

    return training_data, testing_data


# division of the data set into kfolds
def divide_into_cv_folds(train_dataset_filename, dataset, kfolds, pairing, set_seed):

    header = dataset.columns.values.tolist()
    samples, annotation, negatives, positives = preproc.get_data_info(dataset, header)

    negative_samples = dataset.iloc[:negatives].copy()  # copy negative samples
    positive_samples = dataset.iloc[negatives:samples].copy()  # copy positive samples
    positive_samples_temp = []  # used only if pairing is True, allows to leave positive_samples untouched

    negative_samples_to_draw = int(round(negatives/kfolds))  # number of neg samples to draw
    positive_samples_to_draw = int(round(positives/kfolds))  # number of pos samples to draw

    negative_folds = []  # list of negative folds
    positive_folds = []  # list of positive folds

    train_datasets = []  # list of train folds
    val_datasets = []   # list of test folds

    if pairing is True:
        positive_samples_temp = dataset.iloc[negatives:samples].copy()

    for fold in range(1, kfolds):  # k-1 times draw positive and negative samples

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

    for fold in range(0, kfolds):  # create k validation folds

        val_fold = negative_folds[fold].append(positive_folds[fold])  # merge negative and positive samples
        val_datasets.append(val_fold)  # add to list of validation folds

        print("VALIDATION FOLD ", fold+1, len(val_fold.index.values))

    for fold in range(0, kfolds):  # create k train folds

        train_folds_to_merge = []  # list of folds to merge
        for i in range(0, kfolds):
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
        filename = train_dataset_filename.replace(".csv", new_name)
        train_set.to_csv(filename, sep=";", index=False)

        new_name = "_val_" + str(fold) + ".csv"  # validation fold name
        filename = train_dataset_filename.replace(".csv", new_name)
        val_set.to_csv(filename, sep=";", index=False)

        fold = fold + 1

    return train_datasets, val_datasets
