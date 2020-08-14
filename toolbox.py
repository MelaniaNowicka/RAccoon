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
def divide_into_cv_folds(dataset, kfolds):

    header = dataset.columns.values.tolist()
    samples, annotation, negatives, positives = preproc.get_data_info(dataset, header)

    negative_samples = dataset.iloc[:negatives].copy()  # copy negative samples
    positive_samples = dataset.iloc[negatives:samples].copy()  # copy positive samples

    negative_samples_to_draw = int(round(negatives/kfolds))  # number of neg samples to draw
    positive_samples_to_draw = int(round(positives/kfolds))  # number of pos samples to draw

    negative_folds = []  # list of negative folds
    positive_folds = []  # list of positive folds

    train_datasets = []  # list of train folds
    test_datasets = []   # list of test folds

    for fold in range(1, kfolds):  # k-1 times draw positive and negative samples

        positive_data_fold = positive_samples.sample(n=positive_samples_to_draw)  # draw n samples
        positive_folds.append(positive_data_fold.sort_index())  # add fold to positive folds
        ids_to_drop = positive_data_fold.index.values  # drop drawn ids
        positive_samples.drop(ids_to_drop, inplace=True)

        negative_data_fold = negative_samples.sample(n=negative_samples_to_draw)
        negative_folds.append(negative_data_fold.sort_index())
        ids_to_drop = negative_data_fold.index.values
        negative_samples.drop(ids_to_drop, inplace=True)

    positive_folds.append(positive_samples)  # add remaining samples as last folds
    negative_folds.append(negative_samples)

    val_folds = []

    for fold in range(0, kfolds):  # create k test folds
        val_fold = negative_folds[fold].append(positive_folds[fold])
        val_folds.append(val_fold)

        print("VALIDATION FOLD ", fold+1, len(val_fold.index.values))

        test_datasets.append(val_fold)

    for fold in range(0, kfolds):  # create train folds

        train_folds_to_merge = []
        for i in range(0, kfolds):
            if i != fold:
                train_folds_to_merge.append(val_folds[i])

        train_fold = train_folds_to_merge[0].copy()
        for i in range(1, len(train_folds_to_merge)):
            train_fold = train_fold.append(train_folds_to_merge[i])

        print("TRAIN FOLD ", fold + 1, len(train_fold.index.values))
        train_fold = train_fold.sort_index()

        train_datasets.append(train_fold)

    return train_datasets, test_datasets
