import sys
import pandas
import numpy
import math
from decimal import Decimal, ROUND_HALF_UP

numpy.random.seed(1)


# reading binarized data set
# reading data set
def read_data(dataset_filename):

    # reading the data
    # throws an exception when datafile not found
    try:
        dataset = pandas.read_csv(dataset_filename, sep=';', header=None)
    except IOError:
        print("Error: ", dataset_filename, " not found.")
        sys.exit(0)

    # rename the header with the first row
    # must be done to check whether data contains duplicates of features
    dataset = dataset.rename(columns=dataset.iloc[0], copy=False).iloc[1:].reset_index(drop=True)
    header = dataset.columns.values.tolist()

    # check whether sample IDs are unique
    ids = dataset[header[0]]  # get sample IDs
    if len(ids) > len(set(ids)):  # compare length of the set with the length of the ids list
        print("Error: IDs of samples must be unique. Note, the first column must include IDs of samples and the second"
              "\ntheir annotation. Annotate samples as follows: 0 - negative class, 1 - positive class.")
        sys.exit(0)

    # check whether second column contains correct annotation
    annotation = sorted(set(dataset[header[1]]))
    if len(annotation) != 2:
        print("Error: annotation consists of less or more than two classes.")
        sys.exit(0)

    if int(annotation[0]) != 0 or int(annotation[1]) != 1:
        print("Error: annotate samples as follows: 0 - negative class, 1 - positive class.")
        sys.exit(0)

    # check whether data includes feature duplicates
    features = header[2:]
    if len(features) > len(set(features)):
        print("Error: IDs of features must be unique!")
        sys.exit(0)

    # counting negative and positive samples
    samples = len(dataset.index)
    annotation = list(dataset[header[1]])
    negatives = annotation.count('0')  # count negative samples
    positives = samples - negatives  # count positive samples

    print("DATA SET INFO")
    print("Number of samples in the data set: " + str(samples))
    print("Number of negative samples: " + str(negatives))
    print("Number of positive samples: " + str(positives))

    return dataset, negatives, positives, features


# removal of irrelevant (non-regulated) features
def remove_irrelevant_features(dataset):

    relevant_features = []  # list of relevant features
    irrelevant_features = []  # list of irrelevant features

    # get header
    header = dataset.columns.values.tolist()

    # sum of feature levels (0/1) in each column
    column_sum = dataset.sum(axis=0, skipna=True)

    # if feature levels sum up to 0 or the number of samples in the dataset - feature is irrelevant (non-regulated)
    # (in other words: the whole column is filled in with 0s or 1s)
    for id, sum in column_sum.items():
        if id not in [header[0], header[1]]:  # skip IDs and annotation
            if sum == 0 or sum == len(dataset.index):
                irrelevant_features.append(id)
            else:
                relevant_features.append(id)

    # removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_features, axis=1)

    # creating log message
    print("\n##FILTERING FEATURES##")
    print("Number of relevant features according to a given threshold: ", str(len(relevant_features)))
    print("Number of irrelevant features according to a given threshold: ", str(len(irrelevant_features)))

    return dataset, relevant_features


# discretize miRNA expression levels
# discretization according to Wang et al. (2014)
# Wang, H.-Q.et al.(2014). Biology-constrained gene expression discretization for cancer classification.
# Neurocomputing, 145, 30–36.
def discretize_miRNA(feature_levels, annotation, negatives, positives, m_segments, alpha_param, lambda_param):

    # sort miRNA expression levels and annotation
    feature_levels_sorted, annotation_sorted = zip(*sorted(zip(feature_levels, annotation)))

    # calculate segment step
    segment_step = (max(feature_levels) - min(feature_levels)) / m_segments

    # class diversity distributions
    cdds = []

    # calculate segment thresholds
    for m in range(1, m_segments+1):

        if m == m_segments:
            segment_threshold = Decimal(max(feature_levels))
        else:
            segment_threshold = Decimal(min(feature_levels)) + Decimal(segment_step) * m
        segment = [i for i in feature_levels_sorted if Decimal(i) <= Decimal(segment_threshold)]

        neg_class = annotation_sorted[0:len(segment)].count(0)
        pos_class = annotation_sorted[0:len(segment)].count(1)

        cdd = neg_class / negatives - pos_class / positives
        cdds.append(cdd)

    # max and min cdd
    cdd_max = max(cdds)
    cdd_min = min(cdds)

    # absolute max and min cdds
    cdd_max_abs = math.fabs(max(cdds))
    cdd_min_abs = math.fabs(min(cdds))

    # difference between absolute values of max and min cdda
    global_cdd = math.fabs(cdd_max-cdd_min)

    # assign threshold = 0
    threshold = None
    pattern = 0

    # one state miRNAs
    if global_cdd < alpha_param or max(cdd_max_abs, cdd_min_abs) < lambda_param:
        threshold = None
        pattern = 0

    # two states miRNAs
    if global_cdd >= alpha_param:
        if max(cdd_max_abs, cdd_min_abs) >= lambda_param:
            if min(cdd_max_abs, cdd_min_abs) < lambda_param:

                pattern = 1
                if cdd_max_abs > cdd_min_abs:
                    index = cdds.index(cdd_max) + 1
                    threshold = min(feature_levels) + segment_step * index
                    print(threshold)

                if cdd_max_abs <= cdd_min_abs:
                    index = cdds.index(cdd_min) + 1
                    threshold = min(feature_levels) + segment_step * index

    # complicated patterns
    if global_cdd >= alpha_param and min(cdd_max_abs, cdd_min_abs) >= lambda_param:
        threshold = None
        pattern = 2

    return threshold, global_cdd, pattern


# discretize train data set
def discretize_train_data(train_dataset, m_segments, alpha_param, lambda_param, print_results):

    if isinstance(train_dataset, pandas.DataFrame):
        dataset = train_dataset.__copy__()
        header = dataset.columns.values.tolist()
        samples = len(train_dataset.index)
        negatives = train_dataset[train_dataset[header[1]] == 0].count()[header[1]]
        positives = samples - negatives
    else:
        # read data
        dataset, negatives, positives, features = read_data(train_dataset)
        header = dataset.columns.values.tolist()

    # create a new name for a discretized data set
    new_file = str(train_dataset.replace(".csv", "")) + "_discretized_" + str(m_segments) \
               + "_" + str(alpha_param) + "_" + str(lambda_param)

    # list of sample annotation
    header = dataset.columns.values.tolist()
    annotation = dataset[header[1]].tolist()

    # get miRNA IDs
    features = dataset.columns.values.tolist()[2:]

    # list of discretization thresholds
    thresholds = []

    # global miRNA cdds
    global_cdds = []

    # drop all the miRNAs from the train data set, keep the sample IDs and annotation
    data_discretized = dataset.drop(features, axis=1)

    # count miRNAs with one/two states or another complicated pattern
    one_state_features = 0
    two_states_features = 0
    other_pattern_features = 0

    relevant = []  # list of relevant features
    feature_discretized = 0

    # iterate over miRNAs
    for feature in features:

        # get single miRNA gene expression levels
        feature_levels = dataset[feature].tolist()

        # discretize miRNA
        threshold, global_cdd, pattern = discretize_miRNA(feature_levels, annotation, negatives, positives, m_segments,
                                                          alpha_param, lambda_param)

        # count miRNA expression patterns
        if pattern == 0:
            one_state_features += 1
            feature_discretized = [0 for i in feature_levels]  # set all values to 0
        if pattern == 1:
            two_states_features += 1
            relevant.append(feature)
            # discretize miRNAs according to thresholds
            feature_discretized = [0 if i <= threshold else 1 for i in feature_levels]
        if pattern == 2:
            other_pattern_features += 1
            feature_discretized = [1 for i in feature_levels]  # set all values to 1

        # add threshold to a list of thresholds
        thresholds.append(threshold)
        global_cdds.append(global_cdd)

        # add discretized miRNAs to the data
        data_discretized[feature] = feature_discretized

    # create a dictrionary of miRNAs and its cdds
    miRNA_cdds = dict(zip(features, global_cdds))

    if print_results is True:
        print("feature CDDs:")
        for feature in relevant:
            print("feature ", feature, " : ", miRNA_cdds[feature])

    cdd_list = [miRNA_cdds[miRNA] for miRNA in relevant]
    print("\nDISCRETIZATION RESULTS")
    print("ONE STATE FEATURES: ", one_state_features)
    print("TWO STATE FEATURES: ", two_states_features)
    print("COMPLICATED FEATURES: ", other_pattern_features)
    print("AVG CDD: ", numpy.average(cdd_list))
    print("STD CDD: ", numpy.std(cdd_list))
    print("AVG CDD ALL: ", numpy.average(list(miRNA_cdds.values())))
    print("STD CDD ALL: ", numpy.std(list(miRNA_cdds.values())))

    if not isinstance(train_dataset, pandas.DataFrame):
        # write data to a file
        data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return data_discretized, features, thresholds, miRNA_cdds


# discretize test data
def discretize_test_data(test_dataset, thresholds):

    if isinstance(test_dataset, pandas.DataFrame):
        dataset = test_dataset.__copy__()
    else:
        # read data
        dataset, negatives, positives, features = read_data(test_dataset)

    # create a new name for a discretized data set
    new_file = str(test_dataset.replace(".csv", "")) + "_discretized"

    # get feature IDs
    features = dataset.columns.values.tolist()[2:]

    # drop all the features from the data set
    data_discretized = dataset.drop(features, axis=1)

    # zip features and thresholds into a dictionary
    feature_dict = dict(zip(features, thresholds))

    feature_discretized = 0

    # iterate over miRNAs
    for feature in features:

        # get single miRNA gene expression levels
        feature_levels = dataset[feature].tolist()

        if feature_dict[feature] == None:
            feature_discretized = [1 for i in feature_levels]
        else:
            # discretize miRNAs according to thresholds
            feature_discretized = [0 if i <= feature_dict[feature] else 1 for i in feature_levels]

        # add discretized miRNAs to the data
        data_discretized[feature] = feature_discretized

    if not isinstance(test_dataset, pandas.DataFrame):
        # write data set to file
        data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return data_discretized


# discretize several data sets
def discretize_data_for_tests(train_list, test_list, m_segments, alpha_param, lambda_param, print_results):

    discretized_train_data = []
    discretized_test_data = []
    feature_cdds = []

    fold = 1

    # iterate over train and test data
    for train, test in zip(train_list, test_list):

        print("\nDISCRETIZATION: ", fold)
        fold += 1

        # discretize train data and return thresholds
        data_discretized, features, thresholds, feature_cdds \
            = discretize_train_data(train, m_segments, alpha_param, lambda_param, print_results)

        discretized_train_data.append(data_discretized)  # add fold to the list

        # discretize test data avvording to thresholds
        data_discretized = discretize_test_data(test, thresholds)

        discretized_test_data.append(data_discretized)  # add fold to the list

    return discretized_train_data, discretized_test_data, feature_cdds

