import sys
import pandas
import numpy
import math
from decimal import Decimal, ROUND_HALF_UP

numpy.random.seed(1)


# get the number of samples, annotation, number of negatives and positives in the data
def get_data_info(dataset, header):

    """

    Extracts information about the data set.

    The function returns the number of samples, data annotation, the number of positive and negative samples in the
    data set.

    Parameters
    ----------
    dataset : Pandas DataFrame
        data set including sample ids, annotation and feature profiles
    header : list
        data frame header

    Returns
    -------
    samples : int
        number of samples
    annotation : list
        binary annotation of samples
    negatives : int
        number of negative samples in the data
    positives : int
        number of positive samples in the data

    """

    samples = len(dataset.index)  # get number of samples
    annotation = list(dataset[header[1]])  # get annotation
    negatives = annotation.count(0)  # count negative samples
    positives = samples - negatives  # count positive samples

    return samples, annotation, negatives, positives


# reading binarized data set
# reading data set
def read_data(dataset_filename):

    """

    Reads the data.

    The function check the correctness of data set and returns the data set as a data frame, the number of samples,
    data annotation, the number of positive and negative samples in data as well as a list of features.

    Parameters
    ----------
    dataset_filename : str
        data set file name

    Returns
    -------
    dataset : str
        path to a data set file
    samples : int
        number of samples
    annotation : list
        annotation of samples
    negatives : int
        number of negative samples in the data
    positives : int
        number of positive samples in the data
    features : list
        list of features in the data set

    """

    # reading the data
    # throws an exception when datafile not found
    try:
        dataset = pandas.read_csv(dataset_filename, sep=';', header=0)
        # rename the header with the first row
        # must be done to check whether data contains duplicates of features
        #dataset = dataset.rename(columns=dataset.iloc[0], copy=False).iloc[1:].reset_index(drop=True)
        #header = dataset.columns.values.tolist()

        # check whether sample IDs are unique
        #ids = dataset[header[0]]  # get sample IDs
        #if len(ids) > len(set(ids)):  # compare length of the set with the length of the ids list
            #print("Error: IDs of samples must be unique. Note, the first column must include IDs of samples and "
                  #"\nthe second their annotation. Annotate samples as follows: 0 - negative class, 1 - positive class.")
            #sys.exit(0)

    except IOError:
        print("Error: ", dataset_filename, " not found.")
        sys.exit(0)

    # if file found read the data set with header
    # this allows to avoid type conversion
    dataset = pandas.read_csv(dataset_filename, sep=';', header=0)
    header = dataset.columns.values.tolist()

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
    samples, annotation, negatives, positives = get_data_info(dataset, header)

    print("DATA SET INFO")
    print("Number of samples in the data set: " + str(samples))
    print("Number of negative samples: " + str(negatives))
    print("Number of positive samples: " + str(positives))

    return dataset, annotation, negatives, positives, features


# removal of irrelevant (non-regulated) features
def remove_irrelevant_features(dataset):

    """

    Removes irrelevant features from the data set.

    The function filters relevant features (expressing differences between the classes) and removes irrelevant ones
    from the data set.

    Parameters
    ----------
    dataset : Pandas DataFrame
        data set

    Returns
    -------
    dataset : Pandas DataFrame
        updated data set
    relevant_features : list
        list of relevant features

    """

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
def discretize_feature(feature_levels, annotation, negatives, positives, m_segments, alpha_param, lambda_param):

    """

    Discretizes single feature.

    The function discretizes single feature according to
    `Wang et al. (2014) <https://doi.org/10.1016/j.neucom.2014.04.064>`_. The implementation allows to discretize a
    feature into two states (originally in Wang et al. into three states): the feature is either not regulated
    (one class) or regulated (two classes). The function returns information about the number of features
    that may be discretized into three states.

    Parameters
    ----------
    feature_levels : list
        feature levels
    annotation : list
        binary annotation of samples
    negatives : int
        number of negative samples
    positives : int
        number of positive samples
    m_segments : int
        number of discretization segments
    alpha_param : float
        binarization alpha
    lambda_param : float
        binarization lambda

    Returns
    -------
    threshold : float
        discretization threshold
    global_cdd : float
        feature class distribution diversity
    pattern : int
        discretization pattern (0 - one class, 1 - two classes, 2 - three classes)


    """

    # sort miRNA expression levels and annotation
    feature_levels_sorted, annotation_sorted = zip(*sorted(zip(feature_levels, annotation)))

    # calculate segment step
    segment_step = (max(feature_levels) - min(feature_levels)) / m_segments

    # class diversity distributions
    cdds = []
    segment_thresholds = []

    # calculate segment thresholds
    for m in range(1, m_segments+1):

        if m == m_segments:  # if this is the last segment
            segment_threshold = Decimal(max(feature_levels))  # add max level as threshold
        else:  # otherwise calculate the segment threshold
            segment_threshold = Decimal(min(feature_levels)) + Decimal(segment_step) * m

        segment_thresholds.append(segment_threshold)  # store segment threshold
        # add all the levels between min level and threshold to the segment
        segment = [i for i in feature_levels_sorted if Decimal(i) <= Decimal(segment_threshold)]

        neg_class = annotation_sorted[0:len(segment)].count(0)  # calculate number of negative samples in segment
        pos_class = annotation_sorted[0:len(segment)].count(1)  # calculate number of positive samples in segment

        cdd = neg_class / negatives - pos_class / positives  # calculate cdd
        cdds.append(cdd)  # store cdd

    # max and min cdd
    cdd_max = max(cdds)
    cdd_min = min(cdds)

    # absolute max and min cdds
    cdd_max_abs = math.fabs(max(cdds))
    cdd_min_abs = math.fabs(min(cdds))

    # calculate global cdd
    global_cdd = math.fabs(cdd_max-cdd_min)

    # assign threshold = None
    threshold = None
    pattern = 0
    index = 0

    # one state miRNAs
    if global_cdd < alpha_param or max(cdd_max_abs, cdd_min_abs) < lambda_param:  # critetion 1
        threshold = None
        pattern = 0

    # two states miRNAs
    if global_cdd >= alpha_param:  # criterion 2
        if max(cdd_max_abs, cdd_min_abs) >= lambda_param:
            if min(cdd_max_abs, cdd_min_abs) < lambda_param:

                pattern = 1
                if cdd_max_abs > cdd_min_abs:
                    index = cdds.index(cdd_max)

                if cdd_max_abs <= cdd_min_abs:
                    index = cdds.index(cdd_min)

                threshold = segment_thresholds[index]  # find threshold

    # complicated patterns
    if global_cdd >= alpha_param and min(cdd_max_abs, cdd_min_abs) >= lambda_param:  # criterion 3
        threshold = None
        pattern = 2

    return threshold, global_cdd, pattern


# discretize train data set
def discretize_train_data(train_dataset, m_segments, alpha_param, lambda_param, print_results):

    """

    Discretizes all features in a training data set.

    Parameters
    ----------
    train_dataset : str/Pandas DataFrame
        path to train data file (str) or already read data set (Pandas DataFrame)
    m_segments : int
        number of discretization segments
    alpha_param : float
        binarization alpha
    lambda_param : float
        binarization lambda
    print_results : bool
        if True the discretization information is shown

    Returns
    -------
    data_discretized : Pandas DataFrame
        discretized training data
    threshold : float
        discretization threshold
    global_cdd : float
        feature class distribution diversity
    pattern : int
        discretization pattern (0 - one class, 1 - two classes, 2 - three classes)

    """

    if isinstance(train_dataset, pandas.DataFrame):
        dataset = train_dataset.__copy__()
        header = dataset.columns.values.tolist()
        samples, annotation, negatives, positives = get_data_info(dataset, header)
    else:
        # read data
        dataset, annotation, negatives, positives, features = read_data(train_dataset)

    # create a new name for a discretized data set
    new_file = str(train_dataset.replace(".csv", "")) + "_discretized_" + str(m_segments) \
               + "_" + str(alpha_param) + "_" + str(lambda_param)

    # get feature IDs
    features = dataset.columns.values.tolist()[2:]

    # list of discretization thresholds
    thresholds = []

    # global feature cdds
    global_cdds = []

    # drop all the feature from the train data set, keep the sample IDs and annotation
    data_discretized = dataset.drop(features, axis=1)

    # count feature with one/two states or another pattern
    one_state_features = 0
    two_states_features = 0
    other_pattern_features = 0

    relevant_features = []  # list of relevant features
    feature_discretized = 0

    # iterate over miRNAs
    for feature in features:

        # get single miRNA gene expression levels
        feature_levels = dataset[feature].tolist()

        # discretize miRNA
        threshold, global_cdd, pattern = discretize_feature(feature_levels, annotation, negatives, positives,
                                                            m_segments, alpha_param, lambda_param)

        # count miRNA expression patterns
        if pattern == 0:
            one_state_features += 1
            feature_discretized = [0 for i in feature_levels]  # set all values to 0

        if pattern == 1:
            two_states_features += 1
            # discretize miRNAs according to thresholds
            feature_discretized = [0 if i < threshold else 1 for i in feature_levels]
            relevant_features.append(feature)

        if pattern == 2:
            other_pattern_features += 1
            feature_discretized = [1 for i in feature_levels]  # set all values to 1

        # add threshold to list of thresholds and global cdd to list of global cdds
        thresholds.append(threshold)
        global_cdds.append(global_cdd)

        # add discretized feature to the data
        data_discretized[feature] = feature_discretized

    # create a dictrionary of features and cdds
    cdds = dict(zip(features, global_cdds))

    if print_results is True:
        print("feature CDDs:")
        for feature in relevant_features:
            print(feature, " : ", cdds[feature])

    cdd_list = [cdds[feature] for feature in relevant_features]

    print("\nDISCRETIZATION RESULTS")
    print("ONE STATE FEATURES: ", one_state_features)
    print("TWO STATE FEATURES: ", two_states_features)
    print("OTHER PATTERN FEATURES: ", other_pattern_features)
    print("AVG CDD: ", numpy.average(cdd_list))
    print("STD CDD: ", numpy.std(cdd_list, ddof=1))
    print("AVG CDD ALL: ", numpy.average(list(cdds.values())))
    print("STD CDD ALL: ", numpy.std(list(cdds.values()), ddof=1))

    if not isinstance(train_dataset, pandas.DataFrame):
        # write data to a file
        data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return data_discretized, features, thresholds, cdds


# discretize test data
def discretize_test_data(test_dataset, thresholds):

    """

    Discretizes all features in a test data set according to the given thresholds.

    Parameters
    ----------
    test_dataset : str/Pandas DataFrame
        path to train data file (str) or already read data set (Pandas DataFrame)
    thresholds : list
        list of feature thresholds

    Returns
    -------
    data_discretized : Pandas DataFrame
        discretized test data

    """

    # if the data set is a dataframe copy it
    if isinstance(test_dataset, pandas.DataFrame):
        dataset = test_dataset.__copy__()
    else:
        # otherwise read data
        dataset, annotation, negatives, positives, features = read_data(test_dataset)

    # create a new name for a discretized data set
    new_file = str(test_dataset.replace(".csv", "")) + "_discretized"

    # get feature IDs
    features = dataset.columns.values.tolist()[2:]

    # drop all the features from the data set (keep the ids and annotation)
    data_discretized = dataset.drop(features, axis=1)

    # zip features and thresholds into a dictionary
    threshold_dict = dict(zip(features, thresholds))

    feature_discretized = 0

    # iterate over miRNAs
    for feature in features:

        # get single feature gene expression levels
        feature_levels = dataset[feature].tolist()

        if threshold_dict[feature] is None:  # if there is no threshold
            feature_discretized = [0 for i in feature_levels]  # discretize into one state
        else:
            # discretize features according to thresholds
            feature_discretized = [0 if i < threshold_dict[feature] else 1 for i in feature_levels]

        # add discretized miRNAs to the data
        data_discretized[feature] = feature_discretized

    if not isinstance(test_dataset, pandas.DataFrame):
        # write data set to file
        data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return data_discretized


# discretize several training and testing data sets (use lists: training_fold_list, testing_fold_list)
def discretize_data_for_tests(training_fold_list, validation_fold_list, m_segments, alpha_param, lambda_param,
                              print_results):

    """

    Discretizes all training and validation data sets.

    The function discretizes training and validation data sets according to `Wang et al. (2014)
    <https://doi.org/10.1016/j.neucom.2014.04.064>`_.

    Parameters
    ----------
    training_fold_list : list
        list of training data sets
    validation_fold_list : list
        list of validation data sets
    m_segments : int
        number of discretization segments
    alpha_param : float
        binarization alpha
    lambda_param : float
        binarization lambda
    print_results : bool
        if True detailed discretization information is shown

    Returns
    -------
    discretized_train_data : list
        list of discretized training data sets
    discretized_test_data : list
        list of discretized validation data sets
    feature_cdds : list
        list of feature cdds

    """

    discretized_train_data = []  # list of discretized train fractions
    discretized_test_data = []  # list of discretized test fractions
    feature_cdds = []  # list of feature's cdds

    fold = 1  # count folds

    # iterate over train and test data
    for training_fold, testing_fold in zip(training_fold_list, validation_fold_list):

        print("\nDISCRETIZATION: ", fold)
        fold += 1

        # discretize train data and return thresholds
        train_data_discretized, features, thresholds, cdds \
            = discretize_train_data(training_fold, m_segments, alpha_param, lambda_param, print_results)

        discretized_train_data.append(train_data_discretized)  # add discretized fold to the list
        feature_cdds.append(cdds)  # add feature cdds

        # discretize test data according to thresholds
        test_data_discretized = discretize_test_data(testing_fold, thresholds)

        discretized_test_data.append(test_data_discretized)  # add discretized fold to the list

    return discretized_train_data, discretized_test_data, feature_cdds

