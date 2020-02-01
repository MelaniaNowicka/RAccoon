import sys
import pandas
import math
import toolbox
from decimal import Decimal, ROUND_HALF_UP


# reading binarized data set.
def read_data(dataset_filename, log_message):

    # reading the data
    # throws an exception when datafile not found
    try:
        dataset = pandas.read_csv(dataset_filename, sep=';', header=0)
    except IOError:
        print("Error: No such file or directory.")
        sys.exit(0)

    # simple check whether data is in the right format
    # needs to be improved
    header = dataset.columns.values.tolist()

    if header[0] != 'ID' or header[1] != 'Annots':
        print("Error: wrong format. The first column must include sample IDs and the second "
              "- the annotation of samples.")
        sys.exit(0)

    # extract miRNA's names
    mirnas = header[2:]
    # counting negative and positive samples
    samples = len(dataset.index)
    negatives = dataset[dataset["Annots"] == 0].count()["Annots"]
    positives = samples - negatives

    log_message = log_message + "Number of samples: " + str(samples) + "\n"
    log_message = log_message + "Number of negative samples: " + str(negatives) + "\n"
    log_message = log_message + "Number of positive samples: " + str(positives) + "\n\n"

    print("Number of samples: " + str(samples))
    print("Number of negative samples: " + str(negatives))
    print("Number of positive samples: " + str(positives))

    if negatives == 0 or positives == 0:
        print("Error: no negative or positive samples in the dataset!")
        sys.exit(0)

    return dataset, negatives, positives, mirnas, log_message


# removal of irrelevant (non-regulated) miRNAs (filled with only 0/1).
def remove_irrelevant_mirna(dataset, log_message):

    relevant_mirna = []
    irrelevant_mirna = []

    # sum of miRNA levels (0/1) in each column
    column_sum = dataset.sum(axis=0, skipna=True)

    # if miRNA levels sum up to 0 or the number of samples in the dataset - miRNA is irrelevant (non-regulated)
    # (in other words: the whole column is filled in with 0s or 1s)
    for id, sum in column_sum.items():
        if id not in ["ID", "Annots"]:
            if sum == 0 or sum == len(dataset.index):
                irrelevant_mirna.append(id)
            else:
                relevant_mirna.append(id)

    # removing irrelevant miRNAs from the dataset
    dataset = dataset.drop(irrelevant_mirna, axis=1)

    # creating log message
    log_message = log_message + "Number of relevant miRNAs according to a given threshold: " + str(len(relevant_mirna)) \
                  + "\n"
    log_message = log_message + "Number of irrelevant miRNAs according to a given threshold: " \
                  + str(len(irrelevant_mirna)) + "\n\n"

    log_message = log_message + "Relevant miRNAs: "

    for i in relevant_mirna:
        log_message = log_message + str(i) + "; "

    log_message = log_message + "\n\n"

    return dataset, relevant_mirna, log_message


# discretize miRNA expression levels
# discretization according to Wang et al. (2014)
# add citation
def discretize_miRNA(miR_expr, annots, negatives, positives, m_segments, alpha_param, lambda_param):

    # sort miRNA expression levels and annotation
    miR_expr_sorted, annots_sorted = zip(*sorted(zip(miR_expr, annots)))

    # calculate segment step
    segment_step = (max(miR_expr) - min(miR_expr))/m_segments

    # segments
    segments = []

    # class diversity distribution
    cdds = []


    # calculate segment thresholds
    for m in range(1, m_segments+1):

        if(m==m_segments): segment_threshold = Decimal(max(miR_expr))
        else: segment_threshold = Decimal(min(miR_expr)) + Decimal(segment_step)*m
        segment = [i for i in miR_expr_sorted if Decimal(i) <= Decimal(segment_threshold)]
        segments.append(segment)

        neg_class = annots_sorted[0:len(segment)].count(0)
        pos_class = annots_sorted[0:len(segment)].count(1)

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
    threshold = 0

    # one state miRNAs
    if global_cdd < alpha_param or max(cdd_max_abs, cdd_min_abs) < lambda_param:
        threshold = 0

    # two states miRNAs
    if global_cdd >= alpha_param:
        if max(cdd_max_abs, cdd_min_abs) >= lambda_param:
            if min(cdd_max_abs, cdd_min_abs) < lambda_param:

                if cdd_max_abs > cdd_min_abs:
                    index = cdds.index(cdd_max) + 1
                    threshold = min(miR_expr) + segment_step * index

                if cdd_max_abs <= cdd_min_abs:
                    index = cdds.index(cdd_min) + 1
                    threshold = min(miR_expr) + segment_step * index

    # complicated patterns
    if global_cdd >= alpha_param and min(cdd_max_abs, cdd_min_abs) >= lambda_param:
        threshold = -1

    return threshold, global_cdd


# discretize train data set
def discretize_train_data(train_dataset, m_segments, alpha_param, lambda_param):

    if isinstance(train_dataset, pandas.DataFrame):
        dataset = train_dataset.__copy__()
        samples = len(train_dataset.index)
        negatives = train_dataset[train_dataset["Annots"] == 0].count()["Annots"]
        positives = samples - negatives
    else:
        # read data
        dataset, negatives, positives = toolbox.read_data(train_dataset)

    # create a new name for a discretized data set
    new_file = str(train_dataset.replace(".csv", "")) + "_discretized_" + str(m_segments) \
               + "_" + str(alpha_param) + "_" + str(lambda_param)

    # list of sample annotation
    annots = dataset["Annots"].tolist()

    # get miRNA IDs
    miRNAs = dataset.columns.values.tolist()[2:]

    # list of discretization thresholds
    thresholds = []

    # global miRNA cdds
    global_cdds = []

    # drop all the miRNAs from the train data set, keep the sample IDs and annotation
    data_discretized = dataset.drop(miRNAs, axis=1)

    # count miRNAs with one/two states or another complicated pattern
    one_state_miRNAs = 0
    two_states_miRNAs = 0
    complicated_pattern_miRNAs = 0

    # iterate over miRNAs
    for miRNA in miRNAs:

        # set threshold to zero
        threshold = 0

        # get single miRNA gene expression levels
        miR_expr = dataset[miRNA].tolist()

        # discretize miRNA
        threshold, global_cdd = discretize_miRNA(miR_expr, annots, negatives, positives, m_segments, alpha_param, lambda_param)

        # count miRNA expression patterns
        if threshold == -1:
            complicated_pattern_miRNAs += 1
        if threshold == 0:
            one_state_miRNAs += 1
            threshold = -1
        if threshold > 0:
            two_states_miRNAs += 1

        # add threshold to a list of thresholds
        thresholds.append(threshold)
        global_cdds.append(global_cdd)

        # discretize miRNAs according to thresholds
        miR_discretized = [0 if i <= threshold else 1 for i in miR_expr]

        # add discretized miRNAs to the data
        data_discretized[miRNA] = miR_discretized

    # create a dictrionary of miRNAs and its cdds
    miRNA_cdds = dict(zip(miRNAs, global_cdds))

    print("ONE STATE miRNAs: ", one_state_miRNAs)
    print("TWO STATE miRNAs: ", two_states_miRNAs)
    print("COMPLICATED PATTERNS: ", complicated_pattern_miRNAs)

    if not isinstance(train_dataset, pandas.DataFrame):
        # write data to a file
        data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return data_discretized, miRNAs, thresholds, miRNA_cdds


# discretize test data
def discretize_test_data(test_dataset, thresholds):

    if isinstance(test_dataset, pandas.DataFrame):
        dataset = test_dataset.__copy__()
        samples = len(test_dataset.index)
        negatives = test_dataset[test_dataset["Annots"] == 0].count()["Annots"]
        positives = samples - negatives
    else:
        # read data
        dataset, negatives, positives = read_data(test_dataset)

    # create a new name for a discretized data set
    new_file = str(test_dataset.replace(".csv", "")) + "_discretized"

    # list of sample annotation
    annots = dataset["Annots"].tolist()

    # get miRNA IDs
    miRNAs = dataset.columns.values.tolist()[2:]

    # drop all the miRNAs from the train data set
    data_discretized = dataset.drop(miRNAs, axis=1)

    # zip miRNAs and thresholds into a dictionary
    miR_dict = dict(zip(miRNAs, thresholds))

    # iterate over miRNAs
    for miRNA in miRNAs:

        # get single miRNA gene expression levels
        miR_expr = dataset[miRNA].tolist()

        # discretize miRNAs according to thresholds
        miR_discretized = [0 if i <= miR_dict[miRNA] else 1 for i in miR_expr]

        # add discretized miRNAs to the data
        data_discretized[miRNA] = miR_discretized

    if not isinstance(test_dataset, pandas.DataFrame):
        # write data set to file
        data_discretized.to_csv(new_file+".csv", index=False, sep=";")

    return data_discretized


# discretize several data sets
def discretize_data_for_tests(train_list, test_list, m_segments, alpha_param, lambda_param):

    discretized_train_data = []
    discretized_test_data = []

    fold = 1

    # iterate over train and test data
    for train, test in zip(train_list, test_list):

        print("\nDISCRETIZATION: ", fold)
        fold += 1

        print("THRESHOLD INFORMATION")
        # discretize train data and return thresholds
        data_discretized, miRNAs, thresholds, miRNA_cdds = discretize_train_data(train, m_segments, alpha_param, lambda_param)

        discretized_train_data.append(data_discretized)

        #print("miRNAs")
        #print(miRNAs)

        #print("Thresholds")
        #print(thresholds)

        # discretize test data avvording to thresholds
        data_discretized = discretize_test_data(test, thresholds)

        discretized_test_data.append(data_discretized)

    return discretized_train_data, discretized_test_data, miRNA_cdds

