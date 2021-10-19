from urllib import request
import pandas as pd
import os

path = os.path.normpath("C:/Users/melan/OneDrive/Dokumenty/PhD Thesis/Chapter5/")  # path to files

# get GSM ids and ids to create urls
url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?view=data&acc=&id=&db=GeoDb_blob46"  # general url
url_ids = pd.read_csv(os.path.join(path, "GSE22058_url_ids.csv"), sep=";")
acc_list = list(url_ids.GSM_id)
ids_list = list(url_ids.id)

# create urls, write them to files and download single sample files
with open(os.path.join(path, "GSE22058_urls.txt"), 'a+') as output:
    for i in range(0, len(acc_list)):
        print("Downloading sample: ", i)
        temp_url = url.replace("acc=", "acc="+acc_list[i])
        temp_url = temp_url.replace("id=", "id="+str(ids_list[i]))
        output.write(temp_url+"\n")
        request.urlretrieve(temp_url, os.path.join(path, "samples/html/GSE22058_"+acc_list[i]))  # download sample file

# process sample files
header = ["ID_REF", "RAW_VALUE", "VALUE", "PVALUE"]  # create data header
raw_values = pd.DataFrame.from_dict(data={})  # create empty data frame for raw values
normalized_values = pd.DataFrame.from_dict(data={})  # create empty data frame for normalized values
web_mir_ids = []  # list of miRNA ids from web page
i = 0  # sample counter

# read and pre-process samples
for filename in os.listdir(os.path.join(path, "samples/html/")):  # iterate over files

    print("Processing file: ", filename)

    # strip and separate file lines
    with open(os.path.join(path, "samples/html/")+filename) as f:
        lines = [line.rstrip() for line in f]  # strip lines
    sample = lines[24:244]  # get sample-related lines (rest is html code)

    sample_split = [line.split("\t") for line in sample]  # split sample data by tab
    sample_split_df = pd.DataFrame(sample_split, columns=header)  # and create data frame

    if i == 0:  # if this is the first sample get the miRNA ids as template
        web_mir_ids = [row[0] for row in sample_split]
        raw_values["miR_IDS"] = web_mir_ids
        print("IDs added!")
    else:  # if not just get the miRNA ids from the new file
        temp_ids = [row[0] for row in sample_split]
        if temp_ids != web_mir_ids:  # and compare with the template
            print("IDs do not match for sample:" + str(i) + "!")

    # create raw and normalized value data sets
    temp_raw_values = [row[1] for row in sample_split]  # get the raw values
    temp_normalized_values = [row[2] for row in sample_split]  # get the normalized values
    raw_values[acc_list[i]] = temp_raw_values  # assign the raw values as a column with the sample accession number
    normalized_values[acc_list[i]] = temp_normalized_values
    i += 1

    # save all sample files separately
    filename_temp = filename + ".csv"
    sample_split_df.to_csv(path_or_buf=os.path.join(path, "samples/", filename_temp), sep=';', index=False)

# read sample order and annotation
sample_order_and_annot = pd.read_csv(os.path.join(path, "samples/")+"GSE22058_sample_info.csv", sep=";")

# create dictionaries translating from gsm id to int id and gsm id to annotation
gsm_to_id = dict(zip(sample_order_and_annot.original_ids, sample_order_and_annot.new_ids))
gsm_to_annot = dict(zip(sample_order_and_annot.original_ids, sample_order_and_annot.annotation))

# compare miRNA IDs from GPL platform and sample web page
# web_mir_ids = [int(i) for i in web_mir_ids]
# if list(mirna_ids.ID) == web_mir_ids:
#     print("miRNA ids are complete and identical.")
# else:
#    print("miRNA ids are incomplete, in wrong order or not identical!")
#    print(set(list(mirna_ids.ID)) - set(web_mir_ids))

# compare platform IDs from series matrix (sm) with the sample web page
mirna_ids_from_sm = pd.read_csv(os.path.join(path, "samples/")+"mir_ids_from_series_matrix.csv", sep=";")
web_mir_ids = [int(i) for i in web_mir_ids]  # convert from string to int
if list(mirna_ids_from_sm.platform_ID) == web_mir_ids:
    print("miRNA ids are complete and identical.")
else:  # if not complete or not not identical show which ones differ
    print("miRNA ids are incomplete, in wrong order or not identical!")
    print(set(list(mirna_ids_from_sm.platform_ID)) - set(web_mir_ids))

# create raw values dataset with the order "ID", "Annots", "mir1", "mir2", etc.
header = ["ID", "Annots"] + list(mirna_ids_from_sm.miR_ID)
raw_dataset = pd.DataFrame(columns=header)
list_of_sample_dicts = []
for sample in list(raw_values.columns[1:]):  # iterate over samples
    sample_dict = dict(zip(header, [gsm_to_id[sample], gsm_to_annot[sample]]+list(raw_values[sample])))
    list_of_sample_dicts.append(sample_dict)

# create data set and sort by ID
raw_dataset = pd.DataFrame(list_of_sample_dicts)
raw_dataset = raw_dataset.sort_values("ID", axis=0)

# create normalized values dataset with the order "ID", "Annots", "mir1", "mir2", etc.
normalized_dataset_web = pd.DataFrame(columns=header)  # create empty data frame
list_of_sample_dicts = []
for sample in list(normalized_values.columns):  # iterate over samples
    values = [str(i).rstrip('0') for i in list(normalized_values[sample])]
    sample_dict = dict(zip(header, [gsm_to_id[sample], gsm_to_annot[sample]]+values))
    list_of_sample_dicts.append(sample_dict)

# create data set and sort by ID
normalized_dataset_web = pd.DataFrame(list_of_sample_dicts)
normalized_dataset_web = normalized_dataset_web.sort_values("ID", axis=0)

# read platform miRNA ids and miRNA ids
# mirna_ids = pd.read_csv(os.path.join(path, "samples/")+"GSE22058_mirna_ids.csv", sep="\t")
# id_to_mirna_id = dict(zip(mirna_ids.ID, mirna_ids.miRNA_ID))

# filter * and non-human miRNAs from the header
header = [i for i in header if "*" not in i]
header = [i for i in header if "hsa" in i]
header = ["ID", "Annots"] + header

# filter data sets using the filtered header
raw_dataset_filtered = raw_dataset[header]
normalized_dataset_web_filtered = normalized_dataset_web[header]

# save raw value table
raw_dataset_filtered.to_csv(path_or_buf=os.path.join(path, "samples/")+"GSE22058_non_norm_formatted.csv", sep=';',
                            index=False)
normalized_dataset_web_filtered.to_csv(path_or_buf=os.path.join(path, "samples/")+"GSE22058_norm_formatted.csv",
                                       sep=';', index=False)

