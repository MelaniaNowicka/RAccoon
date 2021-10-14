from urllib import request
import pandas as pd
import csv
import os

path = "C:/Users/melan/OneDrive/Dokumenty/PhD Thesis/Chapter5/"
url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?view=data&acc=&id=&db=GeoDb_blob46"  # general url
acc_file = open(path+"GSE22058_GSEids.txt", "r")  # accession numbers
acc = acc_file.read()
acc_list = acc.split("\n")

ids_file = open(path+"GSE22058_ids.txt", "r")  # ids
ids = ids_file.read()
ids_list = ids.split("\n")

# download the files
with open(path+"GSE22058_urls.txt", 'a+') as output:
    for i in range(0, len(acc_list)):
        temp_url = url.replace("acc=", "acc="+acc_list[i])
        temp_url = temp_url.replace("id=", "id="+ids_list[i])
        output.write(temp_url+"\n")
        request.urlretrieve(temp_url, os.path.join(path, "samples/GSE22058_", acc_list[i]))

header = ["ID_REF", "RAW_VALUE", "VALUE", "PVALUE"]  # create data header
raw_values = pd.DataFrame.from_dict(data={})  # create empty data frame
mir_ids = []
i = 0

# read and pre-process samples
for filename in os.listdir(path+"samples/"):

    # strip and separate file lines
    with open(path+"samples/"+filename) as f:
        lines = [line.rstrip() for line in f]
    sample = lines[24:244]  # get sample-related lines

    sample_split = [line.split("\t") for line in sample]  # split by tab

    if i == 0:  # if this is the first sample get the miRNA ids as template
        mir_ids = [row[0] for row in sample_split]
        raw_values["miR_IDS"] = mir_ids
        print("IDs added!")
    else:  # if not just get the miRNA ids from the new file
        temp_ids = [row[0] for row in sample_split]
        if temp_ids != mir_ids:  # and compare with the template
            print("IDs do not match for sample:" + str(i) + "!")

    temp_raw_values = [row[1] for row in sample_split]  # get the raw values
    temp_normalized_values = [row[2] for row in sample_split]  # get the normalized values
    raw_values[acc_list[i]] = temp_raw_values  # assign the raw values as a column with the sample accession number
    i += 1

    filename_temp = filename + ".csv"

    # save all sample files separately
    with open(os.path.join(path, "samples/")+filename_temp, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sample_split)

# save raw value table
raw_values.to_csv(path_or_buf=os.path.join(path, "samples/")+"GSE22058_non_norm.csv", sep=',', index=False)