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

