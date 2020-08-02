# RAccoon

A genetic algorithm to designing distributed cell classifier circuits. 

## Data format

Use the following format of the .csv file: the first column includes unique IDs of samples, the second column includes 
annotation (0 - negative samples, 1 - positive samples), the following columns include miRNA profiles. Use semicolon as 
a separator. An example may be found below:

Continuous data:

| ID | Annots | miR-1 |
| -- | ------ | ----- |
| 1 | 0 | 34 |
| 2 | 1 | 475 |

If you use continuous data, keep the discretization on.

Discretized data:

| ID | Annots | miR-1 |
| -- | ------ | ----- |
| 1 | 0 | 0 |
| 2 | 1 | 1 |


## Training and testing classifiers

To train a classifier run: 

```
python run_GA.py --train train_data.csv 
```

Use exemplary data to try it: train_data.csv, test_data.csv.

Description of parameters:

***--train*** - training data set in the .csv format

***--test*** - testing data set in the .csv format (default: None)

***--filter*** - filtering non-relevant features (default: t, f to turn off)

***--discretize*** - discretize the data (default: t, f to turn off)

***--mbin*** - discretization parameter: m segments (default: 50)

***--abin*** - discretization parameter: alpha (default: 0.5)

***--lbin*** - discretization parameter: lambda (default: 0.1)

***-c*** - maximal size of a classifier (default: 5)

***-a*** - classification threshold (default: 0.45)

***-w*** - multi-objective function weight (default: 0.5)

***-i*** - number of iterations without improvement after which the algorithm is terminated (default: 30)

***-p*** - population size (default: 300)

***-x*** - crossover probability (default: 0.8)

***-m*** - mutation probability (default: 0.1)

***-t*** - tournament size (default: 0.2)


## Running complex testing scheme

Run the analysis using:

```
python run_tests.py --train train_data.csv --test test_data.csv --config config.ini
```
Use exemplary data to try it: train_data.csv, test_data.csv.

You may change all the parameters in config.ini.

## simDataGenerator

simDataGenerator allows to generate a simulated GED data set with compcodeR package, preprocess it by splitting into train/test data sets and normalize with TMM normalization method (edgeR). 

***LIBRARIES REQUIRED: copcodeR, edgeR, matrixStats***

Run *prepareSimulatedDataset()* with parameters:

***n.genes*** - number of genes, e.g., 500

***samples.per.cond*** - number of samples per class, e.g., 100

***n.diffexp*** - number of differentially expressed genes, e.g., 50 (10%)

***fraction.upregulated*** - fraction of differentially expresed genes that are upregulated, e.g., 0.5

***random.outlier.high.prob*** - number of random outliers (higher values), e.g., 0.5

***random.outlier.low.prob*** - number of random outliers (lower values), e.g., 0.5

***train.fraction*** - fraction of data that becomes training data set, e.g, 0.8 (1-train.fraction = test.fraction)

***is.seed*** - set to TRUE to be able to reproduce the results for certain conditions, set to FALSE if you want to generate different data sets

***generateSummary*** - set to TRUE if you want to generate a compCodeR data report

***imbalanced*** - set to TRUE if the data set should be imbalanced
