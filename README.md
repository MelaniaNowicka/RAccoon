# RAccoon

A genetic algorithm to designing distributed cell classifier circuits.

## Running complex analysis

Run the analysis from a command line using:

```
python run_tests.py --train train_data.csv --test test_data.csv --config config.ini
```

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
