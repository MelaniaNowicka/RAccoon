###########################################
Documentation of RAccoon
###########################################

A genetic algorithm to designing distributed cell classifier circuits.

Requirements
============
RAccoon has the following dependencies:

- Python 3+

Following packages needs to be installed:

- pandas
- scikit-learn
- numpy

Installation
============

To download the RAccoon from Github, do::

    $ git clone https://github.com/MelaniaNowicka/RAccoon


Data format
===========

Use the following format of the .csv file: the first column includes unique IDs of samples, the second column includes
annotation (0 - negative samples, 1 - positive samples), the following columns include miRNA profiles. Use semicolon as
a separator. An example may be found below:

Continuous data:

+-------+----------+--------+---------+
|  ID   |  Annots  |  miR1  |  miR2   |
+=======+==========+========+=========+
| 1     | 0        | 345    | 12      |
+-------+----------+--------+---------+
| 2     | 0        | 1232   | 234     |
+-------+----------+--------+---------+
| 3     | 1        | 2      | 23      |
+-------+----------+--------+---------+

If you use continuous data, keep the discretization on.

Discretized data:

+-------+----------+--------+---------+
|  ID   |  Annots  |  miR1  |  miR2   |
+=======+==========+========+=========+
| 1     | 0        | 1      | 0       |
+-------+----------+--------+---------+
| 2     | 0        | 1      | 0       |
+-------+----------+--------+---------+
| 3     | 1        | 0      | 1       |
+-------+----------+--------+---------+

Training and testing classifiers
================================

To train a classifier run::


$ python run_GA.py --train train_data.csv


Use exemplary data to try it: *train_data.csv, test_data.csv*.

Description of parameters:

**- -train** - training data set in the .csv format

**- -test** - testing data set in the .csv format (default: None)

**- -filter** - filtering non-relevant features (default: t, f to turn off)

**- -discretize** - discretize the data (default: t, f to turn off)

**- -mbin** - discretization parameter: m segments (default: 50)

**- -abin** - discretization parameter: alpha (default: 0.5)

**- -lbin** - discretization parameter: lambda (default: 0.1)

**-c** - maximal size of a classifier (default: 5)

**-a** - classification threshold (default: 0.45)

**-w** - multi-objective function weight (default: 0.5)

**-i** - number of iterations without improvement after which the algorithm is terminated (default: 30)

**-p** - population size (default: 300)

**-x** - crossover probability (default: 0.8)

**-m** - mutation probability (default: 0.1)

**-t** - tournament size (default: 0.2)


**Running complex testing scheme**

Run the analysis using::

$ python run_tests.py --train train_data.csv --test test_data.csv --config config.ini

Use exemplary data to try it: *train_data.csv, test_data.csv*.

You may change all the parameters in config.ini.

Output log description:

**READING CONFIG** - config parameter values

**READING DATA** - data information

**PARAMETER TUNING** - parameter tuning section including data division and pre-processing information as well
as parameter tuning results

**FINAL TEST** - results of the final tests (the classifiers are trained with tuned parameters and tested on test data)


