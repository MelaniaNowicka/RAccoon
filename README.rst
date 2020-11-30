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

**--train** - training data set in the .csv format (obligatory)

**--test** - testing data set in the .csv format (default: None)

Training and test data should be formatted according to the description in the Data format section.

**--filter** - filtering non-relevant features (default: t, f to turn off)

Features that are non-relevant (columns filled with only 0s or 1s) are filtered out as such features are not
informative.

**--discretize** - discretize the data (default: t, f to turn off)

**--mbin** - discretization parameter: m segments (default: 50)

**-abin** - discretization parameter: alpha (default: 0.5)

**--lbin** - discretization parameter: lambda (default: 0.1)

Data discretization according to [Wang et al. (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0925231214008480).
To know more please look into the mentioned publication.

**-c** - maximal size of a classifier (maximal number of single rules in the classifier, default: 5)

**-a** - classification threshold (default: None)

If classification threshold is set to a certain value (e.g., 0.5) the threshold is fixed for all classifiers in
the GA run, if None - different values of thresholds (0.25, 0.45, 0.5, 0.75 and 1.0) are randomly assigned to
the classifiers.

**-w** - multi-objective function weight (default: 0.5)

**-u** - uniqueness option (related to calculation of CDD classifier score, default: True)

**-i** - number of iterations without improvement after which the algorithm terminates (default: 30)

**-f** - number of fixed iterations after which the algorithm terminates (default: None)

**-p** - population size (default: 300)

**--elitism** - if True the best found solutions are added to the population in each selection operation (default: True)

**--rules** - path to a file of pre-optimized rules (default: None)

**--poptfrac** - pre-optimized fraction of population, the rest of solutions is generated randomly (default: 0.5)

**-x** - crossover probability (default: 0.8)

**-m** - mutation probability (default: 0.1)

**-t** - tournament size (default: 0.2)


**Running complex testing scheme**

Run the analysis using::

$ python run_tests.py --train train_data.csv --test test_data.csv --config config.ini

Use exemplary data to try it: *train_data.csv, test_data.csv*.

You may change all the parameters in config.ini. Description may be found
`here <https://github.com/MelaniaNowicka/RAccoon/blob/master/config.ini>`_.

Output log description:

**READING CONFIG** - config parameter values

**READING DATA** - data information

**PARAMETER TUNING** - parameter tuning section including data division and pre-processing information as well
as parameter tuning results

**FINAL TEST** - results of the final tests (the classifiers are trained with tuned parameters and tested on test data)


