# RAccoon

A genetic algorithm to designing distributed cell classifier circuits.

## Running 

Run the algorithm from a command line:

```
python run_GA.py --train train_data.csv --test test_data.csv -f True -i 75 -p 200 -c 5 -a 0.5 -c 1.0 -m 0.3 -t 0.1
```

Description of parameters:

***--train*** - training data set in the .csv format

***--test*** - testing data set in the .csv format

***-f*** - filtering the data set (non-relevant columns are removed)

***-i*** - number of iterations

***-p*** - population size

***-c*** - maximal size of a classifier

***-a*** - classification threshold

***-c*** - crossover probability

***-m*** - mutation probability

***-t*** - tournament size
