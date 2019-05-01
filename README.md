# RAccoon

A genetic algorithm to designing distributed cell classifier circuits.

## Running 

Run the algorithm from a command line:

```
python run_GA.py -train train_data.csv -test test_data.csv -f True -iter 75 -pop 200 -size 5 -thres 0.5 -cp 1.0 -mp 0.3 -ts 0.1
```

Description of parameters:

***-train*** - training data set in the .csv format

***-test*** - testing data set in the .csv format

***-f*** - filtering the data set (non-relevant columns are removed)

***-iter*** - number of iterations

***-pop*** - population size

***-size*** - maximal size of a classifier

***-thres*** - classification threshold

***-cp*** - crossover probability

***-mp*** - mutation probability

***-ts*** - tournament size
