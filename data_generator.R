# Data generation, split intro train/test data sets and normalization 
# script written by M. Nowicka, Free University Berlin, Berlin (2019).
# The script allows to generate a gene expression data set containing
# raw RNA-seq counts using compcodeR package by Soneson C (2014) with
# different parameter sets, split the data set into train/test fractions
# and normalize the data sets with TMM normalization (edgeR package). 
# The test data set is normalized using a reference sample found for 
# the train data set to prevent information leakage.

# Libraries necessary ti run the script: compcodeR, edgeR, matrixStats

#package check
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

if (!require("compcodeR")) {
  BiocManager::install("compcodeR")
}
if (!require("edgeR")) {
  BiocManager::install("edgeR")
}
if (!require("matrixStats")) {
  BiocManager::install("matrixStats")
}

#load packages
library("compcodeR")
library("edgeR")
library("matrixStats")

#####DATA SPLIT#####
#split the data intro train&test data sets
trainTestSplit <- function(data.set, samples.per.cond, train.fraction) {
  
  #split intro train/test
  count.matrix <- data.set@count.matrix
  annotation <- data.set@sample.annotations$condition
  
  #divide data by class
  first.class.data <- count.matrix[,1:samples.per.cond]
  begin = samples.per.cond+1
  end = samples.per.cond*2
  second.class.data <- count.matrix[,begin:end]
  
  #generate ids
  first.class.ids <- c(1:samples.per.cond)
  second.class.ids <- c(begin:end)
  
  #choose randolmy train data samples (by id)
  first.class.train.samples <- sort(sample(first.class.ids, train.fraction*samples.per.cond))
  second.class.train.samples <- sort(sample(second.class.ids, train.fraction*samples.per.cond))
  
  #add the rest of the samples to the test data
  first.class.test.samples <- subset(first.class.ids, !(first.class.ids %in% first.class.train.samples))
  second.class.test.samples <- subset(second.class.ids, !(second.class.ids %in% second.class.train.samples))
  
  #add "sample" to ids (1 -> sample1))
  first.class.train.samples <- paste("sample", first.class.train.samples, sep="")
  second.class.train.samples <- paste("sample", second.class.train.samples, sep="")
  first.class.test.samples <- paste("sample", first.class.test.samples, sep="")
  second.class.test.samples <- paste("sample", second.class.test.samples, sep="")
  
  #split train data
  first.class.train.data <- subset(first.class.data, select=first.class.train.samples)
  second.class.train.data <- subset(second.class.data, select=second.class.train.samples)
  
  #split test data
  first.class.test.data <- subset(first.class.data, select=first.class.test.samples)
  second.class.test.data <- subset(second.class.data, select=second.class.test.samples)
  
  #join data
  #train data
  train.data.set <- cbind(first.class.train.data,second.class.train.data)
  #test data
  test.data.set <- cbind(first.class.test.data,second.class.test.data)
  
  #keep annotation
  first.class.train.annots <- replicate(train.fraction*samples.per.cond, 0)
  second.class.train.annots <- replicate(train.fraction*samples.per.cond, 1)
  
  first.class.test.annots <- replicate(samples.per.cond-length(first.class.train.annots), 0)
  second.class.test.annots <- replicate(samples.per.cond-length(second.class.train.annots), 1)
  
  train.annots <- c(first.class.train.annots, second.class.train.annots)
  test.annots <- c(first.class.test.annots, second.class.test.annots)
  
  data.set.split <- list(train.data.set, train.annots, test.data.set, test.annots)
  names(data.set.split) <- c("train.data.set", "train.annots", "test.data.set", "test.annots")
  
  return(data.set.split)
  
}


#####REFERENCE SAMPLE#####
#find reference sample
returnRefSample <-  function(count.matrix) {
  
  #remove non-regulated genes from the matrix
  filtered.matrix <- count.matrix[rowSums(count.matrix)>0,]
  
  #scale the values for each sample by dividing each value by sum of counts in a given sample
  scaled.matrix <- t(t(filtered.matrix)/colSums(filtered.matrix))
  
  #calculate 75% quantiles for each sample
  quantiles <- colQuantiles(scaled.matrix, p=0.75)
  
  #calculate average quantile over the samples
  avg.quantiles <- mean(quantiles)
  
  #choose the sample with the 75% quantile value the closest to the average as a reference sample
  refSample <- which(abs(quantiles-avg.quantiles)==min(abs(quantiles-avg.quantiles)))
  
  print("REFERENCE SAMPLE")
  print(refSample)
  
  #return the reference sample
  return(refSample)
}


#####NORMALIZATION#####
normalizeTMM <- function(data.set, annots, refSample) {
  
  #create edgeR object
  data.DGEList <- DGEList(counts = data.set, group = annots)
  
  if(refSample == -1){
    #calculate TMM factors without reference sample
    TMM <- calcNormFactors(data.DGEList, method="TMM")
  }
  else{
    #calculate TMM factors with reference sample
    TMM <- calcNormFactors(data.DGEList, method="TMM", refColumn = refSample)
  }
  
  #calculate normalized counts
  normalized.counts <- cpm(TMM)
  
  #run test
  commDisp <- estimateCommonDisp(data.DGEList)
  commDisp.test <- exactTest(commDisp)
  print(table(p.adjust(commDisp.test$table$PValue, method="BH")<0.05))
  
  return(normalized.counts)
}

#####TRANSFORM DATA#####
#transform data set in a following format:
# ID | Annots | gene1 | ...
#  1 |      0 |   100 | ...
#  2 |      0 |   200 | ...
transformData <- function(ID, data.set.annotation, data.set.counts) {
  
  transformed.dataset <- t(data.set.counts) #transform the count matrix
  Annots <- data.set.annotation #add annotation of samples
  transformed.data.set <- cbind(Annots, transformed.dataset) #bind data with annotation
  transformed.data.set <- cbind(ID, transformed.data.set) #bind data with IDs
  
  return(transformed.data.set)
}

#####DATA PREPARATION#####
#generate and prepare the data
#n.genes - number of genes
#samples.per.cond - number of samples in each class
#n.diffexp - number of differentially expressed genes
#fraction.upregulated - fraction of upregulated samples
#train.fraction - fraction of samples in the training data
#random.outlier.high.prob - probability of random outliers (high)
#random.outlier.low.prob - probability of random outliers (low)
#is.seed - set seed to TRUE for reproducibility 
prepareData <- function(n.genes, samples.per.cond, n.diffexp, fraction.upregulated, 
                        train.fraction, random.outlier.high.prob, random.outlier.low.prob, is.seed) {

  
  # set seed for reproducibility
  if (is.seed == TRUE) {
    set.seed(1)
    start_time <- Sys.time()
  }
  
  
  #############DATA SET GENERATION#############
  #write to file
  data.set.file.name = paste("data", n.genes, samples.per.cond, n.diffexp, 
                        fraction.upregulated, train.fraction, random.outlier.high.prob,
                        paste(random.outlier.low.prob, ".rds", sep="", collapse = NULL), 
                        sep = "_", collapse = NULL)
  
  #generate synthetic data with compcodeR
  data.set <- generateSyntheticData(dataset = "mydat", 
                                    n.vars = n.genes, 
                                    samples.per.cond = samples.per.cond, 
                                    n.diffexp = n.diffexp, 
                                    fraction.upregulated = fraction.upregulated, 
                                    random.outlier.high.prob = random.outlier.high.prob, 
                                    random.outlier.low.prob = random.outlier.low.prob, 
                                    repl.id = 1, 
                                    output.file = data.set.file.name)
  
  
  ID <- seq(1, ncol(data.set@count.matrix), by=1) #generate IDs for samples
  data.set.to.write <- transformData(ID, data.set.annotation = data.set@sample.annotations$condition, 
                                     data.set.counts =  data.set@count.matrix)
  
  #write to file
  data.set.name = paste("data", n.genes, samples.per.cond, n.diffexp, 
                        fraction.upregulated, train.fraction, random.outlier.high.prob,
                        paste(random.outlier.low.prob, ".csv", sep="", collapse = NULL), 
                        sep = "_", collapse = NULL)
  
  write.table(data.set.to.write, data.set.name, sep=";", row.names = FALSE)
  
  #generate data set summary
  summarizeSyntheticDataSet(data.set = data.set, output.filename = paste(data.set.name,".html"))

  #############DATA SET SPLIT  #############
  
  # split the data set into train/test according to train.fraction
  data.set.split <- trainTestSplit(data.set, samples.per.cond, train.fraction)
  train.data.set <- data.set.split$train.data.set 
  train.annots <- data.set.split$train.annots
  test.data.set <- data.set.split$test.data.set
  test.annots <- data.set.split$test.annots
  
  #############FIND REFERENCE SAMPLE#############
  
  #generate reference sample
  refSample <- returnRefSample(train.data.set)

  #############WRITE DATA TO FILES#############
  
  #write train data set to file
  ID <- colnames(train.data.set) #get IDs for samples
  train.data.set.to.write <- transformData(ID, train.annots, train.data.set)
  
  train.data.set.name = paste("train", n.genes, samples.per.cond, n.diffexp, 
                              fraction.upregulated, train.fraction, random.outlier.high.prob,
                              paste(random.outlier.low.prob, ".csv", sep="", collapse = NULL), 
                              sep = "_", collapse = NULL)
  
  write.table(train.data.set.to.write, train.data.set.name, sep=";", row.names = FALSE)
  
  #generate data set summary
  summarizeSyntheticDataSet(data.set = train.data.set, output.filename = paste(train.data.set.to.write,".html"))
  
  #write test data set to file
  ID <- colnames(test.data.set)
  test.data.set.to.write <- transformData(ID, test.annots, test.data.set)
  
  test.data.set.name = paste("test", n.genes, samples.per.cond, n.diffexp, 
                             fraction.upregulated, train.fraction, random.outlier.high.prob,
                             paste(random.outlier.low.prob, ".csv", sep="", collapse = NULL), 
                             sep = "_", collapse = NULL)
  
  write.table(test.data.set.to.write, test.data.set.name, sep=";", row.names = FALSE)
  
  #generate data set summary
  summarizeSyntheticDataSet(data.set = test.data.set, output.filename = paste(test.data.set.to.write,".html"))
  
  #############TRAIN DATA NORMALIZATION#############
  print("TRAIN DATA")
  train.normalized.counts <- normalizeTMM(train.data.set, train.annots, refSample = -1)
  
  #write normalized train data to file
  ID <- colnames(train.normalized.counts)
  train.data.set <- transformData(ID, train.annots, train.normalized.counts)
  
  train.data.set.name = paste("train", n.genes, samples.per.cond, n.diffexp, 
                              fraction.upregulated, train.fraction, 
                              random.outlier.high.prob, random.outlier.low.prob,
                              paste("TMMnorm", ".csv", sep="", collapse = NULL), 
                              sep = "_", collapse = NULL)
  
  write.table(train.data.set, train.data.set.name, sep=";", row.names = FALSE)
  
  #generate data set summary
  summarizeSyntheticDataSet(data.set = train.normalized.counts, output.filename = paste(train.data.set.name,".html"))
  
  #############TEST DATA NORMALIZATION#############
  print("TEST DATA")
  test.normalized.counts <- normalizeTMM(test.data.set, test.annots, refSample)
  
  #write test data set to file
  #ID <- seq(1, ncol(test.normalized.counts), by=1)
  ID <- colnames(test.normalized.counts)
  test.data.set <- transformData(ID, test.annots, test.normalized.counts)
  
  test.data.set.name = paste("test", n.genes, samples.per.cond, n.diffexp, 
                             fraction.upregulated, train.fraction, 
                             random.outlier.high.prob, random.outlier.low.prob,
                             paste("TMMnorm", ".csv", sep="", collapse = NULL), 
                             sep = "_", collapse = NULL)
  
  write.table(test.data.set, test.data.set.name, sep=";", row.names = FALSE)
  
  #generate data set summary
  summarizeSyntheticDataSet(data.set = test.normalized.counts, output.filename = paste(test.data.set.name,".html"))
  
  end_time <- Sys.time()
  
  print(paste("RUN TIME: ", end_time - start_time))
}