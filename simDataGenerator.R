# Data generation, split intro train/test data sets and normalization 
# script written by M. Nowicka, Free University Berlin, Berlin (2019).
# The script allows to generate a gene expression data set containing
# raw RNA-seq counts using compcodeR package by Soneson C (2014) with
# different parameter sets, split the data set into train/test fractions
# and normalize the data sets with TMM normalization (edgeR package). 
# The test data set is normalized using a reference sample found for 
# the train data set to prevent information leakage.

# Libraries necessary ti run the script: compcodeR, edgeR, matrixStats

#load packages
library("compcodeR")
library("edgeR")
library("matrixStats")


#############DATA SET GENERATION#############
generateSimData <- function(n.genes, samples.per.cond, n.diffexp, fraction.upregulated, random.outlier.high.prob, random.outlier.low.prob, generateSummary) {
  
  #generate file name
  data.set.file.name = paste("sim_data", n.genes, samples.per.cond, n.diffexp, 
                             fraction.upregulated, random.outlier.high.prob,
                             random.outlier.low.prob, sep = "_", collapse = NULL)
  
  #generate synthetic data with compcodeR
  data.set <- generateSyntheticData(dataset = "mydat", 
                                    n.vars = n.genes, 
                                    samples.per.cond = samples.per.cond, 
                                    n.diffexp = n.diffexp, 
                                    fraction.upregulated = fraction.upregulated, 
                                    random.outlier.high.prob = random.outlier.high.prob, 
                                    random.outlier.low.prob = random.outlier.low.prob, 
                                    repl.id = 1, 
                                    output.file = paste(data.set.file.name, ".rds", sep=""))
  
  #transform data
  annotation <- transformAnnotation(data.set@sample.annotations$condition)
  data.set.to.write <- transformData(data.set.annotation = annotation, 
                                     data.set.counts =  data.set@count.matrix)
  
  #write to file
  data.set.name = paste("data", n.genes, samples.per.cond, n.diffexp, 
                        fraction.upregulated, random.outlier.high.prob,
                        random.outlier.low.prob, sep = "_", collapse = NULL)
  
  write.table(data.set.to.write, paste(data.set.name, ".csv", sep=""), sep=";", row.names = FALSE)
  
  #generate data set summary
  if(generateSummary == TRUE){
    summarizeSyntheticDataSet(data.set = data.set, output.filename = paste(data.set.name,".html"))
  }
  
  return(data.set)
}

#####DATA SPLIT#####
#split the data intro train&test data sets
trainTestSplit <- function(count.matrix, annotation, negative.samples, positive.samples, train.fraction) {
  
  #transform the data to right format
  colnames(count.matrix) <- gsub(pattern = "sample", replacement = "", colnames(count.matrix))
  annotation <- transformAnnotation(annotation) #transform annotation to 0/1
  count.matrix <- rbind(count.matrix, annotation) #temporarily attach annotation to count matrix
  
  #find negative and positive class ids
  negative.class.ids <- which(annotation %in% c(0)) #find samples annotated as 0
  negative.class.ids <- colnames(count.matrix)[negative.class.ids] #assign samples ids as negative class
  positive.class.ids <- which(annotation %in% c(1)) #find samples annotated as 1
  positive.class.ids <- colnames(count.matrix)[positive.class.ids] #assign samples ids as positive class
  
  #choose randolmy train data samples (by id)
  negative.class.train.samples <- sample(negative.class.ids, train.fraction*negative.samples)
  positive.class.train.samples <- sample(positive.class.ids, train.fraction*positive.samples)
  
  #add the rest of the samples to the test data
  negative.class.test.samples <- subset(negative.class.ids, !(negative.class.ids %in% negative.class.train.samples))
  positive.class.test.samples <- subset(positive.class.ids, !(positive.class.ids %in% positive.class.train.samples))
  
  #split train data
  negative.class.train.data <- count.matrix[,negative.class.train.samples]
  positive.class.train.data <- count.matrix[,positive.class.train.samples]
  
  #split test data
  negative.class.test.data <- count.matrix[,negative.class.test.samples]
  positive.class.test.data <- count.matrix[,positive.class.test.samples]

  #join data
  #train data
  train.data.set <- cbind(negative.class.train.data, positive.class.train.data)
  train.data.set <- train.data.set[,order(as.numeric(colnames(train.data.set)))]
  #test data
  test.data.set <- cbind(negative.class.test.data, positive.class.test.data)
  test.data.set <- test.data.set[,order(as.numeric(colnames(test.data.set)))]
  
  #keep annotation
  train.annots <- train.data.set["annotation",]
  test.annots <- test.data.set["annotation",]
  
  train.data.set <- train.data.set[!row.names(train.data.set)%in%c("annotation"),]
  test.data.set <- test.data.set[!row.names(test.data.set)%in%c("annotation"),]
  
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
  print("BEFORE NORMALIZATION")
  commDisp <- estimateCommonDisp(data.DGEList)
  commDisp.test <- exactTest(commDisp)
  print(table(p.adjust(commDisp.test$table$PValue, method="BH")<0.05))
  
  data.DGEList <- DGEList(counts = normalized.counts, group = annots)
  
  print("AFTER NORMALIZATION")
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
transformData <- function(data.set.annotation, data.set.counts) {
  
  ID <- colnames(data.set.counts) #generate IDs for samples
  transformed.data.set <- t(data.set.counts) #transform the count matrix
  Annots <- data.set.annotation #add annotation of samples
  transformed.data.set <- cbind(Annots, transformed.data.set) #bind data with annotation
  transformed.data.set <- cbind(ID, transformed.data.set) #bind data with IDs
  
  return(transformed.data.set)
}

transformAnnotation <- function(annotation){
  
  annotation[annotation==1] <- 0
  annotation[annotation==2] <- 1
  
  return(annotation)
}


#####DATA PREPARATION#####
#generate and prepare the data
#n.genes - number of genes
#samples.per.cond - number of samples in each class
#n.diffexp - number of differentially expressed genes
#fraction.upregulated - fraction of upregulated samples
#random.outlier.high.prob - probability of random outliers (high)
#random.outlier.low.prob - probability of random outliers (low)
#train.fraction - fraction of samples in the training data
#is.seed - set seed to TRUE for reproducibility 
#generateSummary - if TRUE generates a summary for the simulated dataset
#imbalanced - if TRUE it processes an imbalanced data set 
prepareSimulatedDataset <- function(n.genes, 
                        samples.per.cond, 
                        n.diffexp, 
                        fraction.upregulated, 
                        random.outlier.high.prob, 
                        random.outlier.low.prob, 
                        train.fraction,
                        is.seed, 
                        generateSummary,
                        imbalanced) {

  start_time <- Sys.time()
  
  # set seed for reproducibility
  if (is.seed == TRUE) {
    set.seed(1)
  }
  
  #############DATA SET GENERATION#############
  
  data.set <- generateSimData(n.genes, 
                  samples.per.cond, 
                  n.diffexp, 
                  fraction.upregulated, 
                  random.outlier.high.prob, 
                  random.outlier.low.prob,
                  generateSummary)
  
  #############IMBALANCE DATA PROCESSING#############
  
  if (imbalanced == FALSE) {
    negative.samples = samples.per.cond
    positive.samples = samples.per.cond
  }
  else {}
  
  #...
  #...
  #...
  
  
  #############DATA SET SPLIT#############
  
  # split the data set into train/test according to train.fraction
  count.matrix <- data.set@count.matrix
  annotation <- data.set@sample.annotations$condition
  data.set.split <- trainTestSplit(count.matrix, annotation, negative.samples, positive.samples, train.fraction)
  train.data.set <- data.set.split$train.data.set 
  train.annots <- data.set.split$train.annots
  test.data.set <- data.set.split$test.data.set
  test.annots <- data.set.split$test.annots
  
  #############FIND REFERENCE SAMPLE#############
  
  #generate reference sample
  refSample <- returnRefSample(train.data.set)

  #############WRITE DATA TO FILES#############
  
  #write train data set to file
  train.data.set.to.write <- transformData(train.annots, train.data.set)
  
  train.data.set.name = paste("train", n.genes, samples.per.cond, n.diffexp, 
                              fraction.upregulated, train.fraction, random.outlier.high.prob,
                              paste(random.outlier.low.prob, ".csv", sep="", collapse = NULL), 
                              sep = "_", collapse = NULL)
  
  write.table(train.data.set.to.write, train.data.set.name, sep=";", row.names = FALSE)
  
  #write test data set to file
  test.data.set.to.write <- transformData(test.annots, test.data.set)
  
  test.data.set.name = paste("test", n.genes, samples.per.cond, n.diffexp, 
                             fraction.upregulated, train.fraction, random.outlier.high.prob,
                             paste(random.outlier.low.prob, ".csv", sep="", collapse = NULL), 
                             sep = "_", collapse = NULL)
  
  write.table(test.data.set.to.write, test.data.set.name, sep=";", row.names = FALSE)
  
  #############TRAIN DATA NORMALIZATION#############
  print("TRAIN DATA")
  train.normalized.counts <- normalizeTMM(train.data.set, train.annots, refSample = -1)
  
  #write normalized train data to file
  train.data.set <- transformData(train.annots, train.normalized.counts)
  
  train.data.set.name = paste("train", n.genes, samples.per.cond, n.diffexp, 
                              fraction.upregulated, train.fraction, 
                              random.outlier.high.prob, random.outlier.low.prob,
                              paste("TMMnorm", ".csv", sep="", collapse = NULL), 
                              sep = "_", collapse = NULL)
  
  write.table(train.data.set, train.data.set.name, sep=";", row.names = FALSE)
  
  #############TEST DATA NORMALIZATION#############
  print("TEST DATA")
  test.normalized.counts <- normalizeTMM(test.data.set, test.annots, refSample)
  
  #write test data set to file
  #ID <- seq(1, ncol(test.normalized.counts), by=1)
  test.data.set <- transformData(test.annots, test.normalized.counts)
  
  test.data.set.name = paste("test", n.genes, samples.per.cond, n.diffexp, 
                             fraction.upregulated, train.fraction, 
                             random.outlier.high.prob, random.outlier.low.prob,
                             paste("TMMnorm", ".csv", sep="", collapse = NULL), 
                             sep = "_", collapse = NULL)
  
  write.table(test.data.set, test.data.set.name, sep=";", row.names = FALSE)
  
  end_time <- Sys.time()
  
  print(paste("RUN TIME: ", end_time - start_time))
  
  dataTMM <- list(train.normalized.counts, test.normalized.counts)
  names(dataTMM) <- c("train.data.set", "test.data.set")
  
  return(dataTMM)
}