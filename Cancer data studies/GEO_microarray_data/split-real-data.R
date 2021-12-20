source("utility-functions.R")

# DATA SPLIT FOR PAIRED EXPERIMENTS ----------------------------------------------
# split the data intro train&test data sets
# Note, negative and positive samples must be in paired-order
# All negative samples must have ids neg_sample_id + number_of_negative_sample
# ID
# 1 - neg sample
# 2 - neg sample
# 3 - pos sample (pair for 1)
# 4 - pos sample (pair for 2)
#
trainTestSplitPairedExp <- function(count.matrix, annotation, negative.samples, positive.samples, train.fraction) {
  
  #transform the data to right format
  count.matrix <- rbind(count.matrix, annotation) #temporarily attach annotation to count matrix
  
  #find negative and positive class ids
  negative.class.ids <- which(annotation %in% c(0)) #find samples annotated as 0
  positive.class.ids <- which(annotation %in% c(1))
  
  #choose randolmy train data samples (by id)
  negative.class.train.samples <- sample(negative.class.ids, train.fraction*negative.samples)
  positive.class.train.samples <- negative.class.train.samples + negative.samples # find pair
  
  #add the rest of the samples to the test data
  negative.class.test.samples <- subset(negative.class.ids, !(negative.class.ids %in% negative.class.train.samples))
  positive.class.test.samples <- subset(positive.class.ids, !(positive.class.ids %in% positive.class.train.samples))
  
  #get column names
  negative.class.train.samples <- colnames(count.matrix)[negative.class.train.samples]
  positive.class.train.samples <- colnames(count.matrix)[positive.class.train.samples]
  negative.class.test.samples <- colnames(count.matrix)[negative.class.test.samples]
  positive.class.test.samples <- colnames(count.matrix)[positive.class.test.samples]
  
  #split train data
  negative.class.train.data <- count.matrix[,negative.class.train.samples]
  positive.class.train.data <- count.matrix[,positive.class.train.samples]
  
  #sort
  negative.class.train.data <- negative.class.train.data[,order(as.numeric(colnames(negative.class.train.data)))]
  positive.class.train.data <- positive.class.train.data[,order(as.numeric(colnames(positive.class.train.data)))]
  
  #split test data
  negative.class.test.data <- count.matrix[,negative.class.test.samples]
  positive.class.test.data <- count.matrix[,positive.class.test.samples]
  
  #sort
  negative.class.test.data <- negative.class.test.data[,order(as.numeric(colnames(negative.class.test.data)))]
  positive.class.test.data <- positive.class.test.data[,order(as.numeric(colnames(positive.class.test.data)))]
  
  #join data
  #train data
  names <- row.names(train.data.set)
  train.data.set <- cbind(negative.class.train.data, positive.class.train.data, row.names=NULL)
  row.names(train.data.set) <- names
  
  #test data
  names <- row.names(test.data.set)
  test.data.set <- cbind(negative.class.test.data, positive.class.test.data, row.names=NULL)
  row.names(test.data.set) <- names
  
  #keep annotation
  train.annots <- train.data.set["annotation",]
  test.annots <- test.data.set["annotation",]
  
  train.data.set <- train.data.set[!row.names(train.data.set)%in%c("annotation"),]
  test.data.set <- test.data.set[!row.names(test.data.set)%in%c("annotation"),]
  
  data.set.split <- list(train.data.set, train.annots, test.data.set, test.annots)
  names(data.set.split) <- c("train.data.set", "train.annots", "test.data.set", "test.annots")
  
  return(data.set.split)
  
}


# DATA SPLIT ----------------------------------------------
# split the data intro train & test data sets
trainTestSplit <- function(count.matrix, annotation, negative.samples, positive.samples, train.fraction) {
  
  #transform the data to right format
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
  names <- row.names(train.data.set)
  train.data.set <- cbind(negative.class.train.data, positive.class.train.data, row.names=NULL)
  row.names(test.data.set) <- names
  train.data.set <- train.data.set[,order(as.numeric(colnames(train.data.set)))]
  
  #test data
  names <- row.names(test.data.set)
  test.data.set <- cbind(negative.class.test.data, positive.class.test.data, row.names=NULL)
  row.names(test.data.set) <- names
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

# SPLIT DATA BY SAMPLE ID ----------------------------------------------
splitDataBySamples <- function(count.matrix, annotation, train.samples, test.samples) {
  
  count.matrix <- rbind(count.matrix, annotation)
  
  train.data.set <- count.matrix[,train.samples]
  test.data.set <- count.matrix[,test.samples]
  
  train.annots <- as.vector(train.data.set[nrow(count.matrix),])
  test.annots <- as.vector(test.data.set[nrow(count.matrix),])
  
  train.data.set <- train.data.set[-nrow(train.data.set),]
  test.data.set <- test.data.set[-nrow(test.data.set),]
  
  data.set.split <- list(train.data.set, train.annots, test.data.set, test.annots)
  names(data.set.split) <- c("train.data.set", "train.annots", "test.data.set", "test.annots")
  
  return(data.set.split)
  
}


splitRealWorldData <- function(name, count.matrix, annotation, negative.samples, positive.samples, train.fraction, by.sample, train.samples, test.samples, save.to.file=TRUE) {
  
  #############DATA SET SPLIT#############
  
  # split the data set into train/test according to train.fraction
  original_ids <- colnames(count.matrix) # get original sample ids
  new_ids <- seq(1,length(colnames(count.matrix)),1) # assign new ids (1, 2, 3, etc.)
  sample.info <- data.frame(original_ids, new_ids, annotation) # create data frame with original ids, new ids and annotation
  write.table(sample.info, paste(name, "_sample_info.csv", sep=""),  sep = ";", row.names = FALSE, quote=FALSE)
  colnames(count.matrix) <- new_ids # assign new ids
  
  if(by.sample == FALSE) {
    # split data randomly
    data.set.split <- trainTestSplit(count.matrix, annotation, negative.samples, positive.samples, train.fraction)
  }
  else {
    # split data by train.samples and test.samples
    data.set.split <- splitDataBySamples(count.matrix, annotation, train.samples, test.samples)
  }
  
  if (save.to.file==TRUE) {
    train.data.set <- data.set.split$train.data.set 
    train.annots <- data.set.split$train.annots
    test.data.set <- data.set.split$test.data.set
    test.annots <- data.set.split$test.annots
    
    #############WRITE DATA TO FILES#############
    #write train data set to file
    
    train.data.set.name = paste(name, paste("train", ".csv", sep="", collapse = NULL), 
                                sep = "_", collapse = NULL)
    
    save_matrix(train.data.set.name, train.annots, train.data.set)
    
    #write test data set to file
    
    test.data.set.name = paste(name, paste("test", ".csv", sep="", collapse = NULL), 
                               sep = "_", collapse = NULL)
    
    save_matrix(test.data.set.name, test.annots, test.data.set)
  }
  return(data.set.split)
}

