

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

#####DATA SPLIT FOR PAIRED EXPERIMENTS#####
#split the data intro train&test data sets
trainTestSplitPairedExp <- function(count.matrix, annotation, negative.samples, positive.samples, train.fraction) {
  
  #transform the data to right format
  #colnames(count.matrix) <- gsub(pattern = "sample", replacement = "", colnames(count.matrix))
  #annotation <- transformAnnotation(annotation) #transform annotation to 0/1
  count.matrix <- rbind(count.matrix, annotation) #temporarily attach annotation to count matrix
  
  #find negative and positive class ids
  negative.class.ids <- which(annotation %in% c(0)) #find samples annotated as 0
  positive.class.ids <- which(annotation %in% c(1))
  
  #choose randolmy train data samples (by id)
  negative.class.train.samples <- sample(negative.class.ids, train.fraction*negative.samples)
  positive.class.train.samples <- negative.class.train.samples + negative.samples
  
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
  train.data.set <- cbind(negative.class.train.data, positive.class.train.data)
  
  #test data
  test.data.set <- cbind(negative.class.test.data, positive.class.test.data)
  
  #keep annotation
  train.annots <- train.data.set["annotation",]
  test.annots <- test.data.set["annotation",]
  
  train.data.set <- train.data.set[!row.names(train.data.set)%in%c("annotation"),]
  test.data.set <- test.data.set[!row.names(test.data.set)%in%c("annotation"),]
  
  data.set.split <- list(train.data.set, train.annots, test.data.set, test.annots)
  names(data.set.split) <- c("train.data.set", "train.annots", "test.data.set", "test.annots")
  
  return(data.set.split)
  
}


#####DATA SPLIT#####
#split the data intro train&test data sets
trainTestSplit <- function(count.matrix, annotation, negative.samples, positive.samples, train.fraction) {
  
  #transform the data to right format
  #colnames(count.matrix) <- gsub(pattern = "sample", replacement = "", colnames(count.matrix))
  #annotation <- transformAnnotation(annotation) #transform annotation to 0/1
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


splitRealWorldData <- function(name, count.matrix, annotation, negative.samples, positive.samples, train.fraction) {
  
  #############DATA SET SPLIT#############
  
  # split the data set into train/test according to train.fraction
  original_ids <- colnames(count.matrix)
  new_ids <- seq(1,length(colnames(count.matrix)),1)
  sample.info <- data.frame(original_ids,new_ids,annotation)
  write.table(sample.info, paste(name, "_sample_info.csv"),  sep = ";", row.names = FALSE, quote=FALSE)
  colnames(count.matrix) <- seq(1,length(colnames(count.matrix)),1)
  data.set.split <- trainTestSplit(count.matrix, annotation, negative.samples, positive.samples, train.fraction)
  train.data.set <- data.set.split$train.data.set 
  train.annots <- data.set.split$train.annots
  test.data.set <- data.set.split$test.data.set
  test.annots <- data.set.split$test.annots
  
  #############WRITE DATA TO FILES#############
  
  #write train data set to file
  train.data.set.to.write <- transformData(train.annots, train.data.set)
  
  train.data.set.name = paste(name, paste("train", ".csv", sep="", collapse = NULL), 
                              sep = "_", collapse = NULL)
  
  write.table(train.data.set.to.write, train.data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)
  
  #write test data set to file
  test.data.set.to.write <- transformData(test.annots, test.data.set)
  
  test.data.set.name = paste(name, paste("test", ".csv", sep="", collapse = NULL), 
                             sep = "_", collapse = NULL)
  
  write.table(test.data.set.to.write, test.data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)
  
  return(train.data.set.to.write)
}

