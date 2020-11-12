#read the library
library("GEOquery")

set.seed(1)

#read the splitting script
source("splitRealWorldData.R")

#GSE22058/GPL10457

#get the data from series matrix
print("Reading data...")
data <- getGEO(filename="GSE22058/GPL10457/GPL10457_series_matrix.txt")

#get count.matrix
count.matrix <- exprs(object = data)

#set miRNA names to miRNA IDs
rownames(count.matrix) <- data@featureData@data$miRNA_ID
original.data.row.numb <- nrow(count.matrix)
print(paste("Number of features: ", original.data.row.numb))
  
#get only human miRNAs
count.matrix <- count.matrix[c(grep("hsa*", rownames(count.matrix))),]
non.human.data.row.numb <- nrow(count.matrix)
print(paste("Removed non-human features:", original.data.row.numb-nrow(count.matrix)))
print(paste("Number of features: ", non.human.data.row.numb))
#remove pre-miRNAs
count.matrix <- count.matrix[c(-grep("\\*", rownames(count.matrix))),]
print(paste("Removed * features:", non.human.data.row.numb-nrow(count.matrix)))
print(paste("Number of features: ", nrow(count.matrix)))

#get annotation
annotation <- data@phenoData@data$characteristics_ch1.1
annotation <-as.numeric(annotation)
annotation[annotation == 1] <- 0
annotation[annotation == 2] <- 1

#move negative samples first, positive samples next
count.matrix.neg <- count.matrix[,which(annotation %in% c(0))]
print(paste("Number of negative samples: ", ncol(count.matrix.neg)))
count.matrix.pos <- count.matrix[,which(annotation %in% c(1))]
print(paste("Number of positive samples: ", ncol(count.matrix.pos)))
count.matrix <- cbind(count.matrix.neg, count.matrix.pos)

#assign annotation
annotationneg <- rep(0,96)
annotationpos <- rep(1,96)
annotation <- c(annotationneg, annotationpos)

#transform data
original_ids <- colnames(count.matrix)
new_ids <- seq(1,length(colnames(count.matrix)),1)
sample.info <- data.frame(original_ids,new_ids,annotation)
write.table(sample.info, paste("GSE22058", "_sample_info.csv", sep=""),  sep = ";", row.names = FALSE, quote=FALSE)
colnames(count.matrix) <- seq(1,length(colnames(count.matrix)),1)

transformed.data.set <- transformData(annotation, count.matrix)

data.set.name = paste("GSE22058", ".csv", sep="", collapse = NULL)

write.table(transformed.data.set, data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)

#get sample IDs for division
train <- read.csv2("GSE22058_GPL10457_train.csv")
train.samples <- train$ID
test <- read.csv2("GSE22058_GPL10457_test.csv")
test.samples <- test$ID

#split data set
splitRealWorldData("GSE22058_", count.matrix, annotation, 96, 96, 0.8, TRUE, train.samples, test.samples)
#splitRealWorldData("GSE22058_GPL10457", count.matrix, annotation, 96, 96, 0.8)


#GSE10694

#get the data from series matrix
print("Reading data...")
data <- getGEO(filename="GSE10694/GSE10694_series_matrix.txt")

#get count.matrix
count.matrix <- exprs(object = data)

#set miRNA names to miRNA IDs
rownames(count.matrix) <- data@featureData@data$miRNA_ID
original.data.row.numb <- nrow(count.matrix)
print(paste("Number of features: ", original.data.row.numb))

#get only human miRNAs
count.matrix <- count.matrix[c(grep("hsa*", rownames(count.matrix))),]
non.human.data.row.numb <- nrow(count.matrix)
print(paste("Removed non-human features:", original.data.row.numb-nrow(count.matrix)))
print(paste("Number of features: ", non.human.data.row.numb))
#remove pre-miRNAs
count.matrix <- count.matrix[c(-grep("\\*", rownames(count.matrix))),]
print(paste("Removed * features:", non.human.data.row.numb-nrow(count.matrix)))
print(paste("Number of features: ", nrow(count.matrix)))

#move negative samples first, positive samples next
count.matrix.pos <- count.matrix[,1:78]
print(paste("Number of positive samples: ", ncol(count.matrix.pos)))
count.matrix.neg <- count.matrix[,79:156]
print(paste("Number of negative samples: ", ncol(count.matrix.neg)))
count.matrix <- cbind(count.matrix.neg, count.matrix.pos)

#assign annotation
annotationneg <- rep(0,78)
annotationpos <- rep(1,78)
annotation <- c(annotationneg, annotationpos)

#get sample IDs for division
train <- read.csv2("GSE10694_train.csv")
train.samples <- train$ID
test <- read.csv2("GSE10694_test.csv")
test.samples <- test$ID

#transform data
original_ids <- colnames(count.matrix)
new_ids <- seq(1,length(colnames(count.matrix)),1)
sample.info <- data.frame(original_ids,new_ids,annotation)
write.table(sample.info, paste("GSE10694", "_sample_info.csv", sep=""),  sep = ";", row.names = FALSE, quote=FALSE)
colnames(count.matrix) <- seq(1,length(colnames(count.matrix)),1)

transformed.data.set <- transformData(annotation, count.matrix)

data.set.name = paste("GSE10694", ".csv", sep="", collapse = NULL)

write.table(transformed.data.set, data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)

#split data set
splitRealWorldData("GSE10694_", count.matrix, annotation, 78, 78, 0.8, TRUE, train.samples, test.samples)
#splitRealWorldData("GSE10694_", count.matrix, annotation, 78, 78, 0.8)


#GSE36681

#get the data from series matrix
print("Reading data...")
data <- getGEO(filename="GSE36681/GSE36681_series_matrix.txt")

#get count.matrix
count.matrix <- exprs(object = data)

#set miRNA names to miRNA IDs
rownames(count.matrix) <- data@featureData@data$miRNA_ID
original.data.row.numb <- nrow(count.matrix)
print(paste("Number of features: ", original.data.row.numb))

#get only human miRNAs
count.matrix <- count.matrix[c(grep("hsa*", rownames(count.matrix))),]
non.human.data.row.numb <- nrow(count.matrix)
print(paste("Removed non-human features:", original.data.row.numb-nrow(count.matrix)))
#remove pre-miRNAs
count.matrix <- count.matrix[c(-grep("\\*", rownames(count.matrix))),]
print(paste("Removed * features:", non.human.data.row.numb-nrow(count.matrix)))
print(paste("Rows left: ", nrow(count.matrix)))

#assign annotation
annotation <- data@phenoData@data$source_name_ch1
annotation <-as.numeric(annotation)
annotation[annotation == 1] <- 0
annotation[annotation == 2] <- 1
annotation[annotation == 3] <- 0
annotation[annotation == 4] <- 1
annotation.FFPE <- annotation[1:94]
annotation.FF <- annotation[95:length(annotation)]

#move negative samples first, positive samples next
count.matrix.FFPE <- count.matrix[,1:94]
count.matrix.FFPE.neg <- count.matrix.FFPE[,which(annotation.FFPE %in% c(0))]
print(paste("Number of negative samples: ", ncol(count.matrix.FFPE.neg)))
count.matrix.FFPE.pos <- count.matrix.FFPE[,which(annotation.FFPE %in% c(1))]
print(paste("Number of positive samples: ", ncol(count.matrix.FFPE.pos)))
count.matrix.FFPE <- cbind(count.matrix.FFPE.neg, count.matrix.FFPE.pos)

#assign annotation
annotationneg <- rep(0,47)
annotationpos <- rep(1,47)
annotation <- c(annotationneg,annotationpos)

#transform data
original_ids <- colnames(count.matrix.FFPE)
new_ids <- seq(1,length(colnames(count.matrix.FFPE)),1)
sample.info <- data.frame(original_ids,new_ids,annotation)
write.table(sample.info, paste("GSE36681_FFPE", "_sample_info.csv", sep=""),  sep = ";", row.names = FALSE, quote=FALSE)
colnames(count.matrix.FFPE) <- seq(1,length(colnames(count.matrix.FFPE)),1)

transformed.data.set <- transformData(annotation, count.matrix.FFPE)

data.set.name = paste("GSE36681_FFPE", ".csv", sep="", collapse = NULL)

write.table(transformed.data.set, data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)

#move negative samples first, positive samples next
count.matrix.FF <- count.matrix[,95:length(colnames(count.matrix))]
count.matrix.FF.neg <- count.matrix.FF[,which(annotation.FF %in% c(0))]
print(paste("Number of negative samples: ", ncol(count.matrix.FF.neg)))
count.matrix.FF.pos <- count.matrix.FF[,which(annotation.FF %in% c(1))]
print(paste("Number of positive samples: ", ncol(count.matrix.FF.pos)))
count.matrix.FF <- cbind(count.matrix.FF.neg, count.matrix.FF.pos)

#assign annotation
annotationneg <- rep(0,56)
annotationpos <- rep(1,56)
annotation <- c(annotationneg,annotationpos)

#transform data
original_ids <- colnames(count.matrix.FF)
new_ids <- seq(1,length(colnames(count.matrix.FF)),1)
sample.info <- data.frame(original_ids,new_ids,annotation)
write.table(sample.info, paste("GSE36681_FF", "_sample_info.csv", sep=""),  sep = ";", row.names = FALSE, quote=FALSE)
colnames(count.matrix.FF) <- seq(1,length(colnames(count.matrix.FF)),1)

transformed.data.set <- transformData(annotation, count.matrix.FF)

data.set.name = paste("GSE36681_FF", ".csv", sep="", collapse = NULL)

write.table(transformed.data.set, data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)

#get sample IDs for division
train <- read.csv2("GSE36681_FFPE_train.csv")
train.samples <- train$ID
test <- read.csv2("GSE36681_FFPE_test.csv")
test.samples <- test$ID

splitRealWorldData("GSE36681_FFPE_", count.matrix.FFPE, annotation, 47, 47, 0.8, TRUE, train.samples, test.samples)
#splitRealWorldData("GSE36681_FFPE", count.matrix.FFPE, annotation, 47, 47, 0.8)

#get sample IDs for division
train <- read.csv2("GSE36681_FF_train.csv")
train.samples <- train$ID
test <- read.csv2("GSE36681_FF_test.csv")
test.samples <- test$ID

splitRealWorldData("GSE36681_FF_", count.matrix.FF, annotation, 56, 56, 0.8, TRUE, train.samples, test.samples)
#splitRealWorldData("GSE36681_FF", count.matrix.FF, annotation, 56, 56, 0.8)


#GSE36681

#get the data from series matrix
print("Reading data...")
data <- getGEO(filename="GSE36681/GSE36681_series_matrix.txt")

#get count.matrix
count.matrix <- exprs(object = data)

#set miRNA names to miRNA IDs
rownames(count.matrix) <- data@featureData@data$miRNA_ID
original.data.row.numb <- nrow(count.matrix)
print(paste("Number of features: ", original.data.row.numb))

#get only human miRNAs
count.matrix <- count.matrix[c(grep("hsa*", rownames(count.matrix))),]
non.human.data.row.numb <- nrow(count.matrix)
print(paste("Removed non-human features:", original.data.row.numb-nrow(count.matrix)))
#remove pre-miRNAs
count.matrix <- count.matrix[c(-grep("\\*", rownames(count.matrix))),]
print(paste("Removed * features:", non.human.data.row.numb-nrow(count.matrix)))
print(paste("Rows left: ", nrow(count.matrix)))

#assign annotation
annotation <- data@phenoData@data$source_name_ch1
annotation <-as.numeric(annotation)
annotation[annotation == 1] <- 0
annotation[annotation == 2] <- 1
annotation[annotation == 3] <- 0
annotation[annotation == 4] <- 1

#move negative samples first, positive samples next
count.matrix.neg <- count.matrix[,which(annotation %in% c(0))]
print(paste("Number of negative samples: ", ncol(count.matrix.neg)))
count.matrix.pos <- count.matrix[,which(annotation %in% c(1))]
print(paste("Number of positive samples: ", ncol(count.matrix.pos)))
count.matrix <- cbind(count.matrix.neg, count.matrix.pos)

#assign annotation
annotationneg <- rep(0,103)
annotationpos <- rep(1,103)
annotation <- c(annotationneg,annotationpos)

#transform data
original_ids <- colnames(count.matrix)
new_ids <- seq(1,length(colnames(count.matrix)),1)
sample.info <- data.frame(original_ids,new_ids,annotation)
write.table(sample.info, paste("GSE36681", "_sample_info.csv", sep=""),  sep = ";", row.names = FALSE, quote=FALSE)
colnames(count.matrix) <- seq(1,length(colnames(count.matrix)),1)

transformed.data.set <- transformData(annotation, count.matrix)

data.set.name = paste("GSE36681", ".csv", sep="", collapse = NULL)

write.table(transformed.data.set, data.set.name,  sep = ";", row.names = FALSE, quote=FALSE)