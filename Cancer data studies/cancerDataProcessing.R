
#GSE22058/GPL10457
set.seed(1)
source("splitRealWorldData.R")
data <- getGEO(filename="GSE22058/GPL10457/GPL10457_series_matrix.txt")
count.matrix <- exprs(object = data)
count.matrix.pos <- count.matrix[,c(seq(1,191,2))]
count.matrix.neg <- count.matrix[,c(seq(2,192,2))]
count.matrix <- cbind(count.matrix.neg, count.matrix.pos)
rownames(count.matrix) <- data@featureData@data$miRNA_ID
count.matrix <- count.matrix[c(grep("hsa*", data@featureData@data$miRNA_ID)),]
annotationneg <- rep(0,96)
annotationpos <- rep(1,96)
annotation <- c(annotationneg, annotationpos)
splitRealWorldData("GSE22058_GPL10457", count.matrix, annotation, 96, 96, 0.8)

#GSE10694
set.seed(1)
source("splitRealWorldData.R")
data <- getGEO(filename="GSE10694/GSE10694_series_matrix.txt")
count.matrix <- exprs(object = data)
count.matrix.pos <- count.matrix[,1:78]
count.matrix.neg <- count.matrix[,79:156]
count.matrix <- cbind(count.matrix.neg, count.matrix.pos)
rownames(count.matrix) <- data@featureData@data$miRNA_ID
count.matrix <- count.matrix[c(grep("hsa*", data@featureData@data$miRNA_ID)),]
annotationneg <- rep(0,78)
annotationpos <- rep(1,78)
annotation <- c(annotationneg, annotationpos)
splitRealWorldData("GSE10694", count.matrix, annotation, 78, 78, 0.8)


#GSE36681
set.seed(1)
source("splitRealWorldData.R")
data <- getGEO(filename="GSE36681/GSE36681_series_matrix.txt")
count.matrix <- exprs(object = data)
rownames(count.matrix) <- data@featureData@data$miRNA_ID
count.matrix <- count.matrix[c(grep("hsa*", data@featureData@data$miRNA_ID)),]
anno <- data@phenoData@data$source_name_ch1
anno <-as.numeric(anno)
anno[anno == 1] <- 0
anno[anno == 2] <- 1
anno[anno == 3] <- 0
anno[anno == 4] <- 1
anno.FFPE <- anno[1:94]
anno.FF <- anno[95:length(anno)]
count.matrix.FFPE <- count.matrix[,1:94]
count.matrix.FFPE.neg <- count.matrix.FFPE[,which(anno.FFPE %in% c(0))]
count.matrix.FFPE.pos <- count.matrix.FFPE[,which(anno.FFPE %in% c(1))]
count.matrix.FFPE <- cbind(count.matrix.FFPE.neg, count.matrix.FFPE.pos)
annotationneg <- rep(0,47)
annotationpos <- rep(1,47)
annotation <- c(annotationneg,annotationpos)
splitRealWorldData("GSE36681_FFPE", count.matrix.FFPE, annotation, 47, 47, 0.8)
count.matrix.FF <- count.matrix[,95:length(colnames(count.matrix))]
count.matrix.FF.neg <- count.matrix.FF[,which(anno.FF %in% c(0))]
count.matrix.FF.pos <- count.matrix.FF[,which(anno.FF %in% c(1))]
count.matrix.FF <- cbind(count.matrix.FF.neg, count.matrix.FF.pos)
annotationneg <- rep(0,56)
annotationpos <- rep(1,56)
annotation <- c(annotationneg,annotationpos)
splitRealWorldData("GSE36681_FF", count.matrix.FF, annotation, 56, 56, 0.8)
