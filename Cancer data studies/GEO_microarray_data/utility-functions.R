#Transform data ------------------------------------------------------
# transform data set in a following format:
# ID | Annots | gene1 | ...
#  1 |      0 |   100 | ...
#  2 |      0 |   200 | ...

transform_data <- function(annotation, data) {
  
  ID <- colnames(data) # store column names (sample ids)
  transformed.data.set <- t(data) #transform the count matrix
  id.annotation <- cbind(ID, annotation, row.names = NULL) #bind data with annotation
  colnames(id.annotation) <- c("ID", "Annots")
  transformed.data.set <- cbind(id.annotation, transformed.data.set, row.names = NULL) #bind data with IDs
  row.names(transformed.data.set) <- ID
  
  return(as.data.frame.matrix(transformed.data.set))
}

# Transform and save data --------------------------------------------
save_matrix <- function(file_name, annotation, data.set) {
  
  data.set.to.write <- transform_data(annotation, data.set)
  
  write.table(data.set.to.write, file_name,  sep = ";", row.names = FALSE, quote=FALSE)
  
}



