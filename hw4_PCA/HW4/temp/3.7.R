
#3.7.a

setwd('/Users/Yuchen/Desktop/rain/school course/spring 2016/cs498 applied ml/assignment4')
bcw<-read.csv("breast-cancer-wisconsin.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)
wdbc<-read.csv("wdbc.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)
wpbc<-read.csv("wpbc.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)

wdbc.species = revalue(wdbc[, 2], c("M" = 1, "B" = 2))
wdbc.data = wdbc[,3:32]

wdbc.pca <- prcomp(wdbc.data, center = TRUE, scale = TRUE, na.omit = TRUE)
wdbc.pca.rotation <- wdbc.pca$rotation
wdbc.pca.first3 <- wdbc.pca.rotation[,c(1,2,3)]

wdbc.pca.1 <- as.matrix(wdbc.data) %*% wdbc.pca.first3[,1]
wdbc.pca.2 <- as.matrix(wdbc.data) %*% wdbc.pca.first3[,2]
wdbc.pca.3 <- as.matrix(wdbc.data) %*% wdbc.pca.first3[,3]

library(scatterplot3d)
scatterplot3d(wdbc.pca.1, wdbc.pca.2, wdbc.pca.3, color = wdbc.species)

#3.7.b
library('chemometrics')
one_hot <- matrix(data = 0, nrow = 569, ncol = 2)
for (i in 1:569){
  if (wdbc.species[i] == 1){
    one_hot[i,1] = 1
  }
  else{
    one_hot[i,2] = 1
  }
}
  
res <- pls2_nipals(wdbc.data, one_hot, 3, it = 50, tol = 1e-08, scale = FALSE)

projection=res$T
plot_DF=data.frame(x=-projection[,1],y=-projection[,2],z = -projection[,3],class=wdbc.species)
scatterplot3d(-projection[,1], -projection[,2], -projection[,3], color = wdbc.species)
