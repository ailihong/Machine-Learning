wine.data <- read.csv("~/Documents/cs498hw4/wine.data.txt", header=FALSE)

covm=cov(wine.data[,2:14])
a=eigen(covm) 

max1=as.matrix(a$vectors[,1])
max2=as.matrix(a$vectors[,2])
max3=as.matrix(a$vectors[,3])
## then use matlab