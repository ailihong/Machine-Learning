setwd('C:/Users/shuofeng6/Desktop/mp1')
rm(list=ls())
wdat=read.csv('C:/Users/shuofeng6/Desktop/mp1/data.txt.txt', header = FALSE)
library(klaR)
library(caret)



#partition the matrix into data and label
bigx=wdat[,-c(9)]
bigy=as.factor(wdat[,9])
# get the training data
wtd=createDataPartition(y=bigy, p=.8, list=FALSE)
#do training
svm=svmlight(bigx[wtd,], bigy[wtd], pathsvm='C:/Users/shuofeng6/Desktop/mp1/svm_light_windows64/')
#do testing
labels=predict(svm, bigx[-wtd,])
foo=labels$class
sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))