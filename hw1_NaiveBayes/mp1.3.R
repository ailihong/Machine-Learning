setwd('C:/Users/shuofeng6/Desktop/mp1')
wdat=read.csv('C:/Users/shuofeng6/Desktop/mp1/data.txt.txt', header = FALSE)
library(klaR)
library(caret)

#partition the matrix into data and label
bigx=wdat[,-c(9)]
bigy=as.factor(wdat[,9])
wtd=createDataPartition(y=bigy, p=.8, list=FALSE)
# get the training data
trax=bigx[wtd,]
tray=bigy[wtd]
#do training
model=train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
#do testing
teclasses=predict(model,newdata=bigx[-wtd,])
confusionMatrix(data=teclasses, bigy[-wtd])
