require("glmnet")
require("gdata")
setwd('D://cs498ML//hw9')



#p4
data=pomeroy.2002.v1_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err



data=pomeroy.2002.v2_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err



data=ramaswamy.2001_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err



data=shipp.2002.v1_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err


data=singh.2002_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err


data=su.2001_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err


data=west.2001_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err


data=yeoh.2002.v1_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err


data=yeoh.2002.v2_database
#get labels
y=data[,1]
y=as.factor(y)

#get features
x=data[,-c(1)]
x=data.matrix(x)

reg1 <- cv.glmnet(x, y, family = "multinomial",alpha=1,type.measure = "class")
plot(reg1)

pred <- predict(reg1, x, type = "class", s = "lambda.min")
nmright <- sum(y == pred)
err <- (1-nmright/dim(pred))
err

