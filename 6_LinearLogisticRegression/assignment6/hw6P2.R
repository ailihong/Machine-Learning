require("glmnet")
require("gdata")
setwd('D://cs498ML//hw6//assignment6')
data <- read.xls('default_of_credit_card_clients', perl='D://perl//bin//perl.exe',header = FALSE)
data <- data[-c(1:2),]
test <- data[20001:30000,]
train <- data[1:20000,]
x <- data.matrix(train[,2:24])
y <- as.numeric(train[,25])
y <- as.factor(y)
reg1 <- cv.glmnet(x, y, family = "binomial", alpha = 1, type.measure = "class")
plot(reg1)

reg2 <- cv.glmnet(x, y, family = "binomial", alpha = 0, type.measure = "class")
plot(reg2)

reg0_25 <- cv.glmnet(x, y, family = "binomial", alpha = 0.25, type.measure = "class")
plot(reg0_25)

reg0_5 <- cv.glmnet(x, y, family = "binomial", alpha = 0.5, type.measure = "class")
plot(reg0_5)

reg0_75 <- cv.glmnet(x, y, family = "binomial", alpha = 0.75, type.measure = "class")
plot(reg0_75)

xtest <- data.matrix(test[,2:24])
ytest <- as.numeric(test[,25])

l1pred <- predict(reg1, xtest, type = "class", s = "lambda.min")
l1right <- sum(ytest == l1pred)
l1err <- (1-l1right/dim(l1pred))