require("glmnet")
require("MASS")

setwd('D://cs498ML//hw6//assignment6')
#tracks<-read.csv('default_features_1059_tracks.txt', header=FALSE)
tracks<-read.csv('default_plus_chromatic_features_1059_tracks.txt', header=FALSE)
tr1<-tracks[,-c(118)]


foo<-lm(V117~., data=tr1)

a <- predict(foo, tr1,interval="prediction")
latfoo <- data.frame(trueValue=tracks[,c(117)],predict=a[,1])

plot(latfoo)
abline(foo)
summary(foo)$r.squared
#as.numeric(latfoo.lm)
#boxcox()








library(MASS)
wdat<-read.csv('default_features_1059_tracks.txt', header=FALSE)

wdat[,69]<-wdat[,69]+100
mus<-wdat[,-c(69,70)]
datX<-as.matrix((mus))
lat<-wdat[,69]
long<-wdat[,70]
#linear regression
#lat

bar2<-boxcox(wdat[,69]~datX,lambda = seq(-10, 20, length = 100))
lambdalat=with(bar2, x[which.max(y)])

latbc=((wdat[,69])^lambdalat-1)/lambdalat


right=t(mus) %*% as.matrix(latbc)
left=t(datX) %*% datX
beta= solve(left,right)
pred_lat=(datX)%*%beta
#plot((datX)%*%beta,lat)        #

pl<-pred_lat
lat<-stack(wdat, select=69)
foo<-data.frame(latitude=lat[,c("values")],predict_latitude=pl)
foo.lm<-lm(predict_latitude~latitude, data=foo)
plot(foo)
abline(foo.lm)
R_lat=var(pred_lat)/var(wdat[,69])
print(R_lat)














#long
right1=t(mus) %*% as.matrix(long)
beta1= solve(left,right1)
pred_long=(datX)%*%beta1
#plot((datX)%*%beta,lat)        #


pl<-pred_long
long<-stack(wdat, select=70)
foo<-data.frame(longitude=long[,c("values")],predict_longitude=pl)
foo.lm2<-lm(predict_longitude~longitude, data=foo)
plot(foo)
abline(foo.lm2)
R_long=var(pred_long)/var(wdat[,70])
print(R_long)



















tr1[,69]<-tr1[,69]+100
bar<-boxcox(foo.lm)
with(bar, x[which.max(y)])

tr1[,117]<- ( (tr1[,117]-100)**2 -1 )/2
foo<-lm(V117~., data=tr1)

a <- predict(foo, tr1,interval="prediction")
latfoo <- data.frame(trueValue=tracks[,c(117)],predict=a[,1])

plot(latfoo)
abline(foo)
summary(foo)$r.squared

#####
#####


tr2<-tracks[,-c(117)]

foo<-lm(V118~., data=tr2)
a <- predict(foo, tr2,interval="prediction")
latfoo <- data.frame(trueValue=tracks[,c(118)],predict=a[,1])
plot(latfoo)
abline(foo)
summary(foo)$r.squared

tr2[,117]<-tr2[,117]+100
foo<-lm(V118~., data=tr2)
bar2<-boxcox(foo)
with(bar2, x[which.max(y)])


tr2[,117]<- ( (tr2[,117]-100)**0.9899 -1 )/0.9899
foo<-lm(V118~., data=tr2)
a <- predict(foo, tr2,interval="prediction")
latfoo <- data.frame(trueValue=tracks[,c(118)],predict=a[,1])
plot(latfoo)
abline(foo)
summary(foo)$r.squared

## glmnet
feature <- data.matrix(tracks[,1:116])
latitude <- data.matrix(tracks[,117])
latitude <- (latitude**2 - 1) / 2

longtitude <- data.matrix(tracks[,118])
longtitude <- (longtitude - 1)

## lati 1  variable 91
lati_regression <- cv.glmnet(feature, latitude)
plot(lati_regression)
print(lati_regression)

# 2 variable 100
lati_regression <- cv.glmnet(feature, latitude, alpha = 0)
plot(lati_regression)
print(lati_regression)
min(lati_regression$cvm)


#0.25  278.1289 98
lati_regression <- cv.glmnet(feature, latitude, alpha = 0.25)
plot(lati_regression)
print(lati_regression)
min(lati_regression$cvm)
#regularize better

#0.5 274.2872 92 
lati_regression <- cv.glmnet(feature, latitude, alpha = 0.5)
plot(lati_regression)
print(lati_regression)
min(lati_regression$cvm)
#regularize


#0.75  277.1098   91 variables
lati_regression <- cv.glmnet(feature, latitude, alpha = 0.75)
plot(lati_regression)
print(lati_regression)
min(lati_regression$cvm)
#regularize


## long 1
long_regression <- cv.glmnet(feature, longtitude)
plot(long_regression)
print(long_regression)
min(long_regression$cvm)

# 2
long_regression <- cv.glmnet(feature, longtitude, alpha = 0)
plot(long_regression)
print(long_regression)
min(long_regression$cvm)

#0.25  
long_regression <- cv.glmnet(feature, longtitude, alpha = 0.25)
plot(long_regression)
print(long_regression)
min(long_regression$cvm)
#regularize better

#0.5 
long_regression <- cv.glmnet(feature, longtitude, alpha = 0.5)
plot(long_regression)
print(long_regression)
min(long_regression$cvm)
#regularize


#0.75  
long_regression <- cv.glmnet(feature, longtitude, alpha = 0.75)
plot(long_regression)
print(long_regression)
min(long_regression$cvm)


