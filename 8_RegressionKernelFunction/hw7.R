library(np)
require("glmnet")
setwd('/Users/Yuchen/Desktop/rain/school course/spring 2016/cs498 applied ml/assignment7')
location <- read.table('Locations.txt', header = FALSE)
oregon_data <- read.table('Oregon_Met_Data.txt', header = FALSE)


# Problem 2
ndat <- dim(location)[1]
srange <- c(1.0,2,2.5,3,3.5,4)
east <- as.numeric(as.matrix(location[-c(1), ]$V7)) / 1000
north <- as.numeric(as.matrix(location[-c(1), ]$V8)) / 1000
xmat <- cbind(east, north)

spaces <- dist(xmat, method = "euclidean", diag = FALSE, upper = FALSE)
msp <- as.matrix(spaces)
wmat <- exp(-msp/(2*srange[1]^2))
for (i in 2:6){
  gramat <- exp(-msp/(2*srange[i]^2))
  wmat <- cbind(wmat, gramat)
}

# parse temperature
temp <- oregon_data[-c(1), c(1,6)]
ind <- which(with(temp, temp[2] == 9999))
temp <- temp[-ind, ]
temp <- as.matrix(temp)

x <- data.frame(Category=factor(as.factor(temp[,1])),Frequency=as.numeric(as.matrix(temp[,2])))
result <- aggregate(x$Frequency, by=list(Category=x$Category), FUN=mean)

newmat <- matrix(nrow = 112, ncol = 1)
for (i in 1 : 112){
  newmat[i] =as.numeric( result[result[,1]==i,][2])
}

newmat<-as.matrix(newmat)

wmod <- cv.glmnet(wmat, as.vector(as.numeric(newmat)), alpha = 0.5)

xmin <- min(xmat[,1])
xmax <- max(xmat[,1])
ymin <- min(xmat[,2])
ymax <- max(xmat[,2])
xvec <- seq(xmin, xmax, length = 100)
yvec <- seq(ymin, ymax, length = 100)

# these are the points

pmat <- matrix(0, nrow = 100*100, ncol = 2)

ptr <- 1
for (i in 1:100) {
  for (j in 1: 100) {
    pmat[ptr, 1] <- xvec[i]
    pmat[ptr, 2] <- yvec[j]
    ptr <- ptr + 1
  }
}

diff_ij <- function(i, j) sqrt(rowSums((pmat[i,] - xmat[j,])^2))
distsampletopts <- outer(seq_len(10000), seq_len(dim(xmat)[1]), diff_ij)
wmat <- exp(-distsampletopts/(2*srange[1]^2))

for (i in 2:6){
  gramat <- exp(-distsampletopts/(2*srange[i]^2))
  wmat <- cbind(wmat, gramat)
}

preds <- predict.cv.glmnet(wmod, wmat, s = "lambda.min")
zmat <- matrix(0, nrow = 100, ncol = 100)
ptr <- 1
for (i in 1:100) {
  for (j in 1: 100) {
    zmat[i, j] = preds[ptr]
    ptr <- ptr + 1
  }
}
install.packages("fields")
library("fields")
wscale = max(abs(min(preds)), abs(max(preds)))
image.plot(xvec, yvec, (t(zmat) + wscale) / (2*wscale),xlab = "east", ylab = "north", useRaster = TRUE)
image(yvec, xvec, (t(zmat) + wscale) / (2*wscale), xlab = "east", ylab = "north", 
      col = grey(seq(0, 1, length = 256)), useRaster = TRUE)

plot(wmod)