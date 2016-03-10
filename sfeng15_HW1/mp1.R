setwd('C:/Users/shuofeng6/Desktop/mp1')
dat=read.csv('C:/Users/shuofeng6/Desktop/mp1/data.txt.txt', header = FALSE)
library(klaR)
library(caret)


#divide matrix into label and data
bigy=dat[,9]
bigx=dat[-c(9)]

"
bigx=wdat[,-c(9)]
bigx=as.matrix(bigx)
bigx=as.data.frame(bigx)
"

#partition the whole matrix
wtd=createDataPartition(y=bigy, p=.8, list=FALSE)

#extract training matrix
trbx=bigx[wtd,]
trby=bigy[wtd]
posflag=trby==1

#extract training matrix for each class
ptregs=trbx[posflag,]
ntregs=trbx[!posflag,]

#calculate the mean and std for each class
ptrmean=sapply(ptregs, mean, na.rm=TRUE)
ntrmean=sapply(ntregs, mean, na.rm=TRUE)
ptrsd=sapply(ptregs, sd, na.rm=TRUE)
ntrsd=sapply(ntregs, sd, na.rm=TRUE)

#calculate prior p 
pprior=length(ptregs[,1])/(length(ptregs[,1])+length(ntregs[,1]))
nprior=length(ntregs[,1])/(length(ptregs[,1])+length(ntregs[,1]))


#extract testing matrix which is the rest 20%
tex=bigx[-wtd, ]
tey=bigy[-wtd]

#applying prior and likelihood probability
pteoffsets=t(t(tex)-ptrmean)
ptescales=t(t(pteoffsets)/ptrsd)
ptelogs=-(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))

nteoffsets=t(t(tex)-ntrmean)
ntescales=t(t(nteoffsets)/ntrsd)
ntelogs=-(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
res=(ptelogs+log(pprior))>(ntelogs+log(nprior))
gotright=res==tey
tescore=sum(gotright)/(sum(gotright)+sum(!gotright))


