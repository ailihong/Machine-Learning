setwd('C:/Users/shuofeng6/Desktop/mp1')
wdat=read.csv('C:/Users/shuofeng6/Desktop/mp1/data.txt.txt', header = FALSE)
library(klaR)
library(caret)


#divide matrix into label and data
bigy=wdat[,9]
bigx=wdat[,-c(9)]

#substitute NA for missing values
for (i in c(3, 4, 6, 8))
{
  vw=bigx[, i]==0
	bigx[vw, i]=NA
}


wtd=createDataPartition(y=bigy, p=.8, list=FALSE)

#extract training matrix
ntrbx=bigx[wtd,]
ntrby=bigy[wtd]
posflag=ntrby==1

#extract training matrix for each class
ptregs=ntrbx[posflag,]
ntregs=ntrbx[!posflag,]

#calculate the mean and std for each class
ptrmean=sapply(ptregs, mean, na.rm=TRUE)
ntrmean=sapply(ntregs, mean, na.rm=TRUE)
ptrsd=sapply(ptregs, sd, na.rm=TRUE)
ntrsd=sapply(ntregs, sd, na.rm=TRUE)

#calculate prior p 
pprior=length(ptregs[,1])/(length(ptregs[,1])+length(ntregs[,1]))
nprior=length(ntregs[,1])/(length(ptregs[,1])+length(ntregs[,1]))


#extract testing matrix which is the rest 20%
ntebx=bigx[-wtd, ]
nteby=bigy[-wtd]


pteoffsets=t(t(ntebx)-ptrmean)
ptescales=t(t(pteoffsets)/ptrsd)
ptelogs=-(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
nteoffsets=t(t(ntebx)-ntrmean)
ntescales=t(t(nteoffsets)/ntrsd)
ntelogs=-(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
lvwte=(ptelogs+log(pprior))>(ntelogs+log(nprior))
gotright=lvwte==nteby
tescore=sum(gotright)/(sum(gotright)+sum(!gotright))


