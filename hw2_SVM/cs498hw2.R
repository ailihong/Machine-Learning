setwd('~/Downloads/CS498HW2')
library(plotly)
library(caret)
wdat<-read.csv('adult.data.txt',header=FALSE)
bigx<-wdat[-c(15,2,4,6,7,8,9,10,14)]
## this remove the last column
bigy<-wdat[,15]
wtd<-createDataPartition(bigy, 
                         p = 0.8,
                         list=FALSE)
trainbx=bigx[wtd,]
trainy=bigy[wtd]
trainx=trainbx

##for (pi in 1:6){
##meanb=mean(unlist(trainbx[pi]))
##sdb=sd(unlist(trainbx[pi]))
##trainx[pi]=  (trainbx[pi]-meanb )/ sdb
##}



remainx=bigx[-wtd,]
remainy=bigy[-wtd]
validpar<-createDataPartition(remainy, 
                         p = 0.5,
                         list=FALSE)
validationx=remainx[validpar,]
validationy=remainy[validpar]

testx=remainx[-validpar,]
testy=remainy[-validpar]


trainflag<- trainy == " >50K"


ptrain<-trainx[trainflag,-c(15)]
## seperate true false
ntrain<-trainx[!trainflag,-c(15)]
## a b 0 

labely<-vector(mode="list",length=length(trainy))
##  labely : true y labels
for (k in 1:length(trainy)){
  if (trainflag[k]==TRUE) labely[k]<-1
  else labely[k]<--1
}



lambdalist=c(1e-8,1e-2,1e-1,1)

localaccuracy<-matrix(nrow=4,ncol=500)

for (li in 1:1){
lambda<-lambdalist[li]

a=c(0,0,0,0,0,0)
b=0
## start from 0

for (epoch in 1:50){
steplength=1/(0.01*epoch+50)
index=(epoch*300-299):(epoch*300)
thisx<-trainx[index,]
thisy<-labely[index]
##index<-createDataPartition(unlist(labely), 
 ##                                p = 0.0115,
  ##                               list=FALSE)
thisx<-trainx[index,]
thisy<-labely[index]

randomindex<-createDataPartition(unlist(thisy), 
                              p = 0.166,
                              list=FALSE)
for (ni in 1:300){
    pred=unlist(a) %*% unlist(thisx[ni, ]) + b
    ###
  if (unlist(thisy[ni])*pred>=1){
    pa=lambda*a
    pb=0
  }else{
    pa=lambda*a-unlist(thisy)[ni]*thisx[ni,]
    pb=-unlist(thisy)[ni]
  }
  a=a-steplength*pa
  b=b-steplength*pb
  if (ni %% 30 ==0){
    countproper<-0
    for (ki in 1:50){
      if ((  unlist(a) %*% unlist(thisx[randomindex[ki],]) +b )*unlist(thisy[randomindex[ki]])>=1){
        countproper=countproper+1
      }
    }
    localaccuracy[li,(epoch-1)*10+ni/30]=countproper/50
    
  }
  
}
}  


}

plot(localaccuracy[1,],type="s",xlab="trial(step/30)",ylab="accuracy for lambda=1e-3")