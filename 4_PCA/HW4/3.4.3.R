install.packages("pls")
library(pls)
setwd('~/Documents/cs498hw4')
iris.data<-read.csv("iris.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)
library(caret)

iris.species<-iris.data[, 5]
install.packages("chemometrics")
library(chemometrics)
numiris = iris.data[,c(1,2,3,4)]
postscript('irisscatterplot.eps')
speciesnames <- c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
pchr <- c(1,2,3)
colr <- c('red','green','blue','yellow','orange')
ss <- expand.grid(species = 1:3)
parset <- with(ss, simpleTheme(pch = pchr[species],
                               col = colr[species]))


re1<-matrix(nrow=150,ncol=3)

re1[1:100,1]=0
re1[101:150,1]=1

re1[101:150,2]=0
re1[1:50,2]=0
re1[51:100,2]=1


re1[1:50,3]=1
re1[51:150,3]=1








res<-pls2_nipals(numiris, re1, 2, it = 50, tol = 1e-08, scale = FALSE)

projection=res$T
plot_DF=data.frame(x=-projection[,1],y=-projection[,2],class=iris.species)
attach(plot_DF)

plot(x,y,col=c("red","blue","green")[class],xlab='discriminative direction 1',ylab='discriminative direction 2')
detach(plot_DF)
legend(x="topright",legend=levels(plot_DF$class),col=c("red","blue","green"),pch=1)
dev.off()

