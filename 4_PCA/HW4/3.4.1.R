setwd('/Users/Yuchen/Desktop/rain/school course/spring 2016/cs498 applied ml/assignment4')
iris.data<-read.csv("iris.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)
library("lattice")
numiris = iris.data[,c(1,2,3,4)]
postscript('irisscatterplot.eps')
speciesnames <- c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
pchr <- c(1,2,3)
colr <- c('red','green','blue','yellow','orange')
ss <- expand.grid(species = 1:3)
parset <- with(ss, simpleTheme(pch = pchr[species],
                               col = colr[species]))
splom(iris.data[,c(1:4)], groups = iris.data$V5, par.settings = parset,
      varnames = c('Sepal\nLength', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth'),
      key = list(text = list(speciesnames),
                 points = list(pch = pchr), columns = 3))
dev.off()


