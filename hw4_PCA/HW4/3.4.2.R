setwd('~/Documents/cs498hw4')
iris.data<-read.csv("iris.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)
library(caret)

log.iris.data<-log(iris.data[, 1:4])
iris.species<-iris.data[, 5]

ir.pca <- prcomp(iris.data[, 1:4], center = TRUE, scale = TRUE)
ir.pca.eigenvector <- t(ir.pca$rotation)
ir.pca.score <- ir.pca$x
ir.pca.sd = ir.pca$sdev

library(devtools)
##install_github("vqv/ggbiplot")
require(ggbiplot)

graph <- ggbiplot(ir.pca, scale = 0, var.scale = 0, group = iris.species, ellipse = TRUE, circle = TRUE)
graph <- graph + scale_color_discrete(name = '')
graph <- graph + theme(legend.direction = 'horizontal', legend.position = 'top')
