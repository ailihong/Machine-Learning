setwd('/Users/Yuchen/Desktop/rain/school course/spring 2016/cs498 applied ml/assignment4')
wine.data<-read.csv("wine.data", header = FALSE, stringsAsFactors=FALSE, fill=TRUE)

# 3.5.1
wine.label <- wine.data[, c(1)]
wine.cov <- cov(wine.data[,c(-1)])
wine.pca <- prcomp(wine.data[,c(-1)], center = TRUE, scale = TRUE)
wine.eigvector <- wine.pca$rotation
wine.eigvalue <- wine.pca$sdev^2

plot(wine.eigvalue)

# 3.5.3
library(devtools)
##install_github("vqv/ggbiplot")
require(ggbiplot)

graph <- ggbiplot(wine.pca,  group = wine.label, ellipse = TRUE, circle = TRUE)
graph <- graph + theme(legend.direction = 'horizontal', legend.position = 'top')