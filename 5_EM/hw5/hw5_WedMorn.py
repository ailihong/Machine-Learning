import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten

#read image
from PIL import Image
im = Image.open('D:/cs498ML/hw5/balloons.jpg')
rgb_im = im.convert('RGB')
#print im.size
pix = im.load()

#tuple of a pixel
# print pix[1,1]

#three separate r g b values
#r, g, b = rgb_im.getpixel((1, 1))
#print r, g, b

# vector of r g b values
# c = np.array(pix[1,1])
# print c

pixels = []
for i in range(im.size[0]):
    for j in range(im.size[1]):
        tmp=np.array(pix[i,j])/255.0
        c = (list(tmp))
        pixels.append(c)
print pixels


#use k-means to generate 10 centroids

features  = pixels
whitened = whiten(features)
# book = array((whitened[0],whitened[2]))
# kmeans(whitened,book)
np.random.seed((1000,2000))
codes = 10
res=kmeans(whitened,10)
#res is the vector of centers
# print res[0]
center = res[0]  # this is mu, 10 * 3
print center
print center.shape


# pixel[i] is [r, g, b]

#E step

# numerator for E step:

# ne = np.zeros((len(pixels), 10))
# de = np.zeros(len(pixels))
# wij = np.zeros((len(pixels), 10)) # 2400*10
# pi=np.zeros(10)
# for i in range (10):
#     pi[i]=0.1
# for i in range(len(pixels)):
#   xi = pixels[i]
#   temp = 0;
#   for j in range(10):
#     sub = np.subtract(xi, center[j])
#     result = np.dot(np.transpose(sub), sub) * (-1.0 / 2)
#     ne[i][j] = np.exp(result); ## need to multiply pi
#     temp += ne[i][j]
#   de[i] = temp;
#   wij[i][:] = ne[i][:] / de[i]
#
# print ne.shape
# print de.shape
# print wij.shape
# print ne[0][0]
#
#
# #M step
# # don't know how to compute numerator for m step
# newcenter = np.zeros((10, 3))
# dm = np.zeros(10)
# for j in range(10):
#     dm[j] = np.sum(wij[:][j])
#     for i in range(len(pixels)):
#         newcenter[j]+=np.dot(wij[i][j],pixels[i])
#     newcenter[j]=newcenter[j]/dm[j]
#
# print newcenter
#
#
#
#
#
#
#
#
#
#
#
#
