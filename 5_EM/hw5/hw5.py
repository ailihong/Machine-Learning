import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

#read image
from PIL import Image
im = Image.open('D:/cs498ML/hw5/balloons.jpg')
rgb_im = im.convert('RGB')
print im.size
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
#print pixels


#use k-means to generate 10 centroids
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
from numpy import random
features  = pixels
whitened = whiten(features)
# book = array((whitened[0],whitened[2]))
# kmeans(whitened,book)
random.seed((1000,2000))
codes = 10
res=kmeans(whitened,10)
print res#res is the vector of centers


#E step
for i in range(len(pixels)):
    num=pixels[i]
    print num
    for j in range(0,10):
      print res[0][j]
      temp=np.dot(num-res[j],num-res[j])
      print temp



#M step


