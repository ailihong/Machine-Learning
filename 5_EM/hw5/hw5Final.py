import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
import scipy
import matplotlib.image as mpimg
import random


#read image
from PIL import Image
im = Image.open('/Users/Yuchen/Desktop/rain/school course/spring 2016/cs498 applied ml/assignment5/smallballoons.jpg')


rgb_im = im.convert('RGB')
image = np.asarray(im)
print image.shape
#print im.size
pix = im.load()
flag = 0

#tuple of a pixel
# print pix[1,1]

#three separate r g b values
#r, g, b = rgb_im.getpixel((1, 1))
#print r, g, b

# vector of r g b values
# c = np.array(pix[1,1])
# print c

pixels = []
for i in range(im.size[1]):
    for j in range(im.size[0]):
        tmp=np.array(pix[j,i]) / 255.0
        c = (list(tmp))
        pixels.append(c)



# # print pixels
# #use k-means to generate 10 centroids
# #
# features  = pixels
# whitened = whiten(features)
# # book = array((whitened[0],whitened[2]))
# # kmeans(whitened,book)
# np.random.seed((1000,2000))
# codes = 10
# res=kmeans(whitened,10)
# #res is the vector of centers
# # print res[0]
# center = res[0]  # this is mu, 10 * 3
# # print center.shape
seg = 10
center = np.zeros((seg, 3))

for i in range(10):
    center[i]=pixels[random.randint(1,im.size[1]*im.size[0]-1)]

# #
pi = np.zeros(seg)
for i in range(seg):
    pi[i] = 1.0 / seg
# #
ne = np.zeros((len(pixels), seg))
de = np.zeros(seg)
# de = np.zeros((len(pixels)))
wij = np.zeros((len(pixels), seg)) # 240000*10

dm = np.zeros(seg)
myQ = np.zeros((len(pixels), seg))
# #
# # # pixels[i] is [r, g, b]
# numerator for E step:
count = 0
tol = 0.001
Q = 0.1
Qprev = 0
# #
while abs((Qprev - Q) / Q) > tol and count < 30:

    #print center
    #E step
    print abs((Qprev - Q) / Q)

    for i in range(len(pixels)):
      xi = pixels[i]
      temp = 0;
      for j in range(seg):
        sub = np.subtract(xi, center[j])
        result = np.dot(np.transpose(sub), sub) * (-1.0 / 2)
        ne[i][j] = np.exp(result) * pi[j]; ## need to multiply pi
        myQ[i][j] = result + np.log(pi[j])


    for i in range(seg):
        de[i] = np.sum(ne[:,i])


#so far
        wij[:,i] = ne[:,i] / de[i]

    #M step

    newcenter = np.zeros((seg, 3))

    for j in range(seg):
        dm[j] = np.sum(wij[:,j])
        for i in range(len(pixels)):
            newcenter[j] += np.dot(pixels[i], wij[i][j])
        print newcenter[j]
        print dm[j]
        newcenter[j] = newcenter[j] / dm[j]
        # print newcenter[j]
        print newcenter[j]

    ## update pi
    pi = dm / len(pixels)
    #pi = dm / np.sum(wij[i][j])

    Qprev = Q

    Q = 0

    for i in range(len(pixels)):
        for j in range(seg):
            Q += wij[i][j] * myQ[i][j]

    center = newcenter
    count = count + 1
#
newPixel = np.zeros((len(pixels), 3))
for i in range(len(pixels)):
    maxnum = max(wij[i])
    index = list(wij[i]).index(maxnum)
    rgb = center[index]
    newPixel[i] = rgb

newPicture = np.ones(image.shape, dtype=np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        newPicture[i][j] = np.dot(np.asarray(newPixel[i * image.shape[1] + j]), 255)



img = Image.fromarray(newPicture, "RGB")
img.save('my.jpg')
