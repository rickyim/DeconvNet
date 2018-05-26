import numpy as np
import scipy.ndimage 
from time import sleep
import sys
from BVGen import *
import scipy.misc
w=100
h=100
z=100
'''
x=np.linspace(0, w, 50)
y=np.linspace(0, h, 30)
print("start generating blood vessel model, Single Tube, Single Direction, ...")
dataSave=np.zeros((x.size* y.size,1, w, h))
for i in range(x.size):
    for j in range(y.size):
        for theta in range(360):
            ind = j*x.size+i
            x_new = np.rint(x[i]+r*np.cos(theta*2*np.pi/360))
            y_new = np.rint(y[j]+r*np.sin(theta*2*np.pi/360))
            if (x_new>=0 and x_new<w and y_new>=0 and y_new<h):
                dataSave[ind, 0, x_new.astype(int), y_new.astype(int)]=1
    sys.stdout.write('\r')
    sys.stdout.write("[%-60s] %d%%" % ('='*int(i/x.size*60), i/x.size*100))
    sys.stdout.flush()
    sleep(0.25)
'''
numofdata=1000
'''
print("start generating blood vessel model, random move, num of data: 1000  ...")
dataSave=np.zeros((numofdata,z, w, h))
for i in range(numofdata):
    dataSave[i,:,:,:]=newHull(z,w,h)
    sys.stdout.write('\r')
    sys.stdout.write("[%-60s] %d%%" % ('='*int(i/numofdata*60), i/numofdata*100))
    sys.stdout.flush()
    sleep(0.25)
print('generating done, data size: %d, %d, %d, %d'%(numofdata, z, w, h))
print('stacking size: %d' % dataSave.nbytes)
#save groundtruth
for i in range(numofdata):
    np.save('/gdata/zhoutk/Deconv/gtSave/gt%d.npy'%i, dataSave[i,:,:,:])
print("save ground truth done")
'''

#generate training data
print('start generating training data')
psnr=30
path='/gdata/zhoutk/Deconv/BornWolf10'
fileformat='tif'
#dataSave=dataSave.astype('float')
for i in range(numofdata):
    dataLoad=np.load('/gdata/zhoutk/Deconv/gtSave/gt%d.npy'%i)
    dataLoad=convolvePsf3D(dataLoad, path, fileformat, psnr)
    np.save('/gdata/zhoutk/Deconv/trSave/tr%d.npy'%i, dataLoad)
    print('save done', i)
    #dataSave[i,:,:,:]=convolvePsf3D(np.squeeze(dataSave[i,:,:,:]), path, fileformat, psnr)
#for i in range(numofdata):
#    np.save('/gdata/zhoutk/Deconv/trSave/tr%d.npy'%i, dataSave[i,:,:,:])
