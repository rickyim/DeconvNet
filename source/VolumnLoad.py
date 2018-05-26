import numpy as np
import scipy.misc
import scipy.ndimage
import os

def VolumnLoad(idx, path=None):
    spt=path.split('/')
    if spt[4]=='Vessel0510' or spt[4]=='Vessel0501':
        path=path+'/Image%d/original_image/'%idx
        fileIterator=os.walk(path)
        volumn=[]
        namelist=[]
        for i,j,k in fileIterator:#theoretically only one 
            filepath=i
            namelist=k
        if len(namelist)==0:
            print(path)
            print('invalid file path')
            return
        else:
            namelist.sort()
            for filenames in namelist:
                volumn.append(scipy.misc.imread(os.path.join(filepath,filenames)))
            volumn=np.asarray(volumn).astype('float32')
            volumn=scipy.ndimage.zoom(volumn, 2.0)
            volumn=volumn[0:101, 0:101, 0:101]
        return volumn
    else:
        zstart=idx%3*10+35
        hstart=(idx//3)//100*10+500
        wstart=(idx//3)%100*10+500
        volumn=[]
        for z in range(101):
            zind=zstart+z
            im=scipy.misc.imread(os.path.join(path, '%d.tiff'%zind))
            im=im[hstart:hstart+101, wstart:wstart+101]
            volumn.append(im)
        volumn=np.asarray(volumn).astype('float32')
        volumn=volumn-np.min(volumn)# coarsely subtract the background for real time data
        volumn=volumn/np.max(volumn)
        return volumn
        
