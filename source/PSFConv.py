import numpy as np
import scipy.ndimage 
import scipy.misc
import glob
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def loadPsf(psftype, fileformat):
    path='/gdata/zhoutk/Deconv/'+psftype
    files=glob.glob(path+'/'+'*'+fileformat)
    length=len(files)
    if length==0:
        print(path+'/')       
        print('invalid psf file path')
        return
    im0=scipy.misc.imread(files[0])
    shape=im0.shape
    psf=np.zeros((length, shape[0], shape[1]))
    files.sort()
    for i, file in enumerate(files):
        #print(file)
        psf[i,:,:]=scipy.misc.imread(file)
        #print(type(psf))
        #print(psf.shape)
    return psf

def convolvePsf3D(volumn, psf, psnr):
    ##normalize with its largest content
    psf=psf/np.sum(psf)
    #change from int8 to float64
    volumn=volumn.astype('float64')
    #convolve psf with volumn
    #print(volumn.shape)
    #print(psf.shape)
    print('max_volumn: ', np.max(volumn))
    #volumn=scipy.ndimage.zoom(volumn, 2.0)
    if torch.cuda.is_available():
        psf=psf[:99, :, :]
        for i in range(len(psf.shape)):
            psf=np.flip(psf, i)
        psf=torch.from_numpy(psf.copy()).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).cuda()
        psf=Variable(psf, requires_grad=False)
        volumn=torch.from_numpy(volumn.copy()).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).cuda()
        volumn=Variable(volumn, requires_grad=False)
        output=F.conv3d(volumn, psf, padding=(49, 12, 12))
        output=output.squeeze().cpu().data.numpy()
    else:
        output=scipy.ndimage.filters.convolve(volumn, psf, mode='constant')
    print('convolve output shape: ', output.shape)
    print('max_output: ', np.max(output))
    #noise level --- gaussian noise
    sigma=np.max(output)/np.power(10, psnr/20)
    print('gaussian noise level:', sigma)
    noise=np.random.normal(0, sigma, output.shape)
    #add noise to the output
    output=np.clip(output+noise, 0, np.max(output))
    #output=output[0:101,0:101,0:101]
    return output

         

