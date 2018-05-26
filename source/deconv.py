#!/usr/local/bin/python3

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PSFConv import *
from VolumnLoad import *
import time
import random


def TotalVariation3d(volumn):
    return torch.mean(torch.abs(volumn[:,:,:,:,:-1]-volumn[:,:,:,:,1:]))+ \
              torch.mean(torch.abs(volumn[:,:,:,:-1,:]-volumn[:,:,:,1:,:]))+ \
              torch.mean(torch.abs(volumn[:,:,:-1,:,:]-volumn[:,:,1:,:,:]))


class DeconvDataSet(Dataset):
    def __init__(self, gt_dir, tr_dir, start, length):
        self.gt_dir=gt_dir
        self.tr_dir=tr_dir
        self.start=start
        self.length=length
        self.psf=loadPsf('MVXPSF', '.tif')
	
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #gtFile_path=os.path.join(self.gt_dir,'gt%d.npy'%(idx+self.start))
        #trFile_path=os.path.join(self.tr_dir,'tr%d.npy'%(idx+self.start))
        #gt=np.load(gtFile_path)
        #tr=np.load(trFile_path)
        #print(gt.dtype)
        #print(tr.dtype)
        #gt=gt.astype('float64')
        #tr=PSFConv(gt,'BornWolf10')
        #print('load%d'%idx)
        #print(tr.shape)
        gt=VolumnLoad(self.start+idx, self.gt_dir)
        print('loading files: ',self.start,idx)
        gt=gt.astype('float64')
        #random transpose the dimension
        order=np.random.permutation([0,1,2])
        gt=np.transpose(gt, order)
        #flip order
        flpdim=np.random.randint(0, 3)
        gt=np.flip(gt, flpdim)
        #gt=gt/127.5-1
        #tr=gt
        print(gt.max())
        print(gt.min())
        #print(gt.max())
        #psnr=random.uniform(20, 30)
        psnr=30
        tr=convolvePsf3D(gt, self.psf, psnr)
        #normalize#
        gt=(gt/gt.max()-0.5)*2
        tr=(tr/tr.max()-0.5)*2
        #print(gt.dtype)
        #print(tr.dtype)
        #print(gt.shape, gt.nbytes)
        #print(tr.shape, tr.nbytes)
        return {'tr': torch.from_numpy(tr), 'gt': torch.from_numpy(gt)}
    


#define a simple convNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #c3_res3_c2
        '''
        self.conv1 = nn.Conv3d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv3d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv3d(32, 32, 5, padding=2)
        self.conv4 = nn.Conv3d(32, 32, 5, padding=2)
        self.conv5 = nn.Conv3d(32, 32, 5, padding=2)
        self.conv6 = nn.Conv3d(32, 32, 5, padding=2)
        self.conv7 = nn.Conv3d(32, 16, 5, padding=2)
        self.conv8 = nn.Conv3d(16, 1, 5, padding=2)
        '''
        self.conv1 = nn.Conv3d(1, 4, 5, padding=2)
        self.conv2 = nn.Conv3d(4, 16, 5, padding=2)
        self.enc1 = nn.Conv3d(16, 32, 5, stride=2, padding=2)
        self.enc2 = nn.Conv3d(32, 64, 5, stride=2, padding=2)
        #self.enc3 = nn.Conv3d(64, 128, 5, stride=2, padding=2)
        #self.enc4 = nn.Conv3d(128, 256, 5, stride=2, padding=2)
        #self.dec4 = nn.ConvTranspose3d(256, 128, 5, stride=2, padding=2)
        #self.dec3 = nn.ConvTranspose3d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.dec2 = nn.ConvTranspose3d(64, 32, 5, stride=2, padding=2, )
        self.dec1 = nn.ConvTranspose3d(32, 16, 5, stride=2, padding=2)
        self.conv3= nn.Conv3d(16, 8, 5, padding=2)
        self.conv4= nn.Conv3d(8, 1, 5, padding=2)
        self.tanh1= nn.Tanh()
        self.tanh2= nn.Tanh()
        self.tanh3= nn.Tanh()
        self.tanh4= nn.Tanh()
        self.bm1= nn.BatchNorm3d(4)
        self.bm2= nn.BatchNorm3d(16)
        self.bm3= nn.BatchNorm3d(32)
        self.bm4= nn.BatchNorm3d(64)
        self.bm5= nn.BatchNorm3d(32)
        self.bm6= nn.BatchNorm3d(16)
        self.bm7= nn.BatchNorm3d(8)
        
    def forward(self, x):
        #c3_res3_c2
        '''
        x1 = (self.conv1(x))
        x2 = (self.conv2(x1))
        x3 = (self.conv3(x2))
        x4 = (self.conv4(x3))
        x4 = x4 + x3
        x5 = (self.conv5(x4))
        x5 = x5 + x4
        x6 = (self.conv6(x5))
        x6 = x6 + x5
        x7 = (self.conv7(x6))
        x8 = (self.conv8(x7))
        '''
        print('forward...')
        x=self.bm1(self.tanh1(self.conv1(x)))
        x=self.bm2(self.tanh2(self.conv2(x)))
        print(x.cpu().data.min())
        print(x.cpu().data.max())
        x_e1=self.bm3(self.tanh1(self.enc1(x)))
        x_e2=self.bm4(self.tanh2(self.enc2(x_e1)))
        print(x_e2.cpu().data.min())
        print(x_e2.cpu().data.max())
        #x=(self.enc3(x))
        #x=(self.enc4(x))
        #x=(self.dec4(x))
        #x=(self.dec3(x))
        x_d2=self.bm5(self.tanh3(self.dec2(x_e2)))
        x_d2=x_d2+x_e1
        x_d1=self.bm6(self.tanh4(self.dec1(x_d2)))
        print(x_d1.cpu().data.min())
        print(x_d1.cpu().data.max())
        x_d1=x_d1+x
        x_out1=self.bm7(self.tanh3(self.conv3(x_d1)))
        print(x_out1.cpu().data.min())
        print(x_out1.cpu().data.max())
        x_out=self.tanh4(self.conv4(x_out1))
        print('forward done')
        return x_out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DiscrimNet(nn.Module):
    def __init__(self):
        super(DisrimNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, 5, padding=2)
        self.conv2 = nn.Conv3d(4, 16, 5, padding=2)
        self.enc1 = nn.Conv3d(16, 32, 5, stride=2, padding=2)
        self.enc2 = nn.Conv3d(32, 64, 5, stride=2, padding=2)
        self.tanh1= nn.Tanh()
        self.tanh2= nn.Tanh()
        self.bm1= nn.BatchNorm3d(4)
        self.bm2= nn.BatchNorm3d(16)
        self.bm3= nn.BatchNorm3d(32)
        self.bm4= nn.BatchNorm3d(64)

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform(m.weight.data)
        #nn.init.xavier_uniform(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_uniform(m.weight.data)
        #nn.init.xavier_uniform(m.bias.data)


#define a net
net = Net()
print(net)
#params = list(net.parameters())
optimizer=optim.Adam(net.parameters(), lr=0.001)
criterion=nn.MSELoss()

#using Gpus
if torch.cuda.is_available():    
    print("cuda is available...")
    if torch.cuda.device_count() > 1:
        print('using %d Gpus'%torch.cuda.device_count())
        net=nn.DataParallel(net)
    net.cuda()
else:
    print('cuda disabled')

#load pretrained parameters
lambda_TV=1
realdata_path='/gdata/zhoutk/Deconv/VesselReal'
save_path='/gdata/zhoutk/Deconv/params/0518fakedatarealpsf_few_tv1_from_start.pt'
param_path=None
if param_path is None:
    net.apply(weights_init)
    print('random initialize the net parameters')
else:
    net.load_state_dict(torch.load(param_path))
    print('load parameters from path: ', param_path)


'''
#load blur kernel
blur = Variable(torch.randn(1, 1, 5, 5, 5).cuda());
'''
print('loading dataset')
dataSet=DeconvDataSet('/gdata/zhoutk/Deconv/Vessel0510', '/gdata/zhoutk/Deconv/trSave', 0, 1500)
dataLoader=DataLoader(dataSet, batch_size=4, shuffle=True, num_workers=0)
testSet=DeconvDataSet('/gdata/zhoutk/Deconv/Vessel0510', '/gdata/zhoutk/Deconv/trSave', 1500,200)
testLoader=DataLoader(testSet, batch_size=4, shuffle=True, num_workers=0)

time_total=0
t=0
print('start training...')
for epoch in range(40):
    print('start epoch %d' %epoch)
    '''
    #load input
    X = Variable(torch.randn(10, 1, 512, 432, 116).cuda())
    #X = Variable(torch.randn(10, 1, 10, 10, 10).cuda())

    #blur the input
    input = F.conv3d(X, blur, padding=2)
    '''
    for i_batch, sample in enumerate(dataLoader):
    #for i_batch in range(1000):
        print('training phase')    
        input,target=sample['tr'].type(torch.FloatTensor), sample['gt'].type(torch.FloatTensor)
        #data_loaded=dataSet.__getitem__(1)
        #input,target=data_loaded['tr'].type(torch.FloatTensor), data_loaded['gt'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            input, target=input.unsqueeze(1).cuda(), target.unsqueeze(1).cuda()
        else:
            input, target=input.unsqueeze(1), target.unsqueeze(1)
        input, target=Variable(input), Variable(target)
        #feed data into the net
        optimizer.zero_grad()
        #torch.cuda.empty_cache()
        print('put the data into net')
        output=net(input)
        #define loss function
        loss_MSE = criterion(output[:, :, 20:-20, 5:-5, 5:-5], target[:, :, 20:-20, 5:-5, 5:-5])
        loss_TV = TotalVariation3d(output)
        loss = loss_MSE + lambda_TV*loss_TV
        print(type(loss))
        print('back propagate')
        loss.backward()
        optimizer.step()
        #print(torch.cuda.max_memory_allocated())
        #torch.cuda.empty_cache()
        #print(torch.cuda.max_memory_allocated())
        print('iter %d,  training loss: mse %.4f, tv %.4f' %(i_batch, loss_MSE.data[0], loss_TV.data[0]))
        time_total=time_total+int(time.time())-t
        print('training time %d'%time_total)
        #break
        if i_batch%10==0:
            #test
            test_MSE=0
            test_TV=0
            t_i=10
            print('test phase')
            for t_batch, t_sample in enumerate(testLoader):
                print(t_batch)
                t_input, t_target=t_sample['tr'].type(torch.FloatTensor), t_sample['gt'].type(torch.FloatTensor)
                if torch.cuda.is_available():
                    t_input, t_target=t_input.unsqueeze(1).cuda(), t_target.unsqueeze(1).cuda()
                else:
                    t_input, t_target=t_input.unsqueeze(1), t_target.unsqueeze(1)
                t_input, t_target=Variable(t_input, volatile=True), Variable(t_target, volatile=True)
                print('input image shape: ', t_input.data.shape)
                #optimizer.zero_grad()
                t_output=net(t_input)
                t_loss_MSE=criterion(t_output[:,:,20:-20,5:-5,5:-5], t_target[:,:,20:-20,5:-5,5:-5])
                t_loss_TV =TotalVariation3d(t_output)
                test_MSE=test_MSE+t_loss_MSE.data[0]
                test_TV =test_TV +t_loss_TV.data[0]
                if t_i==t_batch:
                    break
                #print(torch.cuda.max_memory_allocated())
                #torch.cuda.empty_cache()
                #print(torch.cuda.max_memory_allocated())
            test_MSE=test_MSE/(t_i+1)
            test_TV =test_TV/(t_i+1)
            print('iter %d,  testing loss mse: %.4f, testing loss tv: %.4f' %(i_batch, test_MSE, test_TV))
            #save fake test data
            '''
            id_test=random.randint(0,99)
            testsample=testSet.__getitem__(id_test)
            test_input, test_target=Variable(testsample['tr'].unsqueeze(0).unsqueeze(1).type(torch.FloatTensor).cuda(), volatile=True), Variable(testsample['gt'].unsqueeze(0).unsqueeze(1).type(torch.FloatTensor).cuda(), volatile=True)
            test_output=net(test_input)
            #optimizer.zero_grad()
            save_gt_path='/gdata/zhoutk/Deconv/gtresult/gt%d'%int(time.time())
            save_result_path='/gdata/zhoutk/Deconv/testresult/test%d'%int(time.time())
            save_blur_path='/gdata/zhoutk/Deconv/blurresult/blur%d'%int(time.time())
            np.save(save_gt_path,test_target.squeeze().cpu().data.numpy())
            np.save(save_result_path,test_output.squeeze().cpu().data.numpy())
            np.save(save_blur_path,test_input.squeeze().cpu().data.numpy())
            print('save test sample id %d blur to %s, result to path %s gt to %s '%(id_test, save_blur_path, save_result_path, save_gt_path))

            #save real test data
            id_test_real=random.randint(0,29999)
            print('test real data: ', id_test_real)
            test_real=VolumnLoad(id_test_real, realdata_path)
            test_real=(test_real/np.max(test_real)-0.5)*2.0
            t_input=Variable(torch.from_numpy(test_real.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1).cuda(), volatile=True)
            t_output=net(t_input)
            real_output_path='/gdata/zhoutk/Deconv/faketestreal/output%d'%int(time.time())
            real_input_path='/gdata/zhoutk/Deconv/faketestreal/input%d'%int(time.time())
            np.save(real_output_path, t_output.squeeze().cpu().data.numpy())
            np.save(real_input_path, t_input.squeeze().cpu().data.numpy())
            print('save real sample id %d to %s, result to %s'%(id_test_real, real_input_path, real_output_path))                      
  
            #save the model
            #torch.save(net.state_dict(),save_path)
            #print(torch.cuda.max_memory_allocated())
            #torch.cuda.empty_cache()
            #print(torch.cuda.max_memory_allocated())
            '''
        t=int(time.time())

print('finish training')
