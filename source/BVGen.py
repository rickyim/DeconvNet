import numpy as np 
from random import *
import math
import scipy.misc
from PSFConv import *
import sys


def singleStepMove(point, stepSize, theta, phi, radius, num_iter, volumn, pointlist, radiuslist, volumnlist): 
    #print(volumn.shape)
    #print('iteration:'+str(num_iter))
    #print('points: '+str(point)+' stepSize: '+str(stepSize)+' theta: '+str(theta)+' phi: '+str(phi)+' radius: ' +str(radius))
    #move in one direction and fill the volumn with a ball
    sz=list(volumn.shape)
    trythres=100 #threshold for num of trials
    ct=0
    #theta:latitude phi: longitude
    #find the next point
    while True:
        ct+=1
        #print('trial: '+str(ct))
        if ct>trythres:
            break
        theta1=(random()*math.pi+theta)%math.pi
        phi1=(2*random()*math.pi+phi)%(2*math.pi)
        direction=[math.cos(theta1)*math.cos(phi1), math.cos(theta1)*math.sin(phi1), math.sin(theta1)]
        x_new=int(point[0]+direction[0]*stepSize)
        y_new=int(point[1]+direction[1]*stepSize)
        z_new=int(point[2]+direction[2]*stepSize)
        if not ((-1<x_new<sz[0]) and (-1<y_new<sz[1]) and (-1<z_new<sz[2])): # out of range
            continue
        flag=False #distance to all other points larger than disthres
        #print('length of pointlist: '+str(len(pointlist)))
        for i in range(len(pointlist)):
            dist=math.sqrt((x_new-pointlist[i][0])**2+(y_new-pointlist[i][1])**2+(z_new-pointlist[i][2])**2)
            disthres=radiuslist[i]+radius
            #print('distance: '+str(dist))
            flag=(dist>disthres)
            if flag is False:
                break
        if flag is True:
            break
    # ct> threshold, none valid point found
    if ct>trythres:
        #print('at iteration '+str(num_iter)+' no valid point found, return')
        return
    else:
        point_new=[x_new,y_new,z_new]

    #when the point is in the volumn, treat the inner side of the ball as blood vessel
    if (-1<point[0]<sz[0]) and (-1<point[1]<sz[1]) and (-1<point[2]<sz[2]):
        list_i=[point[0]-radius+i for i in range(2*radius)]
        list_j=[point[1]-radius+j for j in range(2*radius)]
        list_k=[point[2]-radius+k for k in range(2*radius)]
        for i in list_i:
            for j in list_j:
                for k in list_k:
                    dist=math.sqrt((i-point[0])**2+(j-point[1])**2+(k-point[2])**2)
                    if dist<radius:
                        for r in range(stepSize):
                            x_1=int(i+direction[0]*r)
                            y_1=int(j+direction[1]*r)
                            z_1=int(k+direction[2]*r)
                            if (-1<x_1<sz[0]) and (-1<y_1<sz[1]) and (-1<z_1<sz[2]):
                                volumn[x_1, y_1, z_1]=True
                                volumnlist.append([x_1, y_1, z_1])
        pointlist.append(point_new)
        radiuslist.append(radius)
        if(num_iter>5):
            return
        #update stepSize, radius
        stepSize=stepSize-2
        radius=radius-2
        if stepSize<5 or radius<5:
            return
        singleStepMove(point_new, stepSize, theta1, phi1, radius, num_iter+1, volumn, pointlist, radiuslist, volumnlist)
        singleStepMove(point_new, stepSize, theta1, phi1, radius, num_iter+1, volumn, pointlist, radiuslist, volumnlist)
        return 
    else:
        sys.stdout.write('invalid point')
        return 


def hullExtract(volumn, volumnlist):
    h,w,z=volumn.shape
    hull=np.zeros((h,w,z),dtype=bool)
    for [i,j,k] in volumnlist:
        if volumn[i,j,k]==False:
            continue
        else:
            edge=False
            for [m,n,p] in [[i-1,j,k],[i+1,j,k],[i,j-1,k],[i,j+1,k],[i,j,k-1],[i,j,k+1]]:
                if not((-1<m<h) and (-1<n<w) and (-1<p<z)):
                    edge=True
                    break
                elif volumn[m,n,p]==False:
                    edge=True
                    break
            if edge==True:
                hull[i,j,k]=True
    return hull
                
def newHull(x, y, z, stepSize=80, radius=10): 
    x_st=randint(0,x)
    y_st=randint(0,y)
    z_st=randint(0,z)
    theta=random()*math.pi
    phi=random()*(2*math.pi)
    volumn=np.zeros((x, y, z), dtype=bool)
    pointlist=[[x_st, y_st, z_st]]
    radiuslist=[radius]
    volumnlist=[]
    singleStepMove([x_st, y_st, z_st], stepSize, theta, phi, radius, 1, volumn, pointlist, radiuslist, volumnlist)
    sys.stdout.write("done")
    hull=hullExtract(volumn, volumnlist)
    h,w,z=hull.shape
    return hull

if __name__ == '__main__':
    x=200
    y=200
    z=200
    hull=newHull(x, y, z).astype('uint8')

    psnr=30
    path='./BornWolf10'
    fileformat='tif'
    filtOutput=convolvePsf3D(hull, path, fileformat, psnr)
    h,w,z=filtOutput.shape
    for i in range(z):
        scipy.misc.imsave('./filteredoutput/file'+str(i)+'.png', np.squeeze(filtOutput[:,:,i]))
    print("save done")

