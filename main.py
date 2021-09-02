# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from fn import Growth,Augmentation,Adoption


I=sitk.ReadImage('dog2.jpg')
I=sitk.GetArrayFromImage(I)
I=I[::2,::2,:]
y0=12;y1=82;x0=27;x1=119
N_GMM=5
h,w,d=I.shape
##====================initialziation========================================
ALPHA=np.zeros((h,w))
GAMMA=1
ALPHA[y0:y1,x0:x1]=1
#trimap T
T=np.zeros((h,w))
T[y0:y1,x0:x1]=0.5 #0=bkg, 0.5=undefined, 1=foreground
# GMM class membership

K=np.zeros((h,w))
K=np.copy(k_fix)
#K[:,:]=np.random.randint(1,N_GMM+1,size=(h,w)) #in {1,2,3,4,5}
K[np.where(ALPHA==0)]=K[np.where(ALPHA==0)]*-1 #bkg {-1,-2,-3,-4,-5}
#K-means
#Foreground
abandon=[]

for iter in range(50):
    #print ('iter=',iter)
    GMM_param_obj=[]
    for i in range(N_GMM):   #calculating GMM parameters
        ind=np.where(K==i+1)
        N=ind[0].shape[0]
        if N==0 or N==1:
            abandon.append(i+1)
            GMM_param_obj.append('None')
            continue
        r=np.sum(I[ind[0],ind[1],0])
        g=np.sum(I[ind[0],ind[1],1])
        b=np.sum(I[ind[0],ind[1],2])
        mean=np.array((r/N,g/N,b/N))
        cov=np.cov(I[ind[0],ind[1],:].T)
        pi=N/(np.where(ALPHA==1)[0].shape[0])
        GMM_param_obj.append([pi,mean,cov])
    
    ind=np.where(ALPHA==1)
    K_temp=np.zeros((h,w,N_GMM))
    for i in range(N_GMM): #assign memberships    
        if i+1 in abandon:
            K_temp[ind[0],ind[1],i]=-999999
            continue
        pi=GMM_param_obj[i][0]
        mean=GMM_param_obj[i][1] 
        cov=GMM_param_obj[i][2]
        z=I[ind[0],ind[1],:] #(n,3)
        #K_temp[ind[0],ind[1],i]=np.log(pi)-1/2*np.log(np.linalg.det(cov))-1/2*np.matmul(np.matmul((z-mean),np.linalg.inv(cov)),(z-mean).T).diagonal()
        K_temp[ind[0],ind[1],i]=np.log(pi)-1/2*np.log(np.linalg.det(cov))-1/2*np.sum(np.matmul((z-mean),np.linalg.inv(cov))*(z-mean),axis=-1)
    K[ind[0],ind[1]]=np.argmax(K_temp[ind[0],ind[1],:],axis=-1)+1           
#Background
for iter in range(50):
    #print ('iter=',iter)
    GMM_param_bkg=[]
    for i in range(N_GMM):   #calculating GMM parameters
        ind=np.where(K==-i-1)
        N=ind[0].shape[0]
        if N==0 or N==1:
            abandon.append(-i-1)
            GMM_param_bkg.append('None')
            continue
        r=np.sum(I[ind[0],ind[1],0])
        g=np.sum(I[ind[0],ind[1],1])
        b=np.sum(I[ind[0],ind[1],2])
        mean=np.array((r/N,g/N,b/N))
        cov=np.cov(I[ind[0],ind[1],:].T)
        pi=N/(np.where(ALPHA==0)[0].shape[0])
        GMM_param_bkg.append([pi,mean,cov])
    
    ind=np.where(ALPHA==0)
    K_temp=np.zeros((h,w,N_GMM))
    for i in range(N_GMM): #assign memberships    
        if -i-1 in abandon:
            K_temp[ind[0],ind[1],i]=-999999
            continue
        pi=GMM_param_bkg[i][0]
        mean=GMM_param_bkg[i][1] 
        cov=GMM_param_bkg[i][2]
        z=I[ind[0],ind[1],:] #(n,3)
        K_temp[ind[0],ind[1],i]=np.log(pi)-1/2*np.log(np.linalg.det(cov))-1/2*np.sum(np.matmul((z-mean),np.linalg.inv(cov))*(z-mean),axis=-1)
    K[ind[0],ind[1]]=-np.argmax(K_temp[ind[0],ind[1],:],axis=-1)-1        
# compute part of n-link weights
V=np.zeros((h,w,4))
I_r=np.pad(I[:,1:,:],((0,0),(0,1),(0,0)),'edge')    
V[:,:,0]=np.sum((I-I_r)^2,axis=-1)
I_l=np.pad(I[:,:w-1,:],((0,0),(1,0),(0,0)),'edge')    
V[:,:,1]=np.sum((I-I_l)^2,axis=-1)     
I_d=np.pad(I[1:,:,:],((0,1),(0,0),(0,0)),'edge')    
V[:,:,2]=np.sum((I-I_d)^2,axis=-1)    
I_u=np.pad(I[:h-1,:,:],((1,0),(0,0),(0,0)),'edge')    
V[:,:,3]=np.sum((I-I_u)^2,axis=-1)
Beta=(2*w*h-h-w)/np.sum(V)  


#ALPHA=np.copy(ALPHA_fix)
#GMM_param_obj=GMM_obj_fix
#GMM_param_bkg=GMM_bkg_fix
#T[12:18,99:119]=0 
#ALPHA[12:18,99:119]=0
#T[27:30,108:112]=1
#ALPHA[27:30,108:112]=1

epoch=3
SEG=np.zeros((h,w,epoch))
#==============================iterative minimisation=======================

for iter1 in range(epoch):
#step1. Assign memberships
    print('step1..')
    ind=np.where(ALPHA==1) # assign all the obj pixels
    K_temp=np.zeros((h,w,N_GMM))
    for i in range(N_GMM):
        if i+1 in abandon:
            K_temp[ind[0],ind[1],i]=-99999
            continue
        pi=GMM_param_obj[i][0]
        mean=GMM_param_obj[i][1] 
        cov=GMM_param_obj[i][2]
        z=I[ind[0],ind[1],:] #(n,3)
        K_temp[ind[0],ind[1],i]=np.log(pi)-1/2*np.log(np.linalg.det(cov))-1/2*np.sum(np.matmul((z-mean),np.linalg.inv(cov))*(z-mean),axis=-1)
    K[ind[0],ind[1]]=np.argmax(K_temp[ind[0],ind[1],:],axis=-1)+1
    
    ind=np.where(ALPHA==0) #assign all the bkg pixels
    K_temp=np.zeros((h,w,N_GMM))
    for i in range(N_GMM):  
        if -i-1 in abandon:
            K_temp[ind[0],ind[1],i]=-99999
            continue    
        pi=GMM_param_bkg[i][0]
        mean=GMM_param_bkg[i][1] 
        cov=GMM_param_bkg[i][2]
        z=I[ind[0],ind[1],:] #(n,3)
        K_temp[ind[0],ind[1],i]=np.log(pi)-1/2*np.log(np.linalg.det(cov))-1/2*np.sum(np.matmul((z-mean),np.linalg.inv(cov))*(z-mean),axis=-1)
    K[ind[0],ind[1]]=-np.argmax(K_temp[ind[0],ind[1],:],axis=-1)-1   

#step2. recalculate GMM_params
    print('step2..')
    GMM_param_obj=[]
    for i in range(N_GMM):   #calculating obj GMM parameters
        ind=np.where(K==i+1)
        N=ind[0].shape[0] 
        if N==0:
            abandon.append(i+1)
            GMM_param_obj.append('placeholder')
            continue
        r=np.sum(I[ind[0],ind[1],0])
        g=np.sum(I[ind[0],ind[1],1])
        b=np.sum(I[ind[0],ind[1],2])
        mean=np.array((r/N,g/N,b/N))
        cov=np.cov(I[ind[0],ind[1],:].T)
        pi=N/(np.where(ALPHA==1)[0].shape[0])
        GMM_param_obj.append([pi,mean,cov])   
    GMM_param_bkg=[]
    for i in range(N_GMM):   #calculating bkg GMM parameters
        ind=np.where(K==-i-1)
        N=ind[0].shape[0]
        if N==0:
            abandon.append(-i-1)
            GMM_param_bkg.append('placeholder')
            continue
        r=np.sum(I[ind[0],ind[1],0])
        g=np.sum(I[ind[0],ind[1],1])
        b=np.sum(I[ind[0],ind[1],2])
        mean=np.array((r/N,g/N,b/N))
        cov=np.cov(I[ind[0],ind[1],:].T)
        pi=N/(np.where(ALPHA==0)[0].shape[0])
        GMM_param_bkg.append([pi,mean,cov])    
    
#step3. Estimate segmentation:
    #compute n-link weights V~(h,w,4) (neighbour 4-way connectivity)
    print('step3..')
    a_r=np.pad(ALPHA[:,1:],((0,0),(0,1)),'edge')
    a_l=np.pad(ALPHA[:,:w-1],((0,0),(1,0)),'edge')
    a_d=np.pad(ALPHA[1:,:],((0,1),(0,0)),'edge')
    a_u=np.pad(ALPHA[:h-1,:],((1,0),(0,0)),'edge')

    V[:,:,0]=np.exp(-Beta*V[:,:,0])*(ALPHA!=a_r)
    V[:,:,1]=np.exp(-Beta*V[:,:,1])*(ALPHA!=a_l)
    V[:,:,2]=np.exp(-Beta*V[:,:,2])*(ALPHA!=a_d)
    V[:,:,3]=np.exp(-Beta*V[:,:,3])*(ALPHA!=a_u)
    V=V*GAMMA
    C=1+np.max(np.sum(V,axis=-1)) # the K in paper, used to assign t-links
    #compute t-link weights
    W_S=np.zeros((h,w))
    ind_un=np.where(T==0.5)
    for i in range(N_GMM):
        if -i-1 in abandon:
            continue
        pi=GMM_param_bkg[i][0]
        mean=GMM_param_bkg[i][1] 
        cov=GMM_param_bkg[i][2]
        z=I[ind_un[0],ind_un[1],:] #(n,3)
        W_S[ind_un[0],ind_un[1]]=W_S[ind_un[0],ind_un[1]]-np.log(pi)+1/2*np.log(np.linalg.det(cov))+1/2*np.sum(np.matmul((z-mean),np.linalg.inv(cov))*(z-mean),axis=-1)
    W_S[np.where(T==0)]==0
    W_S[np.where(T==1)]==C
    W_S_inv=np.zeros((h,w))
    
    W_T=np.zeros((h,w))
    for i in range(N_GMM):
        if i+1 in abandon:
            continue
        pi=GMM_param_obj[i][0]
        mean=GMM_param_obj[i][1] 
        cov=GMM_param_obj[i][2]
        z=I[ind_un[0],ind_un[1],:] #(n,3)
        W_T[ind_un[0],ind_un[1]]=W_T[ind_un[0],ind_un[1]]-np.log(pi)+1/2*np.log(np.linalg.det(cov))+1/2*np.sum(np.matmul((z-mean),np.linalg.inv(cov))*(z-mean),axis=-1)    
    W_T[np.where(T==0)]=C
    W_T[np.where(T==1)]=0
    W_T_inv=np.zeros((h,w))
    #minimum cut
    A=['S','T']
    O=[]
    Origin=np.zeros((h,w))
    Tree=np.zeros((h,w)) #S=1, T=-1, free=0
    Parent=np.zeros((h,w)).tolist()
    iter2=0
    active=[]
    while True:
        iter2+=1
        #Growth stage:
        Path,A,Tree,Parent,Origin=Growth(A,Tree,Parent,W_S,W_T,V,Origin)
        if Path==[]:
            break
        #Augmentation stage:
        Parent,W_S,W_S_inv,W_T,W_T_inv,V,O,Origin=Augmentation(Path,Tree,Parent,W_S,W_T,W_S_inv,W_T_inv,V,O,Origin)
        #Adoption stage:
        A,Tree,Parent=Adoption(O,A,Tree,W_S,W_T,Parent,V)
    Tree[Tree==-1]=0
    ALPHA=Tree
    SEG[:,:,iter1]=ALPHA

                        
            

    
    
