# -*- coding: utf-8 -*-

import numpy as np


def Growth(A,Tree,Parent,W_S,W_T,V,Origin):
    #print('Growth')
    h,w=Tree.shape
    while A!=[]:
        p=A[0]
        #print ('current active:',p)
        if p=='S':
            ind=np.where(W_S>0)
            for i in range(ind[0].shape[0]):
                y=ind[0][i];x=ind[1][i]
                if Tree[y,x]==0:
                    Tree[y,x]=1;Origin[y,x]=1
                    Parent[y][x]='S'
                    if [y,x] not in A: A.append([y,x])
                    if Tree[y,x]==0: print('1')
                elif Tree[y,x]!=1:
                    path1=['S',[y,x]]                    
                    a=y;b=x;path2=[]
                    while Parent[a][b]!='T':
                        [a,b]=Parent[a][b]
                        path2.append([a,b])                    
                    return path1+path2+['T'],A,Tree,Parent,Origin
                    
        elif p=='T':
            ind=np.where(W_T>0)
            for i in range(ind[0].shape[0]):
                y=ind[0][i];x=ind[1][i]
                if Tree[y,x]==0 : # if free node, grow tree
                    Tree[y,x]=-1;Origin[y,x]=1
                    Parent[y][x]='T'
                    if [y,x] not in A: A.append([y,x])
                    if Tree[y,x]==0: print('2')
                elif Tree[y,x]!=-1: #meet another Tree
                    path2=[[y,x],'T']
                    a=y;b=x;path1=[]
                    while Parent[a][b]!='S':
                        [a,b]=Parent[a][b]
                        path1.append([a,b])
                    return ['S']+path1[::-1]+path2,A,Tree,Parent,Origin
        else: 
             if Tree[p[0],p[1]]==0:
                 print ('active[',p[0],',',p[1],']')
             for c,d in enumerate([(0,1),(0,-1),(1,0),(-1,0)]): #in the order of R,L,D,U
                 c_inv=[(0,1),(0,-1),(1,0),(-1,0)].index((-d[0],-d[1]))
                 if V[p[0],p[1],c]<=0:
                     continue
                 if -1<p[0]+d[0]<h and -1<p[1]+d[1]<w:
                     if Tree[p[0]+d[0],p[1]+d[1]]==0 and ([p[0]+d[0],p[1]+d[1]] not in A):
                         Tree[p[0]+d[0],p[1]+d[1]]=Tree[p[0],p[1]];Origin[p[0]+d[0],p[1]+d[1]]=1
                         Parent[p[0]+d[0]][p[1]+d[1]]=[p[0],p[1]]
                         if [p[0]+d[0],p[1]+d[1]] not in A: A.append([p[0]+d[0],p[1]+d[1]])
                         if Tree[p[0]+d[0],p[1]+d[1]]==0: 
                             print('3')
                             print(p)
                     if Tree[p[0]+d[0],p[1]+d[1]]!=0 and Tree[p[0]+d[0],p[1]+d[1]]!=Tree[p[0],p[1]]:
                        if Tree[p[0],p[1]]==-1 and V[p[0]+d[0],p[1]+d[1],c_inv]<=0: #####changed here
                            continue
                        a=p[0];b=p[1];path1=[[a,b]]
                        while type(Parent[a][b])!=str:
                            [a,b]=Parent[a][b]
                            path1.append([a,b])
                        a=p[0]+d[0];b=p[1]+d[1];path2=[[a,b]]
                        while type(Parent[a][b])!=str:
                            [a,b]=Parent[a][b]
                            path2.append([a,b])
                        if Tree[p[0],p[1]]==1:
                            path=['S']+path1[::-1]+path2+['T']
                        else:
                            path=['S']+path2[::-1]+path1+['T']
                        return path,A,Tree,Parent,Origin
                            
                        
        del(A[0])
    return [],A,Tree,Parent,Origin

def Augmentation(Path,Tree,Parent,W_S,W_T,W_S_inv,W_T_inv,V,O,Origin):
    #print('Aug')
    
    N=len(Path)
    #1. find bottleneck capacity of Path 
    path_cap=np.zeros((N-1))
    path_cap[0]=W_S[Path[1][0],Path[1][1]]
    path_cap[-1]=W_T[Path[-2][0],Path[-2][1]]
    for i in range(N-3):
        y0=Path[i+1][0];x0=Path[i+1][1];y1=Path[i+2][0];x1=Path[i+2][1]
        k=[(0,1),(0,-1),(1,0),(-1,0)].index((y1-y0,x1-x0))
        path_cap[i+1]=V[y0,x0,k]
    neck=np.min(path_cap)
    #print ('neck=',neck)
    #2. Update residual graph
    W_S[Path[1][0],Path[1][1]]-=neck
    W_S_inv[Path[1][0],Path[1][1]]+=neck
    W_T[Path[-2][0],Path[-2][1]]-=neck
    W_T_inv[Path[-2][0],Path[-2][1]]+=neck
    for i in range(N-3):
        y0=Path[i+1][0];x0=Path[i+1][1];y1=Path[i+2][0];x1=Path[i+2][1]
        k=[(0,1),(0,-1),(1,0),(-1,0)].index((y1-y0,x1-x0))
        k_inv=[(0,1),(0,-1),(1,0),(-1,0)].index((y0-y1,x0-x1))
        V[y0,x0,k]-=neck
        V[y1,x1,k_inv]+=neck
    #3.cut the saturated edges
    ind=np.where(path_cap==neck)
    for i in ind[0]:
        if i==0:
            if Tree[Path[i+1][0],Path[i+1][1]]==1:
                Parent[Path[i+1][0]][Path[i+1][1]]=0
                O.append([Path[i+1][0],Path[i+1][1]])
        elif i==N-2:
            if Tree[Path[-2][0],Path[-2][1]]==-1:
                Parent[Path[-2][0]][Path[-2][1]]=0
                O.append([Path[-2][0],Path[-2][1]])                
        else:
            if Tree[Path[i+1][0],Path[i+1][1]]==Tree[Path[i][0],Path[i][1]]:
                if Tree[Path[i+1][0],Path[i+1][1]]==1:
                    Parent[Path[i+1][0]][Path[i+1][1]]=0
                    O.append([Path[i+1][0],Path[i+1][1]])
                if Tree[Path[i+1][0],Path[i+1][1]]==-1:
                    Parent[Path[i][0]][Path[i][1]]=0
                    O.append([Path[i][0],Path[i][1]])

    return Parent,W_S,W_S_inv,W_T,W_T_inv,V,O,Origin

def find_origin(y,x,Parent):
    while Parent[y][x]!=0 and type(Parent[y][x])!=str:
        y,x=Parent[y][x]
    if Parent[y][x]==0: #means from orphan 
        return False
    else: 
        return True
    
    
def Adoption(O,A,Tree,W_S,W_T,Parent,V):
    #print('Adoption')
    while O!=[]:
        [y0,x0]=O[0];del(O[0])
        if Tree[y0,x0]==1 and W_S[y0,x0]>0:
            Parent[y0][x0]='S'
        elif Tree[y0,x0]==-1 and W_T[y0,x0]>0:
            Parent[y0][x0]='T'
        else:
            for c,d in enumerate([(0,1),(0,-1),(1,0),(-1,0)]):
                y1=y0-d[0];x1=x0-d[1] #L,R,U,D
                if find_origin(y1,x1,Parent) and V[y1,x1,c]>0 and Tree[y1,x1]==Tree[y0,x0]:
                    Parent[y0][x0]=[y1,x1]
                    break
        if Parent[y0][x0]==0: #cannot find a valid parent
            for c,d in enumerate([(0,1),(0,-1),(1,0),(-1,0)]):
                y1=y0-d[0];x1=x0-d[1]
                if Tree[y0,x0]!=Tree[y1,x1] or Tree[y1,x1]==0: #changed here, free nodes shouldn't be added to A
                    continue
                else:
                    if V[y1,x1,c]>0:
                        if [y1,x1] not in A: A.append([y1,x1])
                        if Tree[y1,x1]==0: print('4')
                    if Parent[y1][x1]==[y0,x0]:
                        Parent[y1][x1]=0
                        O.append([y1,x1])
            Tree[y0,x0]=0
            #print('change [',y0,',',x0,']')
            if [y0,x0] in A:A.remove([y0,x0]) #??????????????????
    
    return A,Tree,Parent
        
        
                    
                
                
                      
        
        
        
        
        



















