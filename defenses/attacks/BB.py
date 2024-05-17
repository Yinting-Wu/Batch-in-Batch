
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from UDA.utils.utils import seed_everything
import torch.nn as nn
import copy
from torchattacks.attack import Attack

def A_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad():
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        selected=[]
        labels=[]
        selected.append(Xadv[correct==False,...])
        labels.append(cat_lab[correct==False,...])
        correct=correct.view(n_adv,-1)
        for ii in range(correct.shape[1]):
            if correct[:,ii].all():
                selected.append(ori_ima[ii,...][None,...])
                labels.append(cat_lab[ii][None])
        Xadv=torch.cat(selected,dim=0).contiguous()
        labels=torch.cat(labels,dim=0)
    return Xadv.detach().contiguous(),labels
def B_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        cat_lab=torch.cat((cat_lab,cat_lab[:Xadv.shape[0]//n_adv,...]),dim=0).contiguous()
        Xadv=torch.cat((Xadv,ori_ima),dim=0).contiguous()
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        Xadv=Xadv[correct==False,...]
        labels=cat_lab[correct==False,...]
    return Xadv.detach().contiguous(),labels
def C_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        Xadv=Xadv[correct==False,...]
        labels=cat_lab[correct==False,...]
    return Xadv.detach().contiguous(),labels
def E_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        Xadv=torch.cat((Xadv,ori_ima),dim=0)
        cat_lab=torch.cat(
            (
                cat_lab,
                cat_lab[:ori_ima.shape[0],...]),dim=0)
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        Xadv=Xadv[correct==False,...]
        labels=cat_lab[correct==False,...]
    return Xadv.detach().contiguous(),labels
def D_type(Xadv,n_adv,ori_ima,model,cat_lab):
    return Xadv.detach().contiguous(),cat_lab
def F_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad():
        Xadv=torch.cat((Xadv,ori_ima),dim=0)
        cat_lab=torch.cat(
            (
                cat_lab,
                cat_lab[:ori_ima.shape[0],...]),dim=0)
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        accept=torch.maximum(pred!=cat_lab,torch.randn(pred.shape,device='cuda:0')<=0.7)
        Xadv=Xadv[accept,...]
        labels=cat_lab[accept,...]
    return Xadv.detach().contiguous(),labels
def G_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad():
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=(pred==cat_lab).reshape(n_adv,-1)
        radii=(Xadv-ori_ima.repeat(n_adv,1,1,1)).reshape((-1,np.cumprod(ori_ima.shape[1:])[-1])).abs().max(dim=1).values.reshape(n_adv,-1)
        class_mask=(torch.sum(correct,dim=0)<n_adv)[None,:]&correct
        radii[class_mask]=-1
        big=torch.argsort(radii,dim=0)[-1,:]
        select=torch.zeros(correct.shape,device=correct.device,dtype=bool)
        for ii in range(len(big)):
            select[big[ii],ii]=True
        select=select.flatten()
        Xadv=Xadv[select,...]
        labels=cat_lab[select,...]
    return Xadv.detach().contiguous(),labels
def get_choose_func(type):
    match type:
        case 'A':
            return A_type
        case 'B':
            return B_type
        case 'C':
            return C_type
        case 'D':
            return D_type
        case 'E':
            return E_type
        case 'F':
            return F_type
        case 'G':
            return G_type
class BB(Attack):
    def __init__(self,model,BB_para,clean_warmup,device,init_type):
        super().__init__('BB',model,device)
        self.max_n_adv=BB_para['n_adv']
        self.iters=BB_para['iters']
        self.alpha=BB_para['alpha']
        self.norm_coe=BB_para['norm_coe']
        self.gamma=BB_para['gamma']
        self.eps=BB_para['eps']
        self.noise_des=BB_para['noise_des']
        self.preserve_portion=BB_para['preserve_portion']
        self.perturb_portion=BB_para['perturb_portion']
        self.noise_edge=BB_para['noise_edge']
        self.model=model
        self.clean_warmup=clean_warmup
        self.choose_type=BB_para['choose_type']
        self.ce=nn.CrossEntropyLoss()
        self.init_type=init_type
        self.base_n_adv=self.max_n_adv
    def get_LHD_noise(self,shape,dtype):
        if self.init_type=='LHD':
            self.LHD_noise=LHD(shape=shape,n_adv=self.max_n_adv,eps=self.noise_edge,device=self.device,dtype=dtype)
        '''
        far_edge=self.noise_edge*1.3
        near_edge=self.noise_edge
        interval=(far_edge-near_edge)/(self.n_adv)
        self.alpha=(
            near_edge+torch.arange(
                1,self.n_adv+1,dtype=dtype,device='cuda:0')*interval)\
        /self.iters
        self.alpha=self.alpha[:,None].repeat(1,shape[0]//self.n_adv)\
        .view(shape[0],*([1]*(len(shape)-1)))
        '''
    def update_nadv(self,epoch):
        #epoch是接下来要处理的epoch是第几个epoch,[1,inf]
        #self.base_n_adv=np.minimum(
        #    1+(self.max_n_adv-1)/(49)*(epoch-1),10)
        self.base_n_adv=self.max_n_adv
    def get_nadv(self):
        if np.random.rand()<=self.base_n_adv%1:
            self.n_adv=int(self.base_n_adv)+1
        else:
            self.n_adv=int(self.base_n_adv)
    def forward(self, ori_ima,label,epochi):
        if epochi<=self.clean_warmup:
            self.labels=label
            return ori_ima
        self.get_nadv()
        if self.n_adv==0:
            self.labels=label
            self.logits_rec=self.model(ori_ima)
            return ori_ima
        x_list=[ori_ima for _ in range(self.n_adv)]
        rep_lab=[label for _ in range(self.n_adv)]
        Xadv=torch.cat(x_list,dim=0)
        Xclean=copy.deepcopy(Xadv.detach())
        Xclean.requires_grad=False
        cat_lab=torch.cat(rep_lab,dim=0)
        if self.init_type=='LHD':
            try:
                Xadv.data+=self.LHD_noise[:Xadv.shape[0],...]
            except:
                self.get_LHD_noise(Xadv.shape,Xadv.dtype)
                Xadv.data+=self.LHD_noise
        if self.init_type=='random':
            Xadv.data+=(torch.rand_like(Xadv.data)*2-1)*self.noise_edge
        Xadv=torch.clamp(Xadv,min=0,max=1)
        for ii in range(self.iters):
            Xadv.requires_grad=True
            logits=self.model(Xadv)
            fir=self.ce(logits,cat_lab)
            '''
            sec=(
                (
                    (
                        (
                            (Xadv-Xclean)\
                            .view(Xadv.shape[0],-1))**2)\
                    .sum(dim=-1))\
                **0.5)\
            .mean()
            with torch.no_grad():
                n_coe=fir/sec*self.norm_coe
            '''
            loss=fir
            grad=torch.autograd.grad(loss,Xadv)[0]
            '''
            if momentum is None:
                momentum=grad
            else:
                momentum=momentum*(1-self.gamma)\
                +self.gamma*(
                    self.alpha*grad+noise*noc)
            Xadv.data.add_(
                momentum/torch.norm(
                    momentum.view(
                        momentum.shape[0],
                        -1),
                    dim=1, p=2)[:,*([None]*(
                            len(momentum.shape)-1))]*self.alpha_vec)
            '''
            #Xadv=Xadv.detach()+torch.sign(grad)*self.alpha_vec
            #Xadv=Xadv.detach()+torch.sign(grad)*(-self.alpha/(self.iters-1)*ii+1.5*self.alpha)
            Xadv=Xadv.detach()+torch.sign(grad)*self.alpha
            Xadv=(Xclean+torch.clamp(
                Xadv-Xclean,
                min=-self.noise_edge,
                max=self.noise_edge)).detach()
            Xadv=torch.clamp(Xadv,min=0,max=1)
        choose_func=get_choose_func(self.choose_type)
        Xadv,self.labels=choose_func(Xadv,self.n_adv,ori_ima,self.model,cat_lab)
        if Xadv.shape[0]==0:
            return []
        else:
            return Xadv

def LHD(shape,n_adv,eps,device,dtype,perturb=False):
    #
    if n_adv==1:
        return (torch.rand(shape,device=device,dtype=dtype)*2-1)*eps
    n_ele=1
    bat_size=shape[0]//n_adv
    for ii in range(1,len(shape)):
        n_ele*=shape[ii]
    len_interval=2*eps/(n_adv+1)#两点取3区间
    left_points=(
        -eps+(
            1+torch.arange(
                n_adv,device=device,dtype=dtype))\
        *len_interval)[:,None]
    if perturb:
        noise=(
            torch.rand(
                (n_adv,n_ele),
                device=device,dtype=dtype)*2-1)\
        *len_interval/2+left_points
    else:
        left_points[left_points==0]=1e-4
        noise=left_points.repeat(1,n_ele)
    aa=np.arange(n_adv)
    for ii in range(noise.shape[1]):
        np.random.shuffle(aa)
        noise[:,ii]=noise[aa,ii]
    noise=noise.repeat(1,bat_size)
    return noise.view(shape).contiguous()