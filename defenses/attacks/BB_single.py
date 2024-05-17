
import numpy as np
import torch
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
def F_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        accept=torch.maximum(
            pred!=cat_lab,
            torch.rand_like(pred)<=self.accept_rate)
        Xadv=Xadv[accept==True,...]
        labels=cat_lab[accept==True,...]
    return Xadv.detach().contiguous(),labels
def D_type(Xadv,n_adv,ori_ima,model,cat_lab):
    return Xadv.detach().contiguous(),cat_lab
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
class BB_single(Attack):
    def __init__(self,model,BB_para,device,epoch,clean_warmup,init_type='LHD'):
        super().__init__('BB',model,device)
        self.n_adv=BB_para['n_adv']
        self.eps=BB_para['eps']
        self.noise_edge=BB_para['noise_edge']
        self.preserve_portion=BB_para['preserve_portion']
        self.eps_aug=BB_para['eps_aug']
        self.choose_type=BB_para['choose_type']
        self.model=model
        self.epoch=epoch
        self.alpha=self.noise_edge
        self.init_type=init_type
        self.clean_warmup=clean_warmup
        self.ce=nn.CrossEntropyLoss(reduction='none')
        self.choose_func=get_choose_func(self.choose_type)
        #self.init_alpha()
    def get_LHD_noise(self,shape,dtype):
        if self.init_type=='LHD':
            self.LHD_noise=LHD(
                shape=shape,n_adv=self.n_adv,
                eps=self.noise_edge*self.eps_aug,device=self.device,dtype=dtype)
    '''
    def init_alpha(self):
        interval=self.noise_edge/self.n_adv
        left_points=np.arange(self.n_adv)*interval
        interval2=interval/self.epoch
        sche=np.empty((self.epoch,self.n_adv))
            # 第[i,j]元素是第i个epoch中所有第j个复制品该用的小区间数
                # [1,epoch]
        aa=np.arange(1,self.epoch+1)
        for ii in range(self.n_adv):
            np.random.shuffle(aa)
            sche[:,ii]=aa
        self.alpha_scheduler=left_points[None,:]+sche*interval2
    def get_alpha(self,epochi,dtype,shape):
        self.alpha=torch.tensor(
            self.alpha_scheduler[epochi,:],dtype=dtype,device='cuda:0')\
        [:,None]\
        .repeat(1,shape[0]//self.n_adv)\
        .view(-1,*([1]*(len(shape)-1)))
    '''
    def forward(self, ori_ima,label,epochi):
        if epochi<=self.clean_warmup:
            self.labels=label
            return ori_ima
        Xadv,cat_lab=dup(self,ori_ima,label)
        Xadv=init_noise(self,Xadv)
        Xadv=grad_attack(self,Xadv,cat_lab)
        Xadv,self.labels=self.choose_func(Xadv,self.n_adv,ori_ima,self.model,cat_lab)
        '''
        Xadv=(
            Xclean+torch.clamp(
            Xadv-Xclean,
            min=-self.noise_edge,
            max=self.noise_edge)).detach()
        '''
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
def dup(self,ori_ima,label):
    x_list=[ori_ima for _ in range(self.n_adv)]
    rep_lab=[label for _ in range(self.n_adv)]
    Xadv=torch.cat(x_list,dim=0)
    cat_lab=torch.cat(rep_lab,dim=0)
    return Xadv,cat_lab
def init_noise(self,Xadv):
    if self.init_type=='LHD':
        try:
            Xadv.data+=self.LHD_noise
        except:
            self.get_LHD_noise(Xadv.shape,Xadv.dtype)
            Xadv.data+=self.LHD_noise
    elif self.init_type=='random':
        Xadv.data+=(torch.rand_like(Xadv.data)*2-1)*self.eps_aug*self.noise_edge
    Xadv=torch.clamp(Xadv,min=0,max=1)
    Xadv.requires_grad=True
    return Xadv
def grad_attack(self,Xadv,cat_lab):
    logits=self.model(Xadv)
    loss=self.ce(logits,cat_lab)
    grad=torch.autograd.grad(loss.mean(),Xadv)[0]
    Xadv=Xadv.detach()+torch.sign(grad)*self.alpha
    Xadv=torch.clamp(Xadv,min=0,max=1)
    return Xadv