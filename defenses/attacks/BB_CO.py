
import numpy as np
import torch
import torch.nn as nn
import copy
from torchattacks.attack import Attack
def C_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        Xadv=Xadv[correct==False,...]
        labels=cat_lab[correct==False,...]
    return Xadv.detach().contiguous(),labels
def smart_cast(
    x_random,noise2,limit,init_al,tolerance,ori_ima):
    p1,p2,p3=0,1/2,1
    al=init_al
    with torch.no_grad():
        noise3=x_random+al*noise2-ori_ima.repeat(x_random.shape[0]//ori_ima.shape[0],1,1,1)
        l_i=noise3.abs().max().item()
        if l_i<=limit:
            return al
        while True:
            al=p2*init_al
            noise3=x_random+al*noise2-ori_ima.repeat(x_random.shape[0]//ori_ima.shape[0],1,1,1)
            l_i=noise3.abs().max().item()
            if limit<=l_i and l_i<=tolerance+limit:
                break
            if l_i>(limit+1/2*tolerance):
                p3=p2
                p2=1/2*(p1+p2)
            else:
                p1=p2
                p2=1/2*(p2+p3)
    return al
def smart_cast_bat(bat_random,bat_noise2,limit,init_al,tolerance,ori_ima):
    als=[]
    ori_idx=0
    for ii in range(bat_random.shape[0]):
        als.append(
            smart_cast(
                x_random=bat_random[ii,...],
                noise2=bat_noise2[ii,...],
                limit=limit,init_al=init_al,tolerance=tolerance,ori_ima=ori_ima[ori_idx,...]))
        ori_idx+=1
        if ori_idx==ori_ima.shape[0]:
            ori_idx=0
    return torch.tensor(als,dtype=bat_random.dtype,device='cuda:0')[:,None,None,None]
class BB_CO(Attack):
    def __init__(self,model,BB_para,device,epoch):
        super().__init__('BB',model,device)
        self.n_adv=BB_para['n_adv']
        self.eps=BB_para['eps']#
        self.noise_edge=BB_para['noise_edge']
        self.preserve_portion=BB_para['preserve_portion']
        self.model=model
        self.ce=nn.CrossEntropyLoss(reduction='none')
        self.epoch=epoch
        self.alpha=self.noise_edge
        #self.init_alpha()
    def get_LHD_noise(self,shape,dtype):
        self.LHD_noise=LHD(
            shape=shape,n_adv=self.n_adv,
            eps=self.noise_edge,device=self.device,dtype=dtype)
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
    def forward(self, ori_ima,label):
        x_list=[ori_ima for _ in range(self.n_adv)]
        rep_lab=[label for _ in range(self.n_adv)]
        Xadv=torch.cat(x_list,dim=0)
        Xclean=copy.deepcopy(Xadv.detach())
        Xclean.requires_grad=False
        cat_lab=torch.cat(rep_lab,dim=0)
        try:
            Xadv.data+=self.LHD_noise
            #Xadv.data+=(torch.rand_like(Xadv.data)*2-1)*self.eps_aug*self.noise_edge
        except:
            self.get_LHD_noise(Xadv.shape,Xadv.dtype)
            Xadv.data+=self.LHD_noise
        Xadv=torch.clamp(Xadv,min=0,max=1)
        Xadv.requires_grad=True
        logits=self.model(Xadv)
        loss=self.ce(logits,cat_lab)
        sign_grad=torch.sign(torch.autograd.grad(loss.mean(),Xadv)[0])
        Xadv=Xadv.detach()+sign_grad*smart_cast_bat(
            bat_random=Xadv.detach(),ori_ima=ori_ima,bat_noise2=sign_grad,limit=self.noise_edge,init_al=self.noise_edge,tolerance=1e-3)
        '''
        Xadv=(
            Xclean+torch.clamp(
            Xadv-Xclean,
            min=-self.noise_edge,
            max=self.noise_edge)).detach()
        '''
        Xadv=torch.clamp(Xadv,min=0,max=1)
        choose_func=C_type
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