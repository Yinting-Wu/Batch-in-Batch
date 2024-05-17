
import numpy as np
import torch
import torch.nn as nn
from torchattacks.attack import Attack
from torchattacks import FGSM

class Check_Point(Attack):
    def __init__(self,model,cp_num,eps,device='cuda:0'):
        super().__init__('Check_point',model,device)
        self.device=device
        self.cp_num=cp_num
        self.eps=eps
        self.model=model
        self.ce=nn.CrossEntropyLoss(reduction='none')
        self.FGSM=FGSM(
            self.model, eps=eps,device=device)
        assert self.cp_num>1,'one check point is equivalent to FGSM'
    def forward(self, ori_ima,label):
        adv_ima=self.FGSM(ori_ima,label)
        with torch.no_grad():
            noise=adv_ima-ori_ima
            interval=noise/self.cp_num
        del adv_ima
        inter_rec=self.cp_num*np.ones((self.cp_num-1,ori_ima.shape[0]),dtype=np.int64)
        for ii in range(self.cp_num-1):
            interval_num=ii+1
            with torch.no_grad():
                advi=ori_ima+interval*interval_num
                pres=self.model(advi).max(dim=-1).indices
                inter_rec[ii,(pres!=label).cpu()]=interval_num
        best_interval=torch.tensor(
            np.min(inter_rec,axis=0)+1,
            dtype=ori_ima.dtype,device=ori_ima.device)\
        [None,*([1]*(len(ori_ima.shape)-1))]
        return ori_ima+interval*best_interval