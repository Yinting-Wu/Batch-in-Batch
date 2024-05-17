
import torch
import torch.nn as nn
from .adv_trainer import AdvTrainer
from UDA.CP.defenses.attacks.BB_CO import BB_CO

class BB_CO_trainer(AdvTrainer):
    def __init__(self,model,BB_para,**kwargs):
        super().__init__('BB',model,**kwargs)
        self.model=model
        self.ce=nn.CrossEntropyLoss()
        self.device='cuda:0'
        self.atk=BB_CO(model,BB_para,self.device,epoch=kwargs['epoch'])
        self.passed_steps=0
        self.one_ep_steps=kwargs['one_ep_steps']
        self.epochi=0
        self.epoch=kwargs['epoch']
    def _do_iter(self,train_data):
        images,label=train_data
        images=images.to(self.device)
        label=label.to(self.device)
        if self.passed_steps%self.one_ep_steps==0:
            tem_shape=list(images.shape)
            tem_shape[0]=tem_shape[0]*self.atk.n_adv
            self.atk.get_LHD_noise(tem_shape,dtype=images.dtype)
            #self.atk.get_alpha(epochi=self.epochi,dtype=images.dtype,shape=tem_shape)
            self.epochi+=1
        self.passed_steps+=1
        adv_images=self.atk(images,label)
        if type(adv_images)==list:
            return dict(
                loss=0,
                adv_ce=0,
                adv_norm2=0,
                adv_normi=0,
                near_norm2=0,
                far_norm2=0)
        adv_ce=self.ce(self.model(adv_images),self.atk.labels)
        #adv_ce=self.ce(self.atk.logits,self.atk.labels)
        loss=adv_ce

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
        delta, _ = (
            adv_images-images.repeat((repeat_n,*([1]*(len(images.shape)-1)))))\
        .abs().view(batch_size*repeat_n,-1).max(dim=1)
        normi=delta.mean().item()
        norm2=(
            (
                (adv_images-images.repeat((repeat_n,*([1]*(len(images.shape)-1)))))\
                .view(batch_size*repeat_n,-1)**2)\
            .sum(dim=-1)**0.5)
        '''
        
        return dict(
            loss=loss.item(),
            adv_ce=adv_ce.item(),
            adv_norm2=0,
            adv_normi=0,
            near_norm2=0,
            far_norm2=0)