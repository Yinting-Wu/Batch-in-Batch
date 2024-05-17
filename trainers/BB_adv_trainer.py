
import torch
import torch.nn as nn
import pandas as pd
from UDA.CP.defenses.attacks.BB import BB
from UDA.CP.defenses.trainers.adv_trainer import AdvTrainer
def do_SWA_once(self):
    if (self.swa_model is not None) and (self.epochi>=self.swa_init):
        moving_average(
            net1=self.swa_model,net2=self.model,alpha=1/(self.swa_done+1))
        self.swa_done+=1
def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(
        net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
def sep_train_data(self,train_data):
    images,label=train_data
    images=images.to(self.device)
    label=label.to(self.device)
    return images,label
def epoch_errand(self,images):
    if self.passed_steps%self.one_ep_steps==0:
        self.epochi+=1
        self.atk.update_nadv(self.epochi)
        tem_shape=list(images.shape)
        tem_shape[0]=tem_shape[0]*int(self.atk.base_n_adv)
        self.atk.get_LHD_noise(tem_shape,dtype=images.dtype)
        do_SWA_once(self)
    self.passed_steps+=1
def loss_and_back(self,images,label):
    adv_images=self.atk(images,label,self.epochi)
    #ori_logits=self.model(images)
    if adv_images==[]:
        return torch.tensor(0,device='cuda:0'),torch.tensor(0,device='cuda:0')
    adv_logits=self.model(adv_images)
    #ori_ce=self.ce(ori_logits,label)
    #repeat_n=adv_logits.shape[0]//batch_size
    #adv_ce=self.ce(adv_logits,label.repeat(repeat_n))
    #adv_ce=self.ce(self.atk.logits,self.atk.labels)
    adv_ce=self.ce(adv_logits,self.atk.labels)
    #loss=1/self.atk.n_adv*ori_ce+adv_ce
    loss=adv_ce

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss,adv_ce
def init_setting(self):
    self.step_recorder=pd.DataFrame(
        columns=self.record_keys_step)
    self.ce=nn.CrossEntropyLoss()
    self.epochi=0
    self.device='cuda:0'
    self.passed_steps=0
    self.swa_done=0
class BB_trainer(AdvTrainer):
    def __init__(self,model,BB_para,swa_model,clean_warmup,swa_init,one_ep_steps,**kwargs):
        super().__init__('BB',model,**kwargs)
        self.model=model
        self.atk=BB(model,BB_para,clean_warmup=clean_warmup,device='cuda:0',init_type=BB_para['init_type'])
        self.one_ep_steps=one_ep_steps
        self.swa_init=swa_init
        self.swa_model=swa_model
        self.clean_warmup=clean_warmup
        init_setting(self)
    def _do_iter(self,train_data):
        images,label=sep_train_data(self,train_data)
        epoch_errand(self,images)
        loss,adv_ce=loss_and_back(self,images,label)
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
        '''
        return dict(
            loss=loss.item(),
            adv_ce=adv_ce.item(),
            adv_norm2=norm2.mean().item(),
            adv_normi=normi,
            near_norm2=norm2[:batch_size].mean().item(),
            far_norm2=norm2[-batch_size:].mean().item())
        '''
        
    def save_all(self, save_path, overwrite=True):
        self._check_path(save_path+".pth", overwrite=overwrite, file=True)
        self._check_path(save_path+".csv", overwrite=overwrite, file=True)
        print("Saving Model")
        if (self.swa_model is not None) and (self.epochi>=self.swa_init):
            torch.save(self.swa_model.cpu().state_dict(), save_path+".pth")
            print("...Saved as pth to %s !"%(save_path+".pth"))
            print("Saving Records")
            self.rm.to_csv(save_path+".csv")
            self.step_recorder.to_csv(save_path+"，steps.csv")
            self.swa_model.to(self.device)
        else:
            torch.save(self.model.cpu().state_dict(), save_path+".pth")
            print("...Saved as pth to %s !"%(save_path+".pth"))
            print("Saving Records")
            self.rm.to_csv(save_path+".csv")
            self.step_recorder.to_csv(save_path+"，steps.csv")
            self.model.to(self.device)