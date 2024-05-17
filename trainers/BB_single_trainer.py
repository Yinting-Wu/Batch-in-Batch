
import torch
import torch.nn as nn
from .adv_trainer import AdvTrainer
from UDA.CP.defenses.attacks.BB_single import BB_single

class BB_single_trainer(AdvTrainer):
    def __init__(self,model,swa_model,clean_warmup,swa_init=1e7,BB_para=None,**kwargs):
        super().__init__('BB',model,**kwargs)
        self.model=model
        self.atk=BB_single(
            model,BB_para,self.device,epoch=kwargs['epoch'],
            init_type=BB_para['init_type'],clean_warmup=clean_warmup)
        self.one_ep_steps=kwargs['one_ep_steps']
        self.epoch=kwargs['epoch']
        self.swa_init=swa_init
        self.swa_model=swa_model
        init_setting(self)
    def _do_iter(self,train_data):
        images,label=sep_train_data(self,train_data)
        epoch_errand(self,images)
        adv_images=self.atk(images,label,self.epochi)
        if type(adv_images)==list:
            return dict(loss=0,adv_ce=0,adv_norm2=0,adv_normi=0,near_norm2=0,far_norm2=0)
        loss,adv_ce=loss_and_back(self,adv_images)
        return dict(
            loss=loss.item(),
            adv_ce=adv_ce.item(),
            adv_norm2=0,
            adv_normi=0,
            near_norm2=0,
            far_norm2=0)
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
def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(
        net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
def do_SWA_once(self):
    if (self.swa_model is not None) and (self.epochi>=self.swa_init):
        moving_average(
            net1=self.swa_model,net2=self.model,alpha=1/(self.swa_done+1))
        self.swa_done+=1
def init_setting(self):
    self.ce=nn.CrossEntropyLoss()
    self.device='cuda:0'
    self.passed_steps=0
    self.epochi=0
    self.swa_done=0
def sep_train_data(self,train_data):
    images,label=train_data
    images=images.to(self.device)
    label=label.to(self.device)
    return images,label
def epoch_errand(self,images):
    if self.passed_steps%self.one_ep_steps==0:
        tem_shape=list(images.shape)
        tem_shape[0]=tem_shape[0]*self.atk.n_adv
        self.atk.get_LHD_noise(tem_shape,dtype=images.dtype)
        #self.atk.get_alpha(epochi=self.epochi,dtype=images.dtype,shape=tem_shape)
        self.epochi+=1
        do_SWA_once(self)
    self.passed_steps+=1
def loss_and_back(self,adv_images):
    if adv_images==[]:
        return torch.tensor(0,device='cuda:0'),torch.tensor(0,device='cuda:0')
    adv_ce=self.ce(self.model(adv_images),self.atk.labels)
    #adv_ce=self.ce(self.atk.logits,self.atk.labels)
    loss=adv_ce
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss,adv_ce