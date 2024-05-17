
import torch
import torch.nn as nn
from .adv_trainer import AdvTrainer
from UDA.CP.defenses.attacks.BB_single import BB_single
import torch.nn.functional as F

class smooth_single_trainer(AdvTrainer):
    def __init__(
        self,model,BB_para,teacher1,
        teacher2,swa_model,swa_init,ce_coe,t1_coe,t2_coe,**kwargs):
        super().__init__('Smooth',model,**kwargs)
        self.atk=BB_single(model,BB_para,self.device,epoch=kwargs['epoch'],init_type=BB_para['init_type'])
        self.one_ep_steps=kwargs['one_ep_steps']
        self.epoch=kwargs['epoch']
        self.teacher1=teacher1
        self.teacher2=teacher2
        self.swa_model=swa_model
        self.swa_init=swa_init
        self.ce_coe=ce_coe
        self.t1_coe=t1_coe
        self.t2_coe=t2_coe
        self.model=model
        init_setting(self)
    def _do_iter(self,train_data):
        images,label=divide_train_data(self,train_data)
        epoch_errand(self,images)
        adv_images=self.atk(images,label)
        loss,adv_ce=loss_cacu(self,adv_images)
        loss_backward(self,loss)
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
def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss
def init_setting(self):
    self.ce=nn.CrossEntropyLoss()
    self.device='cuda:0'
    self.passed_steps=0
    self.epochi=0
    self.swa_done=0
    self.teacher1.eval()
    self.teacher2.eval()
    self.swa_model.eval()
def divide_train_data(self,train_data):
    images,label=train_data
    images=images.to(self.device)
    label=label.to(self.device)
    return images,label
def do_SWA_once(self):
    if (self.swa_model is not None) and (self.epochi>=self.swa_init):
        moving_average(
            net1=self.swa_model,net2=self.model,alpha=1/(self.swa_done+1))
        self.swa_done+=1
def epoch_errand(self,images):
    if self.passed_steps%self.one_ep_steps==0:
        #self.atk.get_alpha(epochi=self.epochi,dtype=images.dtype,shape=tem_shape)
        self.epochi+=1
        do_SWA_once(self)
    self.passed_steps+=1
def loss_cacu(self,adv_images):
    logits=self.model(adv_images)
    adv_ce=self.ce(logits,self.atk.labels)
    #adv_ce=self.ce(self.atk.logits,self.atk.labels)
    with torch.no_grad():
        t1_logits=self.teacher1(adv_images)
        t2_logits=self.teacher2(adv_images)
    t1_kl=loss_fn_kd(logits, t1_logits, T=2.)
    t2_kl=loss_fn_kd(logits, t2_logits, T=2.)
    loss=adv_ce*self.ce_coe+t1_kl*self.t1_coe+t2_kl*self.t2_coe
    return loss,adv_ce
def loss_backward(self,loss):
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()