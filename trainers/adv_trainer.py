import os
import torch

# from torchhk import Trainer
from UDA.CP.defenses.trainers.trainer import Trainer
from torchattacks.attack import Attack
from torchattacks import VANILA, FGSM, PGD, GN
from UDA.CP.defenses.attacks.BB import BB

r"""
Trainer for Adversarial Training.

Attributes:
    self.model : model.
    self.device : device where model is.
    self.optimizer : optimizer.
    self.scheduler : scheduler (* Automatically Updated).
    self.max_epoch : total number of epochs.
    self.max_iter : total number of iterations.
    self.epoch : current epoch.
        * Starts from 0.
        * Automatically updated.
    self.iter : current iter.
        * Starts from 0.
        * Automatically updated.
    self.record_keys : items to record (= items returned by do_iter).
    
Functions:
    self.record_rob : recording robust accuracy against FGSM, PGD, GN.

"""

class AdvTrainer(Trainer):
    def __init__(self, name, model, **kwargs):
        self.record_keys_step=[
            'loss','adv_ce','adv_norm2','adv_normi']
        self.record_keys=[
            'Epoch','Iter','lr','train_time']
        super(AdvTrainer, self).__init__(name, model, **kwargs)
        self._flag_record_rob = False
    
    def record_rob(self, train_set, test_set, eps, alpha, steps, BB_alpha):
        self.activated_adv=[]
        for kk,vv in self.adv_switch.items():
            if vv:
                self.record_keys.append('{}(Tr)'.format(kk))
                self.record_keys.append('{}(Te)'.format(kk))
                self.record_keys.append('{}(Tr_norm2)'.format(kk))
                self.record_keys.append('{}(Te_norm2)'.format(kk))
                self.record_keys.append('{}(Tr_normi)'.format(kk))
                self.record_keys.append('{}(Te_normi)'.format(kk))
                self.activated_adv.append(kk)
        self.record_atks={}
        for adv_name in self.activated_adv:
            match adv_name:
                case 'VANILA':
                    self.record_atks['VANILA']=VANILA(self.model,device='cuda:0')
                case 'FGSM':
                    self.record_atks['FGSM']=FGSM(self.model, eps=eps,device='cuda:0')
                case 'PGD':
                    self.record_atks['PGD']=PGD(
                        self.model, eps=8/255, alpha=2/255, 
                        steps=50,device='cuda:0')
                case 'GN':
                    self.record_atks['GN']=GN(self.model, std=0.1,device='cuda:0')
                case 'BB':
                    self.record_atks['BB']=BB(
                            self.model,
                            BB_para=dict(
                                n_adv=10,
                                iters=50,
                                alpha=2/255,
                                norm_coe=0,
                                gamma=1,
                                eps=1e-4,
                                noise_des=1,
                                perturb_portion=0,
                                preserve_portion=1e-7,
                                noise_edge=8/255),
                            device='cuda:0')
        self.train_set = [ii.to('cuda:0') for ii in train_set]
        self.test_set = [ii.to('cuda:0') for ii in test_set]
        self._flag_record_rob = True
    
    # Update Records
    def _update_record(self, records):
        if self._flag_record_rob:
            adv_eval={}
            for adv_name in self.activated_adv:
                atk=self.record_atks[adv_name]
                train_adv_batch=[
                    (
                        atk(*self.train_set), 
                        self.train_set[1])]
                test_adv_batch=[
                    (
                        atk(*self.test_set), 
                        self.test_set[1])]
                adv_eval['{}(Tr)'.format(adv_name)]=get_acc(
                    self.model,train_adv_batch)
                adv_eval['{}(Te)'.format(adv_name)]=get_acc(
                    self.model,test_adv_batch)
                train_noise=(
                    train_adv_batch[0][0]-self.train_set[0]\
                        .repeat(
                            train_adv_batch[0][0].shape[0]//self.train_set[0].shape[0],
                            1,1,1))\
                .view(
                    (train_adv_batch[0][0].shape[0],-1))\
                .clone().contiguous()
                test_noise=(
                    test_adv_batch[0][0]-self.test_set[0].repeat(
                            test_adv_batch[0][0].shape[0]//self.test_set[0].shape[0],
                            1,1,1))\
                .view(
                    (test_adv_batch[0][0].shape[0],-1))\
                .clone().contiguous()
                adv_eval['{}(Tr_normi)'.format(adv_name)]=\
                train_noise.abs().max(dim=-1)[0].mean().item()
                adv_eval['{}(Te_normi)'.format(adv_name)]=\
                test_noise.abs().max(dim=-1)[0].mean().item()
                adv_eval['{}(Tr_norm2)'.format(adv_name)]=\
                (
                    (
                        train_noise**2)\
                    .sum(dim=-1)**0.5)\
                .mean().item()
                adv_eval['{}(Te_norm2)'.format(adv_name)]=\
                (
                    (
                        test_noise**2)\
                    .sum(dim=-1)**0.5)\
                .mean().item()
            records.update(adv_eval)
        records_list=[]
        for ii in self.record_keys:
            records_list.append(records[ii])
        self.rm.add(records_list)
                         
                
def get_acc(model, test_loader, device='cuda'):
    # Set Cuda or Cpu
    device = torch.device(device)
    model.to(device)
    
    # Set Model to Evaluation Mode
    model.eval()
    
    # Initialize
    correct = 0
    total = 0

    # For all Test Data
    for batch_images, batch_labels in test_loader:

        # Get Batches
        X = batch_images.to(device)
        Y = batch_labels.to(device)
        if Y.shape[0]<X.shape[0]:
            Y=torch.cat([Y for _ in range(X.shape[0]//Y.shape[0])],dim=0)
        
        # Forward
        pre = model(X)

        # Calculate Accuracy
        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

    return (100 * float(correct) / total)
