
import torch
import torch.nn as nn
import pandas as pd
from torchattacks import PGD
from UDA.CP.defenses.trainers.adv_trainer import AdvTrainer

class PGD_trainer(AdvTrainer):
    def __init__(self,model,eps,alpha,steps,**kwargs):
        super().__init__('PGD',model,**kwargs)
        self.record_keys_step.extend(['far_norm2','near_norm2'])
        self.step_recorder=pd.DataFrame(
            columns=self.record_keys_step)
        self.model=model
        self.ce=nn.CrossEntropyLoss()
        self.atk=PGD(
            self.model, eps=eps, alpha=alpha, 
            steps=steps,device='cuda:0')
        self.device='cuda:0'
    def _do_iter(self,train_data):
        images,label=train_data
        images=images.to(self.device)
        label=label.to(self.device)
        batch_size=images.shape[0]
        adv_images=self.atk(images,label)
        #ori_logits=self.model(images)
        adv_logits=self.model(adv_images)
        adv_ce=self.ce(adv_logits,label)
        loss=adv_ce

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        delta, _ = (
            adv_images-images)\
        .abs().view(batch_size,-1).max(dim=1)
        normi=delta.mean().item()
        norm2=(
            (
                (adv_images-images)\
                .view(batch_size,-1)**2)\
            .sum(dim=-1)**0.5)
        
        return dict(
            loss=loss.item(),
            adv_ce=adv_ce.item(),
            adv_norm2=norm2.mean().item(),
            adv_normi=normi,
            near_norm2=norm2[:batch_size].mean().item(),
            far_norm2=norm2[-batch_size:].mean().item())