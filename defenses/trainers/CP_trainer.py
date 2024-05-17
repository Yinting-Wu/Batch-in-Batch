
import torch
import torch.nn as nn
from .adv_trainer import AdvTrainer
from UDA.CP.defenses.attacks.CP import Check_Point

class CP_trainer(AdvTrainer):
    def __init__(
        self,model,cp_num,eps,device='cuda:0',
        **kwargs):
        super().__init__('BB',model,**kwargs)
        self.model=model
        self.ce=nn.CrossEntropyLoss()
        self.device=device
        self.atk=Check_Point(model,cp_num,eps,device=device)
    def _do_iter(self,train_data):
        images,label=train_data
        images=images.to(self.device)
        label=label.to(self.device)
        adv_images=self.atk(images,label)
        adv_logits=self.model(adv_images)
        adv_ce=self.ce(adv_logits,label)
        loss=adv_ce

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_size=images.shape[0]
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