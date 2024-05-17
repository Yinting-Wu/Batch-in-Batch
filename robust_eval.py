import os
import torch
import copy

# from torchhk import Trainer
from torchattacks import VANILA, FGSM, PGD, GN
from UDA.CP.defenses.attacks.BB import BB
from defenses.loader import base_loader
from defenses.model import get_model
import pandas as pd
import numpy as np

class di:
    base='D:/Wu_Yin_Ting2/UDA/other/cache/CP/'
    para='D:/Wu_Yin_Ting2/UDA/other/para2/'
def add_new_record_column(self,kk):
    self.record_keys.append('{}(Tr)'.format(kk))
    self.record_keys.append('{}(Te)'.format(kk))
    self.record_keys.append('{}(Tr_norm2)'.format(kk))
    self.record_keys.append('{}(Te_norm2)'.format(kk))
    self.record_keys.append('{}(Te_far_norm2)'.format(kk))
    self.record_keys.append('{}(Te_near_norm2)'.format(kk))
    self.record_keys.append('{}(Tr_normi)'.format(kk))
    self.record_keys.append('{}(Te_normi)'.format(kk))
    self.record_keys.append('{}(Te_far_normi)'.format(kk))
    self.record_keys.append('{}(Te_near_normi)'.format(kk))
def update_norm(epoch_rec,adv_name,norm2,normi):
    epoch_rec['{}(Te_norm2)'.format(adv_name)].append(norm2.mean().item())
    epoch_rec['{}(Te_far_norm2)'.format(adv_name)].append(norm2.max().item())
    epoch_rec['{}(Te_near_norm2)'.format(adv_name)].append(norm2.min().item())
    epoch_rec['{}(Te_normi)'.format(adv_name)].append(normi.mean().item())
    epoch_rec['{}(Te_far_normi)'.format(adv_name)].append(normi.max().item())
    epoch_rec['{}(Te_near_normi)'.format(adv_name)].append(normi.min().item())
def add_new_keys(epoch_rec,adv_name):
    epoch_rec['{}(Tr)'.format(adv_name)]=[]
    epoch_rec['{}(Te)'.format(adv_name)]=[]
    epoch_rec['{}(Tr_norm2)'.format(adv_name)]=[]
    epoch_rec['{}(Te_norm2)'.format(adv_name)]=[]
    epoch_rec['{}(Te_far_norm2)'.format(adv_name)]=[]
    epoch_rec['{}(Te_near_norm2)'.format(adv_name)]=[]
    epoch_rec['{}(Tr_normi)'.format(adv_name)]=[]
    epoch_rec['{}(Te_normi)'.format(adv_name)]=[]
    epoch_rec['{}(Te_far_normi)'.format(adv_name)]=[]
    epoch_rec['{}(Te_near_normi)'.format(adv_name)]=[]
def get_train_set(self,train_iter,train_dataset):
    try:
        self.train_set=next(train_iter)
    except:
        train_iter=iter(train_dataset)
        self.train_set=next(train_iter)
    self.train_set[0]=self.train_set[0].to('cuda:0')
    self.train_set[1]=self.train_set[1].to('cuda:0')
class robust_tester:
    def __init__(self,adv_switch,model):
        self.record_keys=['Epoch']
        self._flag_record_rob = False
        self.adv_switch=adv_switch
        self.model=model
    
    def init_df(self):
        former=[]
        later=[]
        for ii in self.record_keys:
            if 'norm' in ii:
                later.append(ii)
            else:
                former.append(ii)
        self.df=pd.DataFrame(columns=former+later)
        self.epoch=1
    def record_rob(self,eps):
        self.activated_adv,self.record_atks=[],{}
        for kk,vv in self.adv_switch.items():
            if vv:
                add_new_record_column(self,kk)
                self.activated_adv.append(kk)
        self.init_df()
        for adv_name in self.activated_adv:
            match adv_name:
                case 'VANILA':
                    self.record_atks['VANILA']=VANILA(self.model,device='cuda:0')
                case 'FGSM':
                    self.record_atks['FGSM']=FGSM(self.model, eps=eps,device='cuda:0')
                case 'PGD':
                    self.record_atks['PGD']=PGD(
                        self.model, eps=eps, alpha=eps/4, 
                        steps=50,device='cuda:0')
                case 'GN':
                    self.record_atks['GN']=GN(self.model, std=0.07,device='cuda:0')
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
                                preserve_portion=1,
                                noise_edge=8/255),
                            device='cuda:0')
        self._flag_record_rob = True
        self.epoch=1
    def data_pool(self,train_dataset,test_dataset,eval_bat_num):
        train_iter=iter(train_dataset)
        test_iter=iter(test_dataset)
        self.train_pool,self.test_pool=[],[]
        for ii in range(eval_bat_num):
            traini=next(train_iter)
            self.train_pool.append((traini[0].cpu(),traini[1].cpu()))
            testi=next(test_iter)
            self.test_pool.append((testi[0].cpu(),testi[1].cpu()))
    # Update Records
    def update_record(self):
        epoch_rec={}
        epoch_rec['Epoch']=self.epoch
        self.epoch+=1
        for adv_name in self.activated_adv:
            add_new_keys(epoch_rec,adv_name)
            atk=self.record_atks[adv_name]
            for ii in range(len(self.train_pool)):
                self.train_set=copy.deepcopy((self.train_pool[ii][0].cuda(),self.train_pool[ii][1].cuda()))
                train_adv_batch=[
                    (
                        atk(*self.train_set), 
                        self.train_set[1])]
                noise=(
                    train_adv_batch[0][0]-self.train_set[0]\
                    .repeat(
                        train_adv_batch[0][0].shape[0]\
                        //self.train_set[0].shape[0],
                        1,1,1))\
                .view(
                    (train_adv_batch[0][0].shape[0],-1))\
                .clone().contiguous()
                norm2=(
                    (noise**2).sum(dim=1))**0.5
                normi=noise.abs().max(dim=1)[0]
                epoch_rec['{}(Tr)'.format(adv_name)].append(
                    get_acc(
                        self.model,train_adv_batch)/100)
                epoch_rec['{}(Tr_norm2)'.format(adv_name)].append(norm2.mean().item())
                epoch_rec['{}(Tr_normi)'.format(adv_name)].append(normi.mean().item())
            for ii in range(len(self.test_pool)):
                self.test_set=copy.deepcopy((self.test_pool[ii][0].cuda(),self.test_pool[ii][1].cuda()))
                test_adv_batch=[
                    (
                        atk(*self.test_set), 
                        self.test_set[1])]
                noise=(
                    test_adv_batch[0][0]-self.test_set[0].repeat(
                            test_adv_batch[0][0].shape[0]//self.test_set[0].shape[0],
                            1,1,1))\
                .view(
                    (test_adv_batch[0][0].shape[0],-1))\
                .clone().contiguous()
                norm2=(
                    (noise**2).sum(dim=1))**0.5
                normi=noise.abs().max(dim=1)[0]
                epoch_rec['{}(Te)'.format(adv_name)].append(
                    get_acc(
                        self.model,test_adv_batch)/100)
                update_norm(epoch_rec,adv_name,norm2,normi)
            for ii in epoch_rec.keys():
                if type(epoch_rec[ii])==list:
                    epoch_rec[ii]=np.mean(epoch_rec[ii])
        self.df.loc[self.epoch,:]=epoch_rec
                         
                
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
def get_num_class(dataset):
    match dataset:
        case 'CIFAR10':
            num_classes=10
        case 'CIFAR100':
            num_classes=100
        case 'TinyImageNet':
            num_classes=200
        case 'SVHN':
            num_classes=10
    return num_classes
def data_and_model(dataset,model,eps,robust,eval_bat_num):
    train_loader, test_loader = base_loader(
        data_name=dataset,shuffle_train=False)
    global last_model,last_dataset,last_eps
    if (model!=last_model) or (dataset!=last_dataset) or (eps!=last_eps):
        last_model=model
        last_dataset=dataset
        last_eps=eps
        model = get_model(name=model, num_classes=get_num_class(dataset)).cuda()
        robust=robust_tester(
            adv_switch={
                    'VANILA':True,'FGSM':True,
                    'PGD':True,'GN':False,'BB':False},
            model=model)
        robust.record_rob(eps)
        robust.data_pool(train_loader,test_loader,eval_bat_num)
    return train_loader,test_loader,model,robust
def init_and_delete(fi,tar_ts,models,dataset,eps,robust,eval_bat_num):
    nm=fi[:-7]
    kk=tar_ts.index(fi[:-7])
    DATA_NAME = dataset[kk]
    model=models[kk]
    train_loader,test_loader,model,robust=data_and_model(DATA_NAME,model,eps,robust,eval_bat_num)
    tar_ts.pop(kk)
    dataset.pop(kk)
    models.pop(kk)
    return nm,train_loader,test_loader,model,robust
def run(
    models,eps,path=di.base,tar_ts=[],dataset=[],eval_bat_num=50):
    files=os.listdir(path)
    if (path[-1]!='/') and (path[-1]!='\\'):
        path+='/'
    global last_model,last_dataset,last_eps
    last_model,last_dataset,robust,last_eps=None,None,None,None
    for ii in range(len(files)):
        fi=files[ii]
        if fi[:-7] in tar_ts:
            nm,train_loader,test_loader,model,robust=init_and_delete(
                fi=fi,tar_ts=tar_ts,models=models,dataset=dataset,
                eps=pd.read_csv(path+fi[:-7]+'args.csv').loc[0,'eps'],
                robust=robust,eval_bat_num=eval_bat_num)
            path2=path+fi+'/'
            model_files=os.listdir(path2)
            for kk in range(len(model_files)):
                para_filei=model_files[kk]
                parai=torch.load(path2+para_filei)
                robust.model.load_state_dict(parai)
                robust.model.eval()
                robust.update_record()
                robust.df.to_csv(path+fi+'eval.csv',index=False)
                print(
                    '\r处理{}，大进度：{:.2f}%，小进度：{:.2f}%'\
                        .format(
                            nm,
                            (ii+1)/len(files)*100,
                            (kk+1)/len(model_files)*100),
                    end='')
            robust.init_df()
def get_eval_infor(para_files):
    models=[]
    tar_ts=[]
    dataset=[]
    for ii in range(len(para_files)):
        fi=para_files[ii]
        df=pd.read_csv(di.para+fi)
        models.extend(list(df['model']))
        tar_ts.extend(list(df['ts']))
        dataset.extend(list(df['dataset']))
    return models,tar_ts,dataset
if __name__=='__main__':
    para_files=['para35_ts.csv']
    models,tar_ts,dataset=get_eval_infor(para_files)
    #tar_ts=['2023.11.27.09.47','2023.11.27.11.39','2023.11.27.13.34']
    #models=['PRN18']*len(tar_ts)
    #dataset=['SVHN']*len(tar_ts)
    run(
        models=models,
        eps=8/255,path=di.base,
        tar_ts=tar_ts,
        dataset=dataset,
        eval_bat_num=10)