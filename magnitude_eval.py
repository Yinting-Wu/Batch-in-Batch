import pandas as pd
import numpy as np
import os
import torch
from UDA.CP.defenses.attacks.BB import BB
from UDA.CP.defenses.attacks.BB_single import BB_single
from defenses.model import get_model
from defenses.loader import base_loader
stg=('B','C','D','G')
class di:
    base='D:/Wu_Yin_Ting2/UDA/other/cache/CP/'
    para='D:/Wu_Yin_Ting2/UDA/other/para2/'
def B_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        cat_lab=torch.cat((cat_lab,cat_lab[:Xadv.shape[0]//n_adv,...]),dim=0).contiguous()
        Xadv=torch.cat((Xadv,ori_ima),dim=0).contiguous()
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        select=correct==False
        Xadv=Xadv[select,...]
        labels=cat_lab[select,...]
    return Xadv.detach().contiguous(),labels,select
def get_choose_func(type):
    match type:
        case 'A':
            return A_type
        case 'B':
            return B_type
        case 'C':
            return C_type
        case 'D':
            return D_type
        case 'E':
            return E_type
        case 'F':
            return F_type
        case 'G':
            return G_type
def C_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad(): 
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=pred==cat_lab
        select=correct==False
        Xadv=Xadv[select,...]
        labels=cat_lab[select,...]
    return Xadv.detach().contiguous(),labels,select
def D_type(Xadv,n_adv,ori_ima,model,cat_lab):
    return Xadv.detach().contiguous(),cat_lab,[True]*cat_lab.shape[0]
def G_type(Xadv,n_adv,ori_ima,model,cat_lab):
    with torch.no_grad():
        logits=model(Xadv)
        pred=logits.max(dim=-1).indices
        correct=(pred==cat_lab).reshape(n_adv,-1)
        radii=(Xadv-ori_ima.repeat(n_adv,1,1,1)).reshape((-1,np.cumprod(ori_ima.shape[1:])[-1])).abs().max(dim=1).values.reshape(n_adv,-1)
        class_mask=(torch.sum(correct,dim=0)<n_adv)[None,:]&correct
        radii[class_mask]=-1
        big=torch.argsort(radii,dim=0)[-1,:]
        select=torch.zeros(correct.shape,device=correct.device,dtype=bool)
        for ii in range(len(big)):
            select[big[ii],ii]=True
        select=select.flatten()
        Xadv=Xadv[select,...]
        labels=cat_lab[select,...]
    return Xadv.detach().contiguous(),labels,select
class data_holder:
    def __init__(self,dataset,n_bat):
        train_loader, test_loader = base_loader(
            data_name=dataset,shuffle_train=False)
        self.name=dataset
        self.n_bat=n_bat
        train_iter=iter(train_loader)
        test_iter=iter(test_loader)
        self.train_pool,self.test_pool=[],[]
        for ii in range(n_bat):
            traini=next(train_iter)
            self.train_pool.append((traini[0].cpu(),traini[1].cpu()))
            testi=next(test_iter)
            self.test_pool.append((testi[0].cpu(),testi[1].cpu()))
def noise_cacu(ori,adv,ty='l_i',select=None,n_adv=None):
    if adv.shape[0]==0:
        norm_i,norm_2,var=torch.zeros_like(ori),torch.zeros_like(ori),torch.zeros_like(ori)
        norm_i=norm_i.reshape(ori.shape[0],-1).mean(axis=1)
        norm_2=norm_2.reshape(ori.shape[0],-1).mean(axis=1)
        var=var.reshape(ori.shape[0],-1).mean(axis=1)
        return (norm_i,norm_2,var)
    noise=(adv-ori.repeat(len(select)//ori.shape[0],1,1,1)[select,...]).reshape(adv.shape[0],-1)
    if ty=='l_i':
        norm=noise.abs().mean(axis=1)
    if ty=='l_2':
        norm=(noise**2).sum(axis=1)**0.5
    if ty=='both':
        norm_i=noise.abs().max(axis=1).values
        norm_2=(noise**2).sum(axis=1)**0.5
        var=noise.var(axis=1)
    if ty!='both':
        return norm
    else:
        return (norm_i,norm_2,var)
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
class run:
    def __init__(
        self,models,tar_ts,dataset,epss,nadvs,
        path=di.base,eval_bat_num=50):
        self.present_model_ty=None
        self.present_dataset=None
        self.present_eps=None
        self.present_nadv=None
        self.data_pool=None
        self.doing_idx=None
        self.nadvs=nadvs
        self.models=models
        self.tar_ts=tar_ts
        self.epss=epss
        self.dataset=dataset
        self.path=path
        self.eval_bat_num=eval_bat_num
    def all_check(self,dataset,model_ty,eps,n_adv):
        self.check_datapool(dataset)
        self.check_model(model_ty)
        self.check_atk(eps,n_adv)
    def check_datapool(self,dataset):
        if dataset!=self.present_dataset:
            self.data_pool=data_holder(
                dataset=dataset,n_bat=self.eval_bat_num)
            self.present_dataset=dataset
    def check_model(self,model):
        if model!=self.present_model_ty:
            self.present_model_ty=model
            self.model=get_model(
                name=model, num_classes=get_num_class(self.present_dataset)).cuda()
    def check_atk(self,eps,n_adv):
        if (eps!=self.present_eps) or (n_adv!=self.present_nadv):
            self.present_eps,self.present_nadv=eps,n_adv
            self.s_attack=BB_single(
                model=self.model,
                BB_para=dict(
                    n_adv=n_adv,
                    eps=None,
                    noise_edge=eps,
                    preserve_portion=None,
                    eps_aug=2,
                    choose_type='D'),
                device='cuda:0',epoch=None,
                clean_warmup=0,init_type='LHD')
            self.m_attack=BB(
                model=self.model,
                BB_para=dict(
                    n_adv=n_adv,
                    iters=10,
                    alpha=eps/4,
                    norm_coe=None,
                    gamma=None,
                    eps=eps,
                    noise_des=None,
                    preserve_portion=None,
                    perturb_portion=None,
                    noise_edge=eps,
                    choose_type='D'),
                clean_warmup=0,device='cuda:0',init_type='LHD')
    def lists_maintain(self,fi):
        ts=fi[:-7]
        self.doing_idx = self.tar_ts.index(ts)
        nadv = self.nadvs.pop(self.doing_idx)
        model_ty = self.models.pop(self.doing_idx)
        eps = self.epss.pop(self.doing_idx)
        dataset = self.dataset.pop(self.doing_idx)
        self.tar_ts.pop(self.doing_idx)
        return nadv,model_ty,eps,dataset
    def run(self,path=di.base):
        files=os.listdir(path)#文件名列表
        if (path[-1]!='/') and (path[-1]!='\\'):
            path+='/'
        for ii in range(len(files)):
            fi=files[ii]
            if fi[:-7] in self.tar_ts:
                path2=path+fi+'/'
                path3=path+fi[:-7]+'_magnitude.csv'
                model_files=os.listdir(path2)
                mag_df=pd.DataFrame(
                    columns=[
                        'epoch','choose_type','single_train_li','single_test_li',
                        'single_train_l2','single_test_l2',
                        'mul_train_li','mul_test_li',
                        'mul_train_l2','mul_test_l2','s_train_var','s_test_var',
                        'm_train_var','m_test_var'])
                n_adv,model_ty,eps,dataset=self.lists_maintain(fi)
                self.all_check(
                    dataset=dataset,model_ty=model_ty,eps=eps,n_adv=n_adv)
                for kk in range(len(model_files)):
                    para_filei=model_files[kk]
                    parai=torch.load(path2+para_filei)
                    self.model.load_state_dict(parai)
                    self.model.eval()
                    train_s_dic,train_m_dic=self.magnitude(
                        bats=self.data_pool.train_pool,
                        s_attack=self.s_attack,
                        m_attack=self.m_attack,ty='both')
                    test_s_dic,test_m_dic=self.magnitude(
                        bats=self.data_pool.test_pool,
                        s_attack=self.s_attack,
                        m_attack=self.m_attack,ty='both')
                    for ti in stg:
                        #,'s_train_var','s_test_var',
                        #'m_train_var','m_test_var'
                        mag_df.loc[mag_df.shape[0],:]=dict(
                            epoch=kk+1,
                            choose_type=ti,
                            single_train_li=train_s_dic[ti][0],
                            single_test_li=test_s_dic[ti][0],
                            single_train_l2=train_s_dic[ti][1],
                            single_test_l2=test_s_dic[ti][1],
                            mul_train_li=train_m_dic[ti][0],
                            mul_test_li=test_m_dic[ti][0],
                            mul_train_l2=train_m_dic[ti][1],
                            mul_test_l2=test_m_dic[ti][1],
                            s_train_var=train_s_dic[ti][2],
                            s_test_var=test_s_dic[ti][2],
                            m_train_var=train_m_dic[ti][2],
                            m_test_var=test_m_dic[ti][2])
                    mag_df.to_csv(path3,index=False)
    def magnitude(self,
        bats,s_attack,m_attack,ty='both'):
        s_dic,m_dic=dict(B=[],C=[],D=[],G=[]),dict(B=[],C=[],D=[],G=[])
        for ii in range(len(bats)):
            ori_ima,label=bats[ii][0].cuda(),bats[ii][1].cuda()
            adv_s,adv_m=s_attack(ori_ima,label,epochi=1),m_attack(ori_ima,label,epochi=1)
                #未经选择的批量生成对抗样本
            for kk in range(len(stg)):
                stgi=stg[kk]
                selec_fun=get_choose_func(stgi)
                x_s,label_s,select_s=selec_fun(
                    Xadv=adv_s,n_adv=s_attack.n_adv,ori_ima=ori_ima,
                    model=s_attack.model,cat_lab=s_attack.labels)
                    #筛选后的对抗样本
                x_m,label_m,select_m=selec_fun(
                    Xadv=adv_m,n_adv=m_attack.max_n_adv,ori_ima=ori_ima,
                    model=m_attack.model,cat_lab=m_attack.labels)
                s_normi,s_norm2,s_var=noise_cacu(ori_ima,x_s,ty=ty,select=select_s,n_adv=s_attack.n_adv)
                m_normi,m_norm2,m_var=noise_cacu(ori_ima,x_m,ty=ty,select=select_m,n_adv=m_attack.max_n_adv)
                s_dic[stgi].append((s_normi.mean().item(),s_norm2.mean().item(),s_var.mean().item()))
                    # 对batch内求均值
                    # 键：筛选函数类型
                    # 值：列表，描述整个数据集上的攻击半径情况
                        # 第i个元素是元组，描述第i个batch中的图片的平均攻击半径
                            # (无穷范数半径，2范数半径)
                m_dic[stgi].append((m_normi.mean().item(),m_norm2.mean().item(),m_var.mean().item()))
        for dici in (s_dic,m_dic):
            for ki in dici.keys():
                dici[ki]=np.array(dici[ki]).mean(axis=0)
                    # 键：筛选函数类型
                    # 值：长度为2的array,表明key选择类型在该数据集上的两个平均攻击半径
        return s_dic,m_dic
def get_eval_infor(para_files):
    models=[]
    tar_ts=[]
    dataset=[]
    epss=[]
    nadvs=[]
    for ii in range(len(para_files)):
        fi=para_files[ii]
        df=pd.read_csv(di.para+fi)
        models.extend(list(df['model']))
        tar_ts.extend(list(df['ts']))
        dataset.extend(list(df['dataset']))
        epss.extend(list(df['eps']))
        nadvs.extend(list(df['n_adv']))
    for ii in range(len(epss)):
        epss[ii]=eval(epss[ii])
    return models,tar_ts,dataset,epss,nadvs
if __name__=='__main__':
    para_files=['mag_test2.csv']
    models,tar_ts,dataset,epss,nadvs=get_eval_infor(para_files)
    run_ins=run(models,tar_ts,dataset,epss,nadvs,
        path=di.base,eval_bat_num=50)
    run_ins.run(path=di.base)
    1