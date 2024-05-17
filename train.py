from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
import random
import pickle

import torch
import torch.optim as optim
import datetime as dtt

from defenses.loader import base_loader
from defenses.model import get_model
from defenses.trainer import COAdvTrainer
from defenses.trainer import FastAdvTrainer
from defenses.loaders.teacher_loader import get_teachers
from UDA.CP.defenses.trainers.BB_adv_trainer import BB_trainer
from UDA.CP.defenses.trainers.PGD_trainer import PGD_trainer
from UDA.CP.defenses.trainers.BB_single_trainer import BB_single_trainer
from UDA.CP.defenses.trainers.CP_trainer import CP_trainer
from UDA.CP.defenses.trainers.BB_CO_trainer import BB_CO_trainer
from UDA.CP.defenses.trainers.smooth_NFGSM import smooth_single_trainer

teacher_dir=dict(
    CIFAR10=dict(
        NFGSM="D:/Wu_Yin_Ting2/UDA/other/cache/CP/2023.11.10.03.15CP_test/061_000.pth",
        PGD="D:/Wu_Yin_Ting2/UDA/other/cache/CP/2023.10.26.10.30CP_test/62_000.pth",
        Clean="D:/Wu_Yin_Ting2/UDA/other/cache/CP/2023.11.11.16.20CP_test/064_000.pth"),
    CIFAR100=dict(
        NFGSM=[],
        PGD=[],
        Clean=[]),
    SVHN=dict(
        NFGSM=[],
        PGD=[],
        Clean=[]))
def psave(ob,di):
    try:
        tt=open(di,'wb')
        pickle.dump(ob,tt)
    finally:
        try:
            tt.close()
        except:
            pass
def seed_all(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
def run(
    name, method, model, gpu, scheduler, 
    epochs, eps, alpha, c, inf_batch, 
    path, save_type,seed=None,args=None):
    seed_all(seed)

    torch.cuda.set_device(gpu)
    
    DATA_NAME = args.dataset
    match args.dataset:
        case 'CIFAR10':
            num_classes=10
        case 'CIFAR100':
            num_classes=100
        case 'TinyImageNet':
            num_classes=200
        case 'SVHN':
            num_classes=10
    
    # Set Train, Test Loader
    train_loader, test_loader = base_loader(data_name=DATA_NAME,
                                            shuffle_train=True)
    train_loader_ns, _ = base_loader(data_name=DATA_NAME,
                                     shuffle_train=False) #Train w/o Suffle

    # Get First batch
    train_set = next(iter(train_loader_ns))
    test_set = next(iter(test_loader))

    # Set Model
    if args.teacher1 or args.teacher2:
        if method=='smooth_NFGSM' or method=='BB_s':
            adv_name='NFGSM'
        elif method=='BB':
            adv_name='PGD'
    else:
        adv_name='NFGSM' # 不使用teacher模型，这里的adv_name无实意
    teacher1,teacher2,swa_model=get_teachers(
        model_type=model,
        read_tb=dict(
            teacher1=(args.teacher1,teacher_dir[DATA_NAME][adv_name]),
            teacher2=(args.teacher2,teacher_dir[DATA_NAME]['Clean']),
            SWA=args.SWA),
        num_classes=num_classes)
    model = get_model(name=model, num_classes=num_classes).cuda()
    if args.check_point is not None:
        model.load_state_dict(torch.load(args.check_point))

    # Set Trainer
    if method == "fast":
        trainer = FastAdvTrainer(
            model, eps=eps, alpha=alpha, c=c)
    elif method == "CP":
        trainer = COAdvTrainer(
            model, eps=eps, alpha=alpha, c=c, 
            inf_batch=inf_batch)
    elif method=='BB':
        trainer=BB_trainer(
            model=model,
            BB_para=dict(
                n_adv=args.n_adv,
                iters=args.steps,
                alpha=args.BB_alpha,
                norm_coe=args.norm_coe,
                gamma=args.gamma,
                eps=args.init_eps,
                noise_des=1,
                perturb_portion=args.perturb_portion,
                preserve_portion=args.preserve_portion,
                noise_edge=args.BB_noise_edge,
                eps_aug=args.eps_aug,
                choose_type=args.choose_type,
                init_type=args.init_type),
            swa_model=swa_model,clean_warmup=args.clean_warmup,swa_init=args.swa_init,
            adv_switch={
                'VANILA':True,'FGSM':True,
                'PGD':True,'GN':False,'BB':False},
            one_ep_steps=len(train_loader))
    elif method=='PGD':
        trainer=PGD_trainer(model=model,eps=args.eps,alpha=args.alpha,steps=args.steps)
    elif method=='BB_s':
        trainer=BB_single_trainer(
            model=model,swa_model=swa_model,
            clean_warmup=args.clean_warmup,swa_init=args.swa_init,BB_para=dict(
                n_adv=args.n_adv,
                eps=args.init_eps,
                noise_edge=args.BB_noise_edge,
                preserve_portion=args.preserve_portion,
                eps_aug=args.eps_aug,
                choose_type=args.choose_type,
                init_type=args.init_type),
            one_ep_steps=len(train_loader),
            epoch=epochs)
    elif method=='smooth_NFGSM':
        trainer=smooth_single_trainer(
            model=model,
            BB_para=dict(
                n_adv=args.n_adv,
                eps=args.init_eps,
                noise_edge=args.BB_noise_edge,
                preserve_portion=args.preserve_portion,
                eps_aug=args.eps_aug,
                choose_type=args.choose_type,
                init_type=args.init_type),
            teacher1=teacher1,
            teacher2=teacher2,swa_model=swa_model,
            swa_init=args.swa_init,ce_coe=0.3,t1_coe=0.1,t2_coe=0.6,
            one_ep_steps=len(train_loader),
            epoch=epochs)
    elif method=='CP_2':
        trainer=CP_trainer(model=model,cp_num=c,eps=eps,device='cuda:0')
    elif method=='BB_CO':
        trainer=BB_CO_trainer(
            model=model,BB_para=dict(
                n_adv=args.n_adv,
                eps=args.init_eps,
                noise_edge=args.BB_noise_edge,
                preserve_portion=args.preserve_portion),
            one_ep_steps=len(train_loader),
            epoch=epochs)
    else:
        raise ValueError(method + " is not supported method.")
    trainer.adv_switch={
                'VANILA':True,'FGSM':True,
                'PGD':True,'GN':False,'BB':False}
    #trainer.record_rob(train_set, test_set, eps=eps, alpha=alpha, steps=7, BB_alpha=args.BB_alpha)

    optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)"
    
    if scheduler=="Cyclic":
        scheduler="Cyclic(0, 0.3)"
        scheduler_type="Iter"
    elif scheduler=="Stepwise":
        scheduler="Step([28,56], 0.1)"
        scheduler_type="Epoch"
    else:
        raise ValueError("%s is not supported scheduler."%(scheduler))
        
    # Train Model
    if save_type == "None":
        save_type = None
    pd.DataFrame(args.__dict__,index=[0]).to_csv(path+'args.csv',index=False)
    trainer.train(train_loader=train_loader, max_epoch=epochs,
                  optimizer=optimizer, scheduler=scheduler, scheduler_type=scheduler_type,
                  save_type=save_type, save_path=path+name,
                  save_overwrite=True, record_type="Epoch")

    print("Train Done!")
def uijmio():
    return dtt.datetime.now()\
    .strftime(
        '%Y.%m.%d.%H.%M.%S')
def arg_construction(para_dic):
    parser = ArgumentParser(description="Training script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, help='Name for this training script')
    parser.add_argument('--method', type=str, help='Training method')
    parser.add_argument('--model', default='PRN18', type=str, choices=['PRN18', 'WRN28'], help='Model Structure')
    parser.add_argument('--gpu', default=0, type=int, help='GPU to be used')
    parser.add_argument('--scheduler', default='Stepwise', type=str, choices=['Stepwise', 'Cyclic'], help='Scheduler type')
    parser.add_argument('--epochs', default=200, type=int, help='Numbers of epochs')
    parser.add_argument('--eps', default=8, type=float, help='Maximum perturbation (ex.8)')
    parser.add_argument('--alpha', default=10, type=float, help='Stepsize (ex.12)')
    parser.add_argument('--c', default=3, type=int, help='Number of checkpoints')
    parser.add_argument('--inf-batch', default=1024, type=int, help='Number of batches during checkpoints inference')
    parser.add_argument('--path', default="./", type=str, help='Save path')
    parser.add_argument('--save-type', default="None", type=str, choices=['None', 'Epoch'], help='Epoch to save the model every epoch')
    parser.add_argument('--n_adv',default=1, type=int)
    parser.add_argument('--norm_coe',default=1/10, type=float)
    parser.add_argument('--gamma',default=0.7, type=float)
    parser.add_argument('--init_eps',default=1e-4, type=float)
    parser.add_argument('--perturb_portion',default=1/20, type=float)
    parser.add_argument('--preserve_portion',default=1, type=float)
    parser.add_argument('--steps',default=1, type=int)
    parser.add_argument('--adv_coe',default=1, type=float)
    parser.add_argument('--seed',default=1, type=int)
    parser.add_argument('--BB_alpha',default=1, type=float)
    parser.add_argument('--BB_noise_edge',default=1.72, type=float)
    parser.add_argument('--eps_aug',default=2, type=float)
    parser.add_argument('--choose_type',default='A', type=str)
    parser.add_argument('--dataset',default='CIFAR10', type=str)
    parser.add_argument('--init_type',default='LHD', type=str)
    parser.add_argument('--SWA',default=False, type=bool)
    parser.add_argument('--swa_init',default=55, type=int)
    parser.add_argument('--teacher1',default=False, type=bool)
    parser.add_argument('--teacher2',default=False, type=bool)
    parser.add_argument('--clean_warmup',default=0, type=int)
    parser.add_argument('--check_point',default='', type=str)
    arg_set=[
        '--name {}'.format(para_dic['name']),
        '--method {}'.format(para_dic['method']),
        '--gpu {}'.format(para_dic['gpu']),
        '--epochs {}'.format(para_dic['epochs']),
        '--eps {}'.format(para_dic['eps']),
        '--alpha {}'.format(para_dic['alpha']),
        '--c {}'.format(para_dic['c']),
        '--path {}'.format(para_dic['path']),
        '--save-type {}'.format(para_dic['save_type']),
        '--model {}'.format(para_dic['model']),
        '--n_adv {}'.format(para_dic['n_adv']),
        '--norm_coe {}'.format(para_dic['norm_coe']),
        '--gamma {}'.format(para_dic['gamma']),
        '--init_eps {}'.format(para_dic['init_eps']),
        '--perturb_portion {}'.format(para_dic['perturb_portion']),
        '--preserve_portion {}'.format(para_dic['preserve_portion']),
        '--steps {}'.format(para_dic['steps']),
        '--adv_coe {}'.format(para_dic['adv_coe']),
        '--seed {}'.format(para_dic['seed']),
        '--BB_alpha {}'.format(para_dic['BB_alpha']),
        '--BB_noise_edge {}'.format(para_dic['BB_noise_edge']),
        '--eps_aug {}'.format(para_dic['eps_aug']),
        '--choose_type {}'.format(para_dic['choose_type']),
        '--dataset {}'.format(para_dic['dataset']),
        '--init_type {}'.format(para_dic['init_type']),
        '--SWA {}'.format(para_dic['SWA']),
        '--swa_init {}'.format(para_dic['swa_init']),
        '--teacher1 {}'.format(para_dic['teacher1']),
        '--teacher2 {}'.format(para_dic['teacher2']),
        '--clean_warmup {}'.format(para_dic['clean_warmup']),
        '--scheduler {}'.format(para_dic['scheduler']),
        '--check_point {}'.format(para_dic['check_point'])]
    for i in range(len(arg_set)-1,-1,-1):
        arg_set[i:(i+1)]=arg_set[i].split()
    args = parser.parse_args(arg_set)
    args.teacher1,args.teacher2,args.SWA=bool(para_dic['teacher1']),bool(para_dic['teacher2']),bool(para_dic['SWA'])
    if args.check_point=='no':
        args.check_point=None
    return args
def run_para(para_dic):
    args=arg_construction(para_dic)
    run(
        name = args.name,
        method = args.method,
        model = args.model,
        gpu = args.gpu,
        scheduler = args.scheduler,
        epochs = args.epochs,
        eps = args.eps,
        alpha = args.alpha,
        c = args.c,
        inf_batch = args.inf_batch,
        path = args.path,
        save_type = args.save_type,
        seed = args.seed,
        args=args)
if __name__ == "__main__":
    para_dic=dict(
        name='CP_test',
        method='BB_CO',
        n_adv=5,
        choose_type='C',
        steps=10,
        gpu=0,
        epochs=200,
        eps=8/255,
        alpha=2/255,
        BB_alpha=2/255,
        c=4,
        path='D:/Wu_Yin_Ting2/UDA/other/cache/CP/{}'.format(uijmio()),
        save_type='Epoch',
        model='PRN18',
        norm_coe=0,
        gamma=1,
        init_eps=1e-4,
        perturb_portion=0,
        preserve_portion=1,
        adv_coe=1,
        seed=3,
        BB_noise_edge=8/255,
        eps_aug=2)
    args=arg_construction(para_dic)
    run(
        name = args.name,
        method = args.method,
        model = args.model,
        gpu = args.gpu,
        scheduler = args.scheduler,
        epochs = args.epochs,
        eps = args.eps,
        alpha = args.alpha,
        c = args.c,
        inf_batch = args.inf_batch,
        path = args.path,
        save_type = args.save_type,
        seed = args.seed,
        args=args)