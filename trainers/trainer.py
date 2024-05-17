import os
import datetime as dtt
import torch
import pandas as pd
import time
import numpy as np

from torch.optim import *
from torch.optim.lr_scheduler import *
from torchhk import RecordManager

r"""
Trainer.

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

"""

class Trainer():
    def __init__(self, name, model, **kwargs):
        # Set Model
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device
        self.step_recorder=pd.DataFrame(
            columns=self.record_keys_step)
        self.steps_done=0
        
        # Set Custom Arguments
        for k, v in kwargs.items():
#             assert( k in [])
            setattr(self, k, v)
        
    def train(self, train_loader, max_epoch=200, start_epoch=0,
              optimizer=None, scheduler=None, scheduler_type=None,
              save_type="Epoch", save_path=None, save_overwrite=False, 
              record_type="Epoch"):
        
        # Set Train Mode
        self.model.train()
        
        # Set Epoch and Iterations
        self.max_epoch = max_epoch
        self.max_iter = len(train_loader)
            
        # Set Optimizer and Schduler
        self._init_optim(optimizer, scheduler, scheduler_type)
        
        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)
        
        # Check Save Path is given
        if save_type is None:
            save_type = "None"
        else:
            # Check Save Path
            if save_path is not None:
                # Save Initial Model
                self._check_path(save_path, overwrite=save_overwrite)
                self._save_model(save_path, 0)
            else:
                raise ValueError("save_path should be given for save_type != None.")
            
        # Print Training Information
        if record_type is not None:
            self._init_record(record_type)
            print("["+self.name+"]")
            print("Training Information.")
            print("-Epochs:",self.max_epoch)
            print("-Optimizer:",self.optimizer)
            print("-Scheduler:",self.scheduler)
            print("-Save Path:",save_path)
            print("-Save Type:",save_type)
            print("-Record Type:",record_type)
            print("-Device:",self.device)
        
        # Training Start
        total_steps=self.max_epoch*len(train_loader)
        passed_steps=0
        mean_step_time=0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            epoch_record = []
            
            if epoch < start_epoch:
                if self.scheduler_type == "Epoch":
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    for i in range(max_iter):
                        self._update_scheduler()
                else:
                    pass
                continue
            
            for i, train_data in enumerate(train_loader):
                self.iter = i
                t1=time.time()
                iter_record = self._do_iter(train_data)
                t2=time.time()
                passed_steps+=1
                mean_step_time=1/passed_steps*(t2-t1+(passed_steps-1)*mean_step_time)
                iter_record['Epoch']=epoch+1
                iter_record['Iter']=i+1
                iter_record['lr']=self.optimizer.param_groups[0]['lr']
                iter_record['train_time']=t2-t1
                end_time=dtt.datetime.now()+dtt.timedelta(
                    seconds=(total_steps-passed_steps)*mean_step_time)
                print(
                    (
                        '\r进度：{:.2f}%, '\
                        +'adv_normi:{:.2f}, '\
                        +'t_loss:{:.2f}, '\
                        +'adv_ce:{:.2f}, '\
                        +'end_est:{}'\
                        +' '*4)\
                    .format(
                        passed_steps/total_steps*100,
                        iter_record['adv_normi'],
                        iter_record['loss'],
                        iter_record['adv_ce'],
                        end_time.strftime(
                            '%m月%d日%H时%M分%S秒')),
                    end='')
                
                # Check Last Batch
                is_last_batch = (i+1==self.max_iter)
                    
                # Update Records
                epoch_list=[]
                for nmi in self.record_keys:
                    if nmi in iter_record:
                        epoch_list.append(iter_record[nmi])
                epoch_record.append(epoch_list)
                if is_last_batch:
                    epoch_record = torch.tensor(epoch_record).mean(dim=0)
                    epoch_dic={}
                    idd=0
                    for vv in range(len(self.record_keys)):
                        nmi=self.record_keys[vv]
                        if nmi in iter_record:
                            epoch_dic[nmi]=epoch_record[idd].item()
                            idd+=1
                    self._update_record(epoch_dic)
                    epoch_record = []
                self.steps_done+=1
                step_list=[]
                for nmi in self.record_keys_step:
                    step_list.append(iter_record[nmi])
                self.step_recorder.loc[self.steps_done,:]=step_list

                # Save Model
                if save_type == "Epoch":
                    if is_last_batch:
                        self._save_model(save_path, epoch+1)
                elif save_type == "Iter":
                    self._save_model(save_path, epoch+1, i+1)
                else:
                    pass
                
                # Scheduler Step
                if self.scheduler_type=="Epoch" and is_last_batch:
                    self._update_scheduler()
                elif self.scheduler_type=="Iter":
                    self._update_scheduler()
                else:
                    pass
                
                # Set Train Mode
                self.model = self.model.to(self.device)
                self.model.train()
            self.save_all(save_path)
                
        # Print Summary
        if False:
            try:
                if record_type is not None:
                    self.rm.summary()
            except Exception as e:
                print("Summary Error:",e)
        
    def save_all(self, save_path, overwrite=True):
        self._check_path(save_path+".pth", overwrite=overwrite, file=True)
        self._check_path(save_path+".csv", overwrite=overwrite, file=True)
        print("Saving Model")
        torch.save(self.model.cpu().state_dict(), save_path+".pth")
        print("...Saved as pth to %s !"%(save_path+".pth"))
        print("Saving Records")
        self.rm.to_csv(save_path+".csv")
        self.step_recorder.to_csv(save_path+"，steps.csv")
        self.model.to(self.device)
    
    #############################
    # OVERRIDE BELOW FUNCTIONS #
    ############################
    
    # Do Iter
    def _do_iter(self, images, labels):
        raise NotImplementedError
        
    # Scheduler Update
    def _update_scheduler(self):
        self.scheduler.step()
        
    ####################################
    # DO NOT OVERRIDE BELOW FUNCTIONS #
    ###################################
            
    # Initialization RecordManager
    def _init_record(self, record_type):
        self.rm = RecordManager(self.record_keys)
    
    # Update Records
    def _update_record(self, records):
        self.rm.add(records)
        
    # Set Optimizer and Scheduler
    def _init_optim(self, optimizer, scheduler, scheduler_type):
        # Set Optimizer
        if not isinstance(optimizer, str):
            self.optimizer = optimizer     
        else:
            exec("self.optimizer = " + optimizer.split("(")[0] + "(self.model.parameters()," + optimizer.split("(")[1])

        # Set Scheduler
        if not isinstance(scheduler, str):
            self.scheduler = scheduler
            if self.scheduler is None:
                self.scheduler_type = None
            else:
                if scheduler_type is None:
                    raise ValueError("The type of scheduler must be specified as 'Epoch' or 'Iter'.")
                self.scheduler_type = scheduler_type
        else:
            if "Step(" in scheduler:
                # Step(milestones=[2, 4], gamma=0.1)
                exec("self.scheduler = " + "MultiStepLR(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = 'Epoch'

            elif 'Cyclic(' in scheduler:
                # Cyclic(base_lr=0, max_lr=0.3)
                lr_steps = self.max_epoch * self.max_iter
                exec("self.scheduler = " + "CyclicLR(self.optimizer, " + scheduler.split("(")[1].split(")")[0] + \
                     ", step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)")
                self.scheduler_type = 'Iter'

            elif 'Cosine' == scheduler:
                # Cosine
                self.scheduler = CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=0)
                self.scheduler_type = 'Epoch'
                
            else:
                exec("self.scheduler = " + scheduler.split("(")[0] + "(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = scheduler_type
            
    def _save_model(self, save_path, epoch, i=0):
        torch.save(self.model.cpu().state_dict(),
                   save_path+"/"+str(epoch).zfill(len(str(self.max_epoch)))\
                   +"_"+str(i).zfill(len(str(self.max_iter)))+".pth")
        self.model.to(self.device)
        
    # Check and Create Path
    def _check_path(self, path, overwrite=False, file=False):
        if os.path.exists(path):
            if overwrite:
                print("Warning: Save files will be overwritten!")
            else:
                raise ValueError('[%s] is already exists.'%(path))
        else:
            if not file:
                os.makedirs(path)
                
    # Check Valid Options
    def _check_valid_options(self, key):
        if key in ["Epoch", "Iter", None]:
            pass
        else:
            raise ValueError(key, " is not valid. [Hint:'Epoch', 'Iter', None]")