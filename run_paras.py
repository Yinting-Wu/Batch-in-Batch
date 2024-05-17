
from UDA.CP.train import run_para
from UDA.CP.robust_eval import run
import pandas as pd
import datetime as dtt
def uijmio():
    return dtt.datetime.now()\
    .strftime(
        '%Y.%m.%d.%H.%M')
def do_partial(dic,partial_dic):
    for ii in partial_dic.keys():
        if ii not in dic:
            dic[ii]=partial_dic[ii]
    return dic
def run_paras(para_file):
    paras=pd.read_csv(para_file)
    paras['init_eps']=paras['init_eps'].apply(pd.eval)
    paras['ts']=None
    assert ('model' in paras.columns) and ('dataset' in paras.columns),'参数文件未指定模型或数据集'
    ts_rec=[]
    ind=paras.index
    for ii in range(paras.shape[0]):
        indi=ind[ii]
        ts=uijmio()
        partial_dic=dict(
            name='CP_test',
            gpu=0,
            c=4,
            path='D:/Wu_Yin_Ting2/UDA/other/cache/CP/{}'.format(ts),
            save_type='Epoch',
            scheduler="Cyclic")
            # 缺省字典
        parai=do_partial(dict(paras.loc[indi,:]),partial_dic)
        for kk in ['eps','alpha','BB_alpha','BB_noise_edge']:
            parai[kk]=eval(parai[kk])
        ts_rec.append(ts)
        paras.loc[indi,'ts']=ts
        run_para(parai)
        paras.to_csv(para_file[:-4]+'_ts.csv',index=False)
    models=[]
    tar_ts=[]
    dataset=[]
    models.extend(list(paras['model']))
    tar_ts.extend(list(paras['ts']))
    dataset.extend(list(paras['dataset']))
    run(
        models=models,
        eps=8/255,path='D:/Wu_Yin_Ting2/UDA/other/cache/CP/',
        tar_ts=tar_ts,
        dataset=dataset,
        eval_bat_num=8)
if __name__=='__main__':
    run_paras("D:/Wu_Yin_Ting2/UDA/other/para2/para49.csv")