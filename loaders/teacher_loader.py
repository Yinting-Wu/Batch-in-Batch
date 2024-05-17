import torch
import copy
from UDA.CP.defenses.model import get_model

def get_teachers(model_type,read_tb,
    num_classes):
    '''
    输入
        read_tb
            字典
                键，值
                    'teacher1',(读取开关bool,模型参数路径)
                    'teacher2',(读取开关bool,模型参数路径)
                    'SWA':是否使用SWA的bool
    '''
    if read_tb['teacher1'][0]:
        teacher1 = get_model(
            name=model_type, num_classes=num_classes).cuda()
        teacher1.load_state_dict(
            torch.load(
                read_tb['teacher1'][1]))
        teacher1.cuda()
    else:
        teacher1=None
    if read_tb['teacher2'][0]:
        teacher2=get_model(
            name=model_type, num_classes=num_classes).cuda()
        teacher2.load_state_dict(torch.load(read_tb['teacher2'][1]))
        teacher2.cuda()
    else:
        teacher2=None
    if read_tb['SWA']:
        swa_model=get_model(
            name=model_type, num_classes=num_classes).cuda()
        swa_model.cuda()
    else:
        swa_model=None
    return teacher1,teacher2,swa_model
if __name__=='__main__':
    teacher1,teacher2,swa_model=get_teachers(
        model_type='PRN18',read_tb=dict(
            teacher1=(True,"D:/Wu_Yin_Ting2/UDA/other/cache/CP/2023.10.12.16.38.11CP_test/063_000.pth"),
            teacher2=(True,"D:/Wu_Yin_Ting2/UDA/other/cache/CP/2023.10.12.16.38.11CP_test/064_000.pth"),
            SWA=True),
        num_classes=10)
    1