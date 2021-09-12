import torch
from torch import Tensor

import numpy as np

import random

from cogdl.data import Graph

from cogdl import experiment
from cogdl.datasets import NodeDataset, register_dataset


default_dataset_root=r'/data/wanghuijuan/dataset2/rd2pd_ds'
default_dataset_dst="whj_code2/cogdl_fork/cogdl_ds"
dataset_names=['Github','Elliptic','Film','Wiki','Clothing','Electronics','Dblp','Yelpchi',
                'Alpha','Weibo','bgp','ssn5','ssn7','chameleon','squirrel','Aids','Nba',
                'Wisconsin','Texas','Cornell','Pokec_z']

def get_whole_mask(y,ratio:str,seed:int=1234567):
    """对整个数据集按比例进行划分"""
    y_have_label_mask=y!=-1
    total_node_num=len(y)
    y_index_tensor=torch.tensor(list(range(total_node_num)),dtype=int)
    masked_index=y_index_tensor[y_have_label_mask]
    while True:
        (train_mask,val_mask,test_mask)=get_order(ratio,masked_index,total_node_num,seed)
        if check_train_containing(train_mask,y):
            return (train_mask,val_mask,test_mask)
        else:
            seed+=1

def get_order(ratio:str,masked_index:Tensor,total_node_num:int,seed:int=1234567):
    """
    输入划分比例和原始的索引，输出对应划分的mask元组

    入参：
    ratio格式：'1-1-3'
    masked_index是索引的1维Tensor
    TODO：增加对其他格式masked_index的支持

    返回值：(train_mask,val_mask,test_mask)
    都是长度为总节点数，对应索引置True的布尔Tensor
    """
    random.seed(seed)

    masked_node_num=len(masked_index)
    shuffle_criterion=list(range(masked_node_num))
    random.shuffle(shuffle_criterion)

    train_val_test_list=[int(i) for i in ratio.split('-')]
    tvt_sum=sum(train_val_test_list)
    tvt_ratio_list=[i/tvt_sum for i in train_val_test_list]
    #TODO:支持对masked_node_num数不足的情况下进行处理
    train_end_index=int(tvt_ratio_list[0]*masked_node_num)
    val_end_index=train_end_index+int(tvt_ratio_list[1]*masked_node_num)

    train_mask_index=shuffle_criterion[:train_end_index]
    val_mask_index=shuffle_criterion[train_end_index:val_end_index]
    test_mask_index=shuffle_criterion[val_end_index:]

    train_mask=torch.zeros(total_node_num,dtype=torch.bool)
    train_mask[masked_index[train_mask_index]]=True
    val_mask=torch.zeros(total_node_num,dtype=torch.bool)
    val_mask[masked_index[val_mask_index]]=True
    test_mask=torch.zeros(total_node_num,dtype=torch.bool)
    test_mask[masked_index[test_mask_index]]=True

    return (train_mask,val_mask,test_mask)

def check_train_containing(train_mask,y):
    """（仅用于分类任务）检查train_mask中是否含有y中所有的标签（-1不算）"""
    for label in y.unique():
        l=label.item()
        if l==-1:
            continue
        if l not in y[train_mask]:
            return False
    return True





class RD2CD(NodeDataset):
    def __init__(self, dataset_root,dataset_name,path):
        """
        dataset_root: 原始数据集存储位置（内置名为dataset_name的文件夹，文件夹下内置x.npy、y.npy、edge_index.npy文件）
        dataset_name: 如Github
        path: 处理后数据要存储的位置（在该位置下放置名为dataset_name_data.pt的处理后数据）
        """
        self.dataset_root=dataset_root
        self.dataset_name=dataset_name
        self.data_path=path+'/'+dataset_name+'_data.pt'
        super(RD2CD, self).__init__(path=self.data_path,scale_feat=False)
        #注意：这个scale_feat参数T或F无所谓的

    def download(self):
        print('该数据集暂未上传至公开下载渠道，敬请期待！')
    #TODO:提供下载位置到dataset_root下

    def process(self):
        dataset_root=self.dataset_root
        dataset_name=self.dataset_name
        numpy_x=np.load(dataset_root+'/'+dataset_name+'/x.npy')
        x=torch.from_numpy(numpy_x).to(torch.float)
        numpy_y=np.load(dataset_root+'/'+dataset_name+'/y.npy')
        y=torch.from_numpy(numpy_y).to(torch.long)
        numpy_edge_index=np.load(dataset_root+'/'+dataset_name+'/edge_index.npy')
        edge_index=torch.from_numpy(numpy_edge_index).to(torch.long)

        # set train/val/test mask in node_classification task
        random_seed=14530529  #固定随机种子以保证划分的固定性
        (train_mask,val_mask,test_mask)=get_whole_mask(y,'6-2-2',random_seed)
        data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        torch.save(data, self.path)
        return data







@register_dataset('rd2cd_Github')
class rd2cd_Github(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Github,self).__init__(default_dataset_root,'Github',self.path)

@register_dataset('rd2cd_Elliptic')
class rd2cd_Elliptic(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Elliptic,self).__init__(default_dataset_root,'Elliptic',self.path)

@register_dataset('rd2cd_Film')
class rd2cd_Film(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Film,self).__init__(default_dataset_root,'Film',self.path)

@register_dataset('rd2cd_Wiki')
class rd2cd_Wiki(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Wiki,self).__init__(default_dataset_root,'Wiki',self.path)

@register_dataset('rd2cd_Clothing')
class rd2cd_Clothing(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Clothing,self).__init__(default_dataset_root,'Clothing',self.path)

@register_dataset('rd2cd_Electronics')
class rd2cd_Electronics(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Electronics,self).__init__(default_dataset_root,'Electronics',self.path)

@register_dataset('rd2cd_Dblp')
class rd2cd_Dblp(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Dblp,self).__init__(default_dataset_root,'Dblp',self.path)

@register_dataset('rd2cd_Yelpchi')
class rd2cd_Yelpchi(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Yelpchi,self).__init__(default_dataset_root,'Yelpchi',self.path)

@register_dataset('rd2cd_Alpha')
class rd2cd_Alpha(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Alpha,self).__init__(default_dataset_root,'Alpha',self.path)

@register_dataset('rd2cd_Weibo')
class rd2cd_Weibo(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Weibo,self).__init__(default_dataset_root,'Weibo',self.path)

@register_dataset('rd2cd_bgp')
class rd2cd_bgp(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_bgp,self).__init__(default_dataset_root,'bgp',self.path)

@register_dataset('rd2cd_ssn5')
class rd2cd_ssn5(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_ssn5,self).__init__(default_dataset_root,'ssn5',self.path)

@register_dataset('rd2cd_ssn7')
class rd2cd_ssn7(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_ssn7,self).__init__(default_dataset_root,'ssn7',self.path)

@register_dataset('rd2cd_chameleon')
class rd2cd_chameleon(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_chameleon,self).__init__(default_dataset_root,'chameleon',self.path)

@register_dataset('rd2cd_squirrel')
class rd2cd_squirrel(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_squirrel,self).__init__(default_dataset_root,'squirrel',self.path)

@register_dataset('rd2cd_Aids')
class rd2cd_Aids(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Aids,self).__init__(default_dataset_root,'Aids',self.path)

@register_dataset('rd2cd_Nba')
class rd2cd_Nba(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Nba,self).__init__(default_dataset_root,'Nba',self.path)

@register_dataset('rd2cd_Wisconsin')
class rd2cd_Wisconsin(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Wisconsin,self).__init__(default_dataset_root,'Wisconsin',self.path)

@register_dataset('rd2cd_Texas')
class rd2cd_Texas(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Texas,self).__init__(default_dataset_root,'Texas',self.path)

@register_dataset('rd2cd_Cornell')
class rd2cd_Cornell(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Cornell,self).__init__(default_dataset_root,'Cornell',self.path)

@register_dataset('rd2cd_Pokec_z')
class rd2cd_Pokec_z(RD2CD):
    def __init__(self):
        self.path = default_dataset_dst
        super(rd2cd_Pokec_z,self).__init__(default_dataset_root,'Pokec_z',self.path)

#experiment(task="node_classification", dataset="rd2cd_Github", model="gcn")
#experiment(task="node_classification", dataset="rd2cd_Pokec_z", model="gcn")