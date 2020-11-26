import time
import os
import bit,tvhi


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.batch_size =  32  #train batch size
        self.test_batch_size = 8  #test batch size

        # Gpu
        self.use_gpu=True
        self.use_multi_gpu=False   
        self.device_list="0,1,2"  #id list of gpus used for training 
        
        # Dataset
        assert(dataset_name in ['bit','ut','ci','tvhi'])
        dmap={'bit':bit,'tvhi':tvhi}
        self.dataset_name=dataset_name
        self.dataset_instance=dmap[dataset_name]
        self.train_seqs = []
        self.test_seqs = []
        self.conflict=self.dataset_instance.conflict

        # copy dataset meta data
        self.image_size = self.dataset_instance.image_size
        self.out_size = self.dataset_instance.out_size
        self.num_boxes = self.dataset_instance.num_boxes
        self.num_actions = self.dataset_instance.num_actions
        self.action_weight = self.dataset_instance.action_weight
        self.inter_weight = self.dataset_instance.inter_weight
        self.action_name=self.dataset_instance.action_name

        # Backbone 
        self.backbone='inv3' 
        self.crop_size = 5, 5  #crop size of roi align
        self.train_backbone = False  #if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.emb_features=1056   #output feature map channel of backbone

        # Sample
        self.num_frames = 1

        # GCN
        self.threshold=1.5
        self.num_features_boxes = 1024
        if dataset_name=='bit':
            self.num_features_relation=128
        else:self.num_features_relation=256
        self.num_graph=1  #number of graphs
        self.num_features_gcn=self.num_features_boxes
        self.gcn_layers=1  #number of GCN layers
        self.pos_threshold=0.2  #distance mask threshold in position relation
        self.base_model_name=''

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  #initial learning rate 
        self.lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  #change learning rate in these epochs 
        self.train_dropout_prob = 0.0  #dropout probability
        self.weight_decay = 0  #l2 weight decay
        self.plambda = 0.5  # weight for the loss L3, set to 0 to invalid this loss
        self.pdelta = 0.5   # weight for the loss L4, set to 0 to invalid this loss
    
        self.max_epoch=150  #max training epoch
        self.test_interval_epoch=1
        
        # Exp
        self.training_stage=1  #specify stage1 or stage2
        self.stage1_model_path=''  #path of the base model, need to be set in stage2
        self.test_before_train=False
        self.exp_note='KegNet'
        self.exp_name=None

        # log
        self.log_path=''
        self.linename='line_b01'

        self.using_high_level=True

        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name='[%s_stage%d]<%s>'%(self.exp_note,self.training_stage,time_str)
            
        self.result_path='result/%s'%self.exp_name
        self.log_path='result/%s/log.txt'%self.exp_name
            
        if need_new_folder:
            os.mkdir(self.result_path)

    def print_all_properties(self):
        print('+' * 30)
        for k,v in self.__dict__.items():
            print(k+':', v)
        print('+' * 30)
