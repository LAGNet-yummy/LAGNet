import sys
sys.path.append("..")
from test_net import test_net
from randomseed import setSeed
from config import Config
setSeed(0)

dataset_name='bit'
cfg=Config(dataset_name) # ci or bit
cfg.device_list="0,1,2" #一般只用一个
cfg.message_times=1 #1,2,3,4,5

cfg.training_stage=6 #5:gcn模式

cfg.pos_threshold=0.7 # gcn distance threshhold
cfg.num_features_gcn=1024
cfg.inter_threshold=0.5

cfg.solver = 'adam' # sgd or adam

# loss parameter
cfg.plambda=0.
cfg.pdelta=0.

cfg.use_multi_gpu=True
cfg.train_backbone = False

cfg.pretrained_model = 'base'  # base or gnn

cfg.batch_size=24 #训练的batchsize
cfg.test_batch_size=32 #测试的batchsize
cfg.lambda_h = 0.5
cfg.lambda_g = 0.1
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0
cfg.weight_decay=0
cfg.lr_plan={}
cfg.max_epoch=130
cfg.sgd_moment=0.0

cfg.backbone="inception"
cfg.emb_features=1056


cfg.log_path='../log'
cfg.save_model_path='../model'

cfg.linename='{}_{}_model_lagnet_bs{}_mt{}_{}_tb{}_pt{}'.format(dataset_name,cfg.backbone,cfg.batch_size,
                                                   cfg.message_times,cfg.pretrained_model,
                                                            cfg.train_backbone,
                                                            cfg.pos_threshold)

cfg.base_model_name='model_basemodel_bs24_inception_epoch14.pth'
cfg.data_path='../BIT'
cfg.base_model_path='../model/model_basemodel_bs24_inception_epoch14.pth'
cfg.model_path=''
cfg.test_interval_epoch=2

# set test mode
cfg.is_validation=True
if cfg.is_validation:
    cfg.linename+='_validation'
    cfg.savedmodel_path = '../model/bit_model_inception_model_gcn_meanfield_bs24_mt1_base_tbFalse_pt0.7_epoch8.pth'

cfg.print_all_properties()
cfg.train_seqs, cfg.test_seqs = cfg.dataset_instance.get_train_test_seq()

test_net(cfg)
