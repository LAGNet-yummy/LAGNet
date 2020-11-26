import sys
sys.path.append("..")
from test_net import *
from randomseed import setSeed
from config import Config
setSeed(10)

dataset_name='tvhi'
cfg=Config(dataset_name)
cfg.device_list="0,1,2"
cfg.message_times=1
cfg.training_stage=6 #meanfield mode

# loss parameter
cfg.plambda=0
cfg.pdelta=0

cfg.pos_threshold=0.4 # gcn distance threshhold
cfg.num_features_gcn=1024
cfg.inter_threshold=0.5

# relation reasoning method
# option：mlp,dotproduct
cfg.solver = 'adam' # sgd or adam

cfg.use_multi_gpu=True
cfg.train_backbone = False

cfg.pretrained_model = 'base'  # base or gnn


cfg.batch_size=16 #train batchsize
cfg.test_batch_size=32 #test batchsize
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.0
cfg.weight_decay=0.0
cfg.sgd_moment=0.0
cfg.lr_plan={}
# cfg.lr_plan={41:1e-4,81:5e-5,121:1e-5}

cfg.lambda_h = .5
cfg.lambda_g = .1

cfg.max_epoch=200

cfg.backbone="inception"
cfg.emb_features=1056

cfg.data_path='../highfive' # dataset root directory
cfg.base_model_path="../model/tvhi_model_basemodel_bs16_inception_epoch8.pth"
cfg.model_path="../model/tvhi_model_basemodel_bs16_inception_epoch8.pth"
cfg.log_path='../log'
cfg.save_model_path='../model'

cfg.linename='{}_{}_lagnet_bs{}_mt{}_{}_tb{}'.format(dataset_name,cfg.backbone,cfg.batch_size,
                                                           cfg.message_times,
                                                       cfg.pretrained_model,
                                                            cfg.train_backbone)

if cfg.plambda:
    cfg.linename += '_l3'
if cfg.pdelta:
    cfg.linename += '_l4'


cfg.test_interval_epoch=2

# set test mode
cfg.is_validation=True
if cfg.is_validation:
    cfg.linename+='_validation'
    cfg.savedmodel_path = '../model/tvhi_model_inception_gcn_meanfield_bs16_mt1_base_tbFalse_epoch50.pth'


cfg.print_all_properties()

cfg.train_seqs,cfg.test_seqs=cfg.dataset_instance.get_train_test_seq()

test_net(cfg)

