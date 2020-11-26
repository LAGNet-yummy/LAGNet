import sys
sys.path.append("..")
from test_net import test_net
from randomseed import setSeed
from config import Config
setSeed(10)

dataset_name='tvhi'
cfg=Config(dataset_name)
cfg.device_list="0,1,2"
cfg.message_times=1
cfg.training_stage=5 #gcn mode

cfg.pos_threshold=0.4 # gcn distance threshhold
cfg.num_features_gcn=1024
cfg.num_graph=1
cfg.inter_threshold=0.5
cfg.using_mse=True

# loss parameter
cfg.plambda=0
cfg.pdelta=0

# relation reasoning method
# option：mlp,dotproduct
cfg.relation_reasoning_method="mlp"
cfg.solver = 'sgd' # sgd or adam

cfg.use_multi_gpu=True

cfg.using_high_level=False #with high_level_info or not

cfg.batch_size=16 #train batchsize
cfg.test_batch_size=32 #test batchsize
cfg.lambda_h = .5
cfg.lambda_g = .5
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.0
cfg.weight_decay=0.0
cfg.sgd_moment=0.0
# cfg.lr_plan={31:1e-4, 61:1e-5, 81:1e-6}
cfg.lr_plan={}
cfg.max_epoch=130

cfg.backbone="inception"
cfg.emb_features=1056


cfg.save_model_path='../model'

cfg.log_path='../log'
cfg.linename='{}_{}_model_gcn_bs{}_gn{}_mse{}_th{:.1f}'.format(dataset_name,cfg.backbone,cfg.batch_size,
                                                   cfg.num_graph,cfg.using_mse,
                                                            cfg.pos_threshold)

cfg.base_model_name='model_basemodel_bs16_inception_epoch8.pth'
cfg.base_model_path='../model/tvhi_model_basemodel_bs16_inception_epoch8.pth'
cfg.data_path=r'../highfive'
cfg.train_seqs=[]
cfg.test_seqs=[]

cfg.test_interval_epoch=2

# set test mode
cfg.is_validation=True
cfg.savedmodel_path='../model/tvhi_model_inception_model_gcn_bs16_gn1_mseTrue_th0.4_epoch128.pth'
if cfg.is_validation:
    cfg.linename+='_validation'

cfg.print_all_properties()

cfg.train_seqs,cfg.test_seqs=cfg.dataset_instance.get_train_test_seq()

test_net(cfg)

