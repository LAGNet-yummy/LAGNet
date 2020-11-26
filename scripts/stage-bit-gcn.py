import sys
sys.path.append("..")
from test_net import test_net
from randomseed import setSeed
from config import Config

setSeed(0)

dataset_name='bit'
cfg=Config(dataset_name)
cfg.device_list="0,1,2"
cfg.message_times=1 #1,2,3,4,5

cfg.training_stage=5 #5:gcn模式

cfg.solver = 'adam' # sgd or adam

cfg.pos_threshold=0.7 # gcn distance threshhold
cfg.num_features_gcn=1024
cfg.inter_threshold=0.5
cfg.using_mse=True

# loss parameter
cfg.plambda=0.
cfg.pdelta=0.

# relation reasoning method
# option：mlp,dotproduct
cfg.train_backbone = False
cfg.pretrained_model = 'base'  # base or gnn

cfg.use_multi_gpu=True


cfg.using_high_level=False #使用high_level_info
cfg.batch_size=24 #训练的batchsize
cfg.test_batch_size=32 #测试的batchsize
cfg.lambda_h = .5
cfg.lambda_g = .5
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

cfg.linename='{}_{}_model_gcn_bs{}_gn{}_mse{}_th{:.1f}'.format(dataset_name,cfg.backbone,cfg.batch_size,
                                                   cfg.num_graph,cfg.using_mse,
                                                            cfg.pos_threshold)


cfg.base_model_name='model_basemodel_bs24_inception_epoch14.pth'
cfg.data_path='../BIT' #存着不同的信息，比如有anntaion文件夹、frame文件夹
cfg.base_model_path='../model/model_basemodel_bs24_inception_epoch14.pth'
cfg.model_path=''
cfg.test_interval_epoch=2

cfg.is_validation=True
cfg.savedmodel_path='../model/bit_model_inception_model_gcn_bs24_gn1_mseTrue_th0.7_epoch4.pth'
if cfg.is_validation:
    cfg.linename+='_validation'

cfg.print_all_properties()
cfg.train_seqs, cfg.test_seqs = cfg.dataset_instance.get_train_test_seq()

test_net(cfg)
