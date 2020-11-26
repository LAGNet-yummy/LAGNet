import sys
sys.path.append("..")
from randomseed import setSeed
setSeed(10)
from test_net import test_net
from config import Config

dataset_name='tvhi'
cfg=Config(dataset_name)
cfg.data_path='../highfive' #root directory of dataset
cfg.device_list="0,1,2" #device list
cfg.training_stage=1 #basemodel mode

cfg.plambda=0
cfg.pdelta=0

cfg.use_multi_gpu=True
cfg.crop_size = (5,5)

cfg.solver = 'sgd'

cfg.backbone="inception"
cfg.emb_features=1056

cfg.batch_size=16 #train batchsize
cfg.test_batch_size=32 #test batchsize
cfg.train_learning_rate=1e-3
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.sgd_moment=0.0
# cfg.lr_plan={}
cfg.lr_plan={31:1e-4,61:1e-5, 81:1e-6}
cfg.max_epoch=100

cfg.test_interval_epoch=2

cfg.log_path='../log'
cfg.save_model_path='../model'
cfg.linename='{}_basemodel_bs{}_{}'.format(dataset_name,cfg.batch_size,cfg.backbone)


# set test mode
cfg.is_validation=True
cfg.savedmodel_path='../model/tvhi_model_basemodel_bs16_inception_epoch8.pth'
if cfg.is_validation:
    cfg.linename+='_validation'

cfg.output_result=False
if cfg.output_result:
    cfg.result_path='../log/'
    cfg.linename +='_result'

cfg.print_all_properties()

cfg.train_seqs,cfg.test_seqs=cfg.dataset_instance.get_train_test_seq()

test_net(cfg)

