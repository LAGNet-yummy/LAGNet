import sys
sys.path.append("..")
from test_net import test_net
from randomseed import setSeed
from config import Config
setSeed(0)

dataset_name='bit'
cfg=Config(dataset_name) # ci or bit
cfg.device_list="0,1,2" #一般只用一个
cfg.training_stage=1 #basemodel 模式

# loss parameter
cfg.plambda=0.
cfg.pdelta=0.

cfg.solver = 'adam'
cfg.use_multi_gpu=True

cfg.using_high_level=True #使用high_level_info

cfg.batch_size=24 #训练的batchsize
cfg.test_batch_size=32 #测试的batchsize
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
# cfg.lr_plan={30:5e-6, 50:1e-6}
cfg.max_epoch=70
cfg.sgd_moment=0.0

cfg.backbone="inception"
cfg.emb_features=1056


cfg.log_path='../log'
cfg.save_model_path='../model'

cfg.linename='{}_basemodel_bs{}_{}'.format(dataset_name,cfg.batch_size,cfg.backbone)


cfg.data_path='../BIT'

cfg.test_interval_epoch=2

# set test mode
cfg.is_validation=True
cfg.savedmodel_path='../model/bit_model_basemodel_bs24_inception_epoch14.pth'
if cfg.is_validation:
    cfg.linename+='_validation'

cfg.output_result=False
if cfg.output_result:
    cfg.result_path='../log/'
    cfg.linename +='_result'

cfg.print_all_properties()
cfg.train_seqs, cfg.test_seqs = cfg.dataset_instance.get_train_test_seq()

test_net(cfg)










