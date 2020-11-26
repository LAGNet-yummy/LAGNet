"""
this module provides the datasets including training and testing datasets
"""
def return_dataset(cfg):

    #read dataset
    train_anns=cfg.dataset_instance.read_dataset(cfg.data_path, cfg.train_seqs)
    train_frames=cfg.dataset_instance.all_frames(train_anns)

    test_anns=cfg.dataset_instance.read_dataset(cfg.data_path, cfg.test_seqs)
    test_frames=cfg.dataset_instance.all_frames(test_anns)

    #create dataset
    training_set=cfg.dataset_instance.IDataset(train_anns,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,num_boxes=cfg.num_boxes,
                                       num_frames=cfg.num_frames,is_training=True,is_finetune=(cfg.training_stage==1),data_augmentation=True)

    validation_set=cfg.dataset_instance.IDataset(test_anns,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,num_boxes=cfg.num_boxes,
                                         num_frames=cfg.num_frames,is_training=False,is_finetune=(cfg.training_stage==1),data_augmentation=False)
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    
    return training_set, validation_set

def return_dataset_statistics(cfg):
    train_anns=cfg.dataset_instance.read_dataset(cfg.data_path, cfg.train_seqs) # a dict
    test_anns=cfg.dataset_instance.read_dataset(cfg.data_path, cfg.test_seqs)

    def return_set_statistics(ss):
        np = 0
        nf = 0
        for k, v in ss.items():
            for fk, fv in v.items():
                nf += 1
                np += fv['box_num']
        return(nf,np)

    nf_train, np_train = return_set_statistics(train_anns)
    nf_test, np_test = return_set_statistics(test_anns)

    return({'Average person number per frame':(np_train+np_test)/(nf_train+nf_test)})
    