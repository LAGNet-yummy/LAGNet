import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torch.utils.data as data
from sklearn.metrics import  precision_recall_fscore_support
import os

from base_model import Basenet
from gcn_model import GCNet
from lagnet import LAGNet
from utils import *
from loss import L3,L4,inter_mse
from dataset import return_dataset


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
           
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def test_net(cfg):
    """
    training net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Reading dataset
    _, validation_set=return_dataset(cfg)

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        #'num_workers': 4
        'num_workers': 0
    }

    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda') #get cuda device
    else:
        device = torch.device('cpu')

    if cfg.training_stage==1:
        model=Basenet(cfg)
    elif cfg.training_stage==5:
        model=GCNet(cfg)
    elif cfg.training_stage==6:
        model=LAGNet(cfg)

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model) 

    model=model.cuda()
    model.apply(set_bn_eval)

    if 'is_validation' in cfg.__dict__ and cfg.is_validation==True:
        print('begin test')
        test_info = test(validation_loader, model, device, None, 'test', cfg)
        save_log(cfg,test_info)
        print('end test')
        exit(0)


def test(data_loader, model, device, optimizer, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    interactions_meter=AverageMeter()
    loss_meter=AverageMeter()
    actions_classification_labels=[0 for i in range(cfg.num_actions)]
    actions_classification_pred_true=[0 for i in range(cfg.num_actions)]
    interactions_classification_labels=[0,0]
    interactions_classification_pred_true=[0,0]

    actions_pred_global=[]
    actions_labels_global=[]
    interactions_pred_global=[]
    interactions_labels_global=[]
    
    result = {}  # save the predicted results
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            seq_name, fid = batch_data[-2], batch_data[-1]
            batch_data=[b.to(device=device) for b in batch_data[0:-2]]
            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
            interactions_in=batch_data[4].reshape((batch_size,num_frames,cfg.num_boxes*(cfg.num_boxes-1)))

            bboxes_num=batch_data[3].reshape(batch_size,num_frames)

            # forward
            if cfg.training_stage == 1 :
                actions_scores, interactions_scores = model((batch_data[0], batch_data[1], batch_data[3]))
            elif cfg.training_stage == 5:
                actions_scores, relation_mat = model((batch_data[0], batch_data[1], batch_data[3]))
            elif cfg.training_stage == 6:
                actions_scores, interactions_scores, relation_mat = model((batch_data[0], batch_data[1], batch_data[3]))

            actions_in_nopad=[]
            interactions_in_nopad=[]
            
            actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            interactions_in=interactions_in.reshape((batch_size*num_frames,cfg.num_boxes*(cfg.num_boxes-1),))
            bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])
                interactions_in_nopad.append(interactions_in[bt,:N*(N-1)])

            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            interactions_in=torch.cat(interactions_in_nopad,dim=0).reshape(-1,)

            if cfg.action_weight != None:
                aweight = torch.Tensor(cfg.action_weight).cuda()
            else:
                aweight = None
            if cfg.inter_weight != None:
                iweight = torch.Tensor(cfg.inter_weight).cuda()
            else:
                iweight = None

            actions_loss=F.cross_entropy(actions_scores,actions_in,weight=aweight)
            if cfg.training_stage!=5:
                interactions_loss=F.cross_entropy(interactions_scores,interactions_in,weight=iweight)
            else:
                if cfg.using_mse:
                    interactions_loss=inter_mse(relation_mat,interactions_in,bboxes_num)
                else:interactions_loss=0
            l3 = l4 = 0.0
            if cfg.plambda > 0:
                l3 = L3(interactions_scores,bboxes_num)
            if cfg.pdelta > 0:
                l4 = L4(relation_mat, interactions_in,bboxes_num)

            actions_pred=torch.argmax(actions_scores,dim=1)  #ALL_N,
            actions_correct=torch.sum(torch.eq(actions_pred.int(),actions_in.int()).float())

            if cfg.training_stage==5:
                relation_mat[relation_mat>=cfg.inter_threshold]=1
                relation_mat[relation_mat<cfg.inter_threshold]=0
                interactions_pred=relation_mat.long()
            else:interactions_pred=torch.argmax(interactions_scores,dim=1)
            interactions_correct=torch.sum(torch.eq(interactions_pred.int(),interactions_in.int()).float())

            # output result
            ahead,ihead=0,0
            if 'output_result' in cfg.__dict__ and cfg.output_result == True:
                for i in range(batch_size):
                    n=bboxes_num[i]
                    frame_key='{}_{}'.format(seq_name[i],fid[i])
                    d={
                        'action_gd':actions_in[ahead:ahead+n].cpu().numpy().tolist(),
                        'interaction_gd':interactions_in[ihead:ihead+n*(n-1)].cpu().numpy().tolist(),
                        'action_score':actions_scores[ahead:ahead+n,:].cpu().numpy().tolist(),
                        'interaction_score':interactions_scores[ihead:ihead+n*(n-1),:].cpu().numpy().tolist()
                    }
                    result[frame_key]=d
                    ahead+=n
                    ihead+=n*(n-1)
            actions_pred_global.append(actions_pred.cpu())
            actions_labels_global.append(actions_in.cpu())
            interactions_pred_global.append(interactions_pred.cpu())
            interactions_labels_global.append(interactions_in.cpu())

            # recall
            for i in range(len(actions_pred)):
                actions_classification_labels[actions_in[i]]+=1
                if actions_pred[i]==actions_in[i]:
                    actions_classification_pred_true[actions_pred[i]]+=1
            for i in range(len(interactions_pred)):
                interactions_classification_labels[interactions_in[i]]+=1
                if interactions_pred[i]==interactions_in[i]:
                    interactions_classification_pred_true[interactions_pred[i]]+=1
            # Get accuracy
            actions_accuracy=actions_correct.item()/actions_scores.shape[0]
            interactions_accuracy=interactions_correct.item()/interactions_in.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            interactions_meter.update(interactions_accuracy, interactions_in.shape[0])

            # Total loss
            total_loss=actions_loss+interactions_loss+cfg.plambda*l3+cfg.pdelta*l4
            loss_meter.update(total_loss.item(), batch_size)

    if 'output_result' in cfg.__dict__ and cfg.output_result == True:
        with open(cfg.result_path+epoch+'.json','w') as f:
            json.dump(result,f)

    for i in range(len(actions_classification_labels)):
        actions_classification_pred_true[i]=actions_classification_pred_true[i]*1.0/actions_classification_labels[i]
    for i in range(len(interactions_classification_labels)):
        interactions_classification_pred_true[i]=interactions_classification_pred_true[i]*1.0/interactions_classification_labels[i]

    actions_pred_global=torch.cat(actions_pred_global)
    actions_labels_global=torch.cat(actions_labels_global)
    interactions_pred_global=torch.cat(interactions_pred_global)
    interactions_labels_global=torch.cat(interactions_labels_global)

    # calculate mean IOU
    cls_iou=torch.Tensor([0 for _ in range(cfg.num_actions+2)]).cuda().float()
    for i in range(cfg.num_actions):
        grd=set((actions_labels_global==i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        prd=set((actions_pred_global==i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        uset=grd.union(prd)
        iset=grd.intersection(prd)
        cls_iou[i]=len(iset)/len(uset)
    for i in range(2):
        grd=set((interactions_labels_global==i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        prd=set((interactions_pred_global==i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        uset=grd.union(prd)
        iset=grd.intersection(prd)
        cls_iou[cfg.num_actions+i]=len(iset)/len(uset)
    mean_iou=cls_iou.mean()

    actions_precision,actions_recall,actions_F1,support=precision_recall_fscore_support(actions_labels_global,actions_pred_global,beta=1,average='macro')
    interactions_precision,interactions_recall,interactions_F1,support=precision_recall_fscore_support(interactions_labels_global,interactions_pred_global,beta=1,average='macro')

    test_info={
        'epoch':epoch,
        'loss':loss_meter.avg,
        'actions_precision':actions_precision,
        'actions_recall':actions_recall,
        'actions_F1':actions_F1,
        'actions_acc':actions_meter.avg*100,
        'actions_classification_recalls':actions_classification_pred_true,
        'interactions_precision':interactions_precision,
        'interactions_recall':interactions_recall,
        'interactions_F1':interactions_F1,
        'interactions_acc':interactions_meter.avg*100,
        'interactions_classification_recalls':interactions_classification_pred_true,
        'mean_iou': mean_iou
    }

    return test_info
