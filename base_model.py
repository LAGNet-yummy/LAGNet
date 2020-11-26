import torch
import torch.nn as nn
import torch.nn.functional as F 
from backbone import get_backbone
from utils import *
from torchvision.ops import *

from meanfield import *

class Basenet(nn.Module):
    """
    main module of base model
    """
    def __init__(self, cfg):
        super(Basenet, self).__init__()
        self.cfg=cfg

        D=self.cfg.emb_features #emb_features=1056   #output feature map channel of backbone
        K=self.cfg.crop_size[0] #crop_size = 5, 5  #crop size of roi align
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024

        self.backbone = get_backbone(cfg.backbone)

        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        self.fc_actions=nn.Linear(NFB,self.cfg.num_actions)
        self.fc_interactions1=nn.Linear(2*NFB,128)
        self.fc_interactions2=nn.Linear(128,2)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        self.nl_action = nn.LayerNorm(self.cfg.num_actions)
        self.nl_interaction=nn.LayerNorm(2)

        if cfg.is_validation:
            self.load_savedmodel(cfg.savedmodel_path)

    def savemodel(self,filepath):
        state = {
            'backbone': self.backbone.state_dict(),
            'fc_emb_1':self.fc_emb_1.state_dict(),
            'fc_actions':self.fc_actions.state_dict(),
            'fc_interactions1':self.fc_interactions1.state_dict(),
            'fc_interactions2':self.fc_interactions2.state_dict(),
            'nl_action':self.nl_action.state_dict(),
            'nl_interaction':self.nl_interaction.state_dict()
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)

    def load_pretrained_basemodel(self,filepath):
        state=torch.load(filepath)
        self.backbone.load_state_dict(state['backbone'])
        self.fc_emb_1.load_state_dict(state['fc_emb_1'])
        self.fc_actions.load_state_dict(state['fc_actions'])
        self.fc_interactions1.load_state_dict(state['fc_interactions1'])
        self.fc_interactions2.load_state_dict(state['fc_interactions2'])

    def loadmodel(self,filepath):
        state=torch.load(filepath)
        self.backbone.load_state_dict(state['backbone'])
        self.fc_emb_1.load_state_dict(state['fc_emb_1'])
        self.fc_actions.load_state_dict(state['fc_actions'])
        self.fc_interactions1.load_state_dict(state['fc_interactions1'])
        self.fc_interactions2.load_state_dict(state['fc_interactions2'])

    def load_savedmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone'])
        self.fc_emb_1.load_state_dict(state['fc_emb_1'])
        self.fc_actions.load_state_dict(state['fc_actions'])
        self.fc_interactions1.load_state_dict(state['fc_interactions1'])
        self.fc_interactions2.load_state_dict(state['fc_interactions2'])
        if self.cfg.dataset_name in ['tvhi','bit']:
            self.nl_action.load_state_dict(state['nl_action'])
            self.nl_interaction.load_state_dict(state['nl_interaction'])

    def forward(self,batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #eq：torch.Size([16, 1, 3, 480, 720])->torch.Size([16, 3, 480, 720])
                
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat) # pass through backbone before applying RoiAlign
            
    
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)
        
        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,

        boxes_idx_flat=boxes_idx_flat.float()
        boxes_idx_flat=torch.reshape(boxes_idx_flat,(-1,1))

        boxes_in_flat.requires_grad=False #set False
        boxes_idx_flat.requires_grad=False
        boxes_features_all=roi_align(features_multiscale,#eq：torch.Size([15, 1056, 57, 87])
                                            torch.cat((boxes_idx_flat,boxes_in_flat),1),#eq：torch.Size([195, 5])
                                            (5,5)) 
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K

        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)
        boxes_features_all=F.relu(boxes_features_all)
        # B*T,MAX_N,NFB
        boxes_features_all=self.dropout_emb_1(boxes_features_all)
        
    
        actions_scores=[]
        interaction_scores=[]

        edgeEmbMat=[]

        bboxes_num_in=bboxes_num_in.reshape(B*T,)  #B*T,
        for bt in range(B*T):
            # handle one frame
            N=bboxes_num_in[bt]
            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB
    
            boxes_states=boxes_features  

            NFS=NFB

            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            actn_score=self.fc_actions(boxes_states_flat)  #1*N, actn_num

            # Predict interactions
            interaction_flat=[]
            for i in range(N):
                for j in range(N):
                    if i!=j:
                        # concatenate features of two nodes
                        interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j]],dim=0))
            interaction_flat=torch.stack(interaction_flat,dim=0) #N(N-1),2048
            interaction_flat=self.fc_interactions1(interaction_flat)  #N(N-1), 128
            interaction_flat=F.relu(interaction_flat)

            edgeEmbMat.append(interaction_flat.clone())
            interaction_score=self.fc_interactions2(interaction_flat) #N(N-1), 2


            actions_scores.append(actn_score)
            interaction_scores.append(interaction_score)

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        interaction_scores=torch.cat(interaction_scores,dim=0)

        return actions_scores,interaction_scores
        