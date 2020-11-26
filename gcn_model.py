# modified based on the source code of "Learning Actor
# Relation Graphs for Group Activity Recognition" by
# Wu et. al., CVPR 2019
#

from utils import *
from torchvision.ops import *
import numpy as np
from backbone import *
import torch


class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()

        self.cfg = cfg

        NFR = cfg.num_features_relation

        NFG = cfg.num_features_gcn

        NG = cfg.num_graph

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG]) for i in range(NG)])

    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat.clone()  # B*T*N, 4
        graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(N, 2)  # B*T, N, 2

        graph_boxes_distances = calc_pairwise_distance(graph_boxes_positions, graph_boxes_positions)  #N, N

        position_mask = (graph_boxes_distances > (pos_threshold * OW))# horizon orient

        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # N,NFR


            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(0, 1))  # N,N

            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(N, N)

            relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=1)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](torch.matmul(relation_graph, graph_boxes_features))  # N, NFG
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list), dim=0)  # N, NFG

        return graph_boxes_features, relation_graph

    def stateDict(self):
        state={
            'fc_rn_theta_list':self.fc_rn_theta_list.state_dict(),
            'fc_rn_phi_list':self.fc_rn_phi_list.state_dict(),
            'fc_gcn_list':self.fc_gcn_list.state_dict(),
            'nl_gcn_list':self.nl_gcn_list.state_dict()
        }
        return state

    def loadmodel(self, state):
        self.fc_rn_theta_list.load_state_dict(state['fc_rn_theta_list'])
        self.fc_rn_phi_list.load_state_dict(state['fc_rn_phi_list'])
        self.fc_gcn_list.load_state_dict(state['fc_gcn_list'])
        self.nl_gcn_list.load_state_dict(state['nl_gcn_list'])

class GCNet(nn.Module):

    def __init__(self, cfg):
        super(GCNet, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features  # emb_features=1056   #output feature map channel of backbone
        K = self.cfg.crop_size[0]  # crop_size = 5, 5  #crop size of roi align
        NFB = self.cfg.num_features_boxes  # num_features_boxes = 1024

        self.backbone = get_backbone(cfg.backbone)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_actions = nn.Linear(NFB, 128)
        self.fc_actions_final=nn.Linear(128,cfg.num_actions)

        self.nl_fc_actions=nn.LayerNorm([128])
        self.nl_emb_1=nn.LayerNorm([NFB])

        self.gcn=GCN_Module(cfg)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if cfg.is_validation:
            self.load_savedmodel(cfg.savedmodel_path)
        else:self.loadbasemodel(cfg.base_model_path)


    def savemodel(self, filepath):
        state = {
            'backbone': self.backbone.state_dict(),
            'fc_emb_1': self.fc_emb_1.state_dict(),
            'fc_actions': self.fc_actions.state_dict(),
            'fc_actions_final':self.fc_actions_final.state_dict(),
            'gcn':self.gcn.stateDict(),
            'nl_emb_1':self.nl_emb_1.state_dict(),
            'nl_fc_actions':self.nl_fc_actions.state_dict(),
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadbasemodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone'])
        self.fc_emb_1.load_state_dict(state['fc_emb_1'])

    def load_savedmodel(self,filepath):
        self.loadmodel(filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.gcn.loadmodel(state['gcn'])
        self.backbone.load_state_dict(state['backbone'])
        self.fc_emb_1.load_state_dict(state['fc_emb_1'])
        self.fc_actions.load_state_dict(state['fc_actions'])
        self.fc_actions_final.load_state_dict(state['fc_actions_final'])

        self.nl_emb_1.load_state_dict(state['nl_emb_1'])
        self.nl_fc_actions.load_state_dict(state['nl_fc_actions'])

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (
        B * T, 3, H, W))  # eq：torch.Size([16, 1, 3, 480, 720])->torch.Size([16, 3, 480, 720])

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(
            images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear',
                                         align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale,dim=1)

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))

        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        boxes_idx_flat = boxes_idx_flat.float()
        boxes_idx_flat = torch.reshape(boxes_idx_flat, (-1, 1))

        # RoI Align       boxes_features_all：eq：torch.Size([195, 1056, 5, 5])
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False

        boxes_features_all = roi_align(features_multiscale,  # eq：torch.Size([15, 1056, 57, 87])
                                       torch.cat((boxes_idx_flat, boxes_in_flat), 1),  # eq：torch.Size([195, 5])
                                       (K, K))

        boxes_features_all = boxes_features_all.reshape(B * T, MAX_N,
                                                        -1)  # B*T,MAX_N, D*K*K #eq：torch.Size([15, 13, 26400])
        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)
        # B*T,MAX_N,NFB
        boxes_features_all = self.dropout_emb_1(boxes_features_all)

        actions_scores = []
        A_list=[]


        bboxes_num_in = bboxes_num_in.reshape(B * T, )  # B*T,
        boxes_in=boxes_in.reshape((B*T,self.cfg.num_boxes,4))
        OH, OW = self.cfg.out_size
        for bt in range(B * T):
            # handle one frame
            N = bboxes_num_in[bt]
            boxes_features = boxes_features_all[bt, :N, :].reshape(1, N, NFB)  # 1,N,NFB
            boxes_states = boxes_features

            NFS = NFB

            # Predict actions
            boxes_states_flat = boxes_states.reshape(-1, NFS)  # 1*N, NFS
            actn_score,A=self.gcn(boxes_states_flat,boxes_in[bt,:N])
            actn_score += boxes_features.squeeze()
            actn_score = self.fc_actions(actn_score)  # 1*N, 128
            actn_score=self.nl_fc_actions(actn_score)
            actn_score=F.relu(actn_score)

            actn_score=self.fc_actions_final(actn_score)

            actions_scores.append(actn_score)

            G=A.clone()
            G=packMat(G)
            A_list.append(G)

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        A_list=torch.cat(A_list,dim=0)

        return actions_scores,A_list


