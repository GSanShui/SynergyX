import torch
import torch.nn as nn
from .head import FusionHead
from .model_utlis import Embeddings, Encoder, EncoderCA, EncoderCell, EncoderCellCA, EncoderD2C, EncoderSSA, EncoderCellSSA
import numpy as np



class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size=[2,2,6]
        drop_rate=0.2
        kernel_size=[16,16,16]

        if in_channel == 3:
            in_channels=[3,8,16]
            out_channels=[8,16,32]         

        elif in_channel == 6:
            in_channels=[6,16,32]
            out_channels=[16,32,64]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)


    def forward(self, x):

        # print('x_cell_embed.shape:',x_cell_embed.shape)
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)  # [batch, out_channel, 53]
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed) # [batch,53,64] or [batch,53,128]
        
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)


def drug_feat(drug_subs_codes, device, patch, length):
    v = drug_subs_codes
    subs = v[:, 0].long().to(device)
    subs_mask = v[:, 1].long().to(device)

    if patch > length:
        padding = torch.zeros(subs.size(0), patch - length).long().to(device)
        subs = torch.cat((subs, padding), 1)
        subs_mask = torch.cat((subs_mask, padding), 1)

    expanded_subs_mask = subs_mask.unsqueeze(1).unsqueeze(2)
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0

    return subs, expanded_subs_mask.float()



class SynergyxNet(torch.nn.Module):

    def __init__(self,
                 num_attention_heads = 8,
                 attention_probs_dropout_prob = 0.1, 
                 hidden_dropout_prob = 0.1,
                 max_length = 50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 args=None):
        super(SynergyxNet, self).__init__()

        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.max_length = max_length
        
        if args.celldataset == 0 :
            self.genes_nums = 697
        elif args.celldataset == 1:
            self.genes_nums = 18498
        elif args.celldataset == 2:
            self.genes_nums = 4079

        
        if self.args.cellencoder == 'cellTrans':
            self.patch = 50
            if self.in_channel == 3:
                feat_dim = 243
                hidden_size = 256
            elif self.in_channel == 6:
                feat_dim = 243*2
                hidden_size = 512
            self.cell_linear = nn.Linear(feat_dim, hidden_size)

        elif self.args.cellencoder == 'cellCNNTrans':
            self.patch = 165
            if self.in_channel == 3:
                hidden_size = 64
            elif self.in_channel == 6:
                hidden_size = 128 
            self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size, args=args)


        intermediate_size = hidden_size*2
        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)        
        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_SA = EncoderCell(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_CA = EncoderCA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_CA = EncoderCellCA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_cell_CA = EncoderD2C(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_SSA = EncoderSSA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_SSA = EncoderCellSSA(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)

        self.head = FusionHead()

        self.cell_fc = nn.Sequential(
                        nn.Linear(self.patch * hidden_size, output_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                    )

        self.drug_fc = nn.Sequential(
                        nn.Linear(self.patch * hidden_size, output_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                    )


    def forward(self, data):
        
        if self.args.mode == 'infer':
            batch_size = 1
        else:
            batch_size = self.args.batch_size  
        

        drugA = data.drugA
        drugB = data.drugB
        drugA, drugA_attention_mask = drug_feat(drugA, self.args.device, self.patch, self.max_length)
        drugB, drugB_attention_mask = drug_feat(drugB, self.args.device, self.patch, self.max_length)


        drugA = self.drug_emb(drugA)
        drugB = self.drug_emb(drugB)
        drugA = drugA.float()
        drugB = drugB.float() 

        x_cell = data.x_cell.type(torch.float32)
        x_cell = x_cell[:,[self.omic_dict[i] for i in self.include_omic]]  # [batch*4079,len(omics)]
        cellA = x_cell.view(batch_size, self.genes_nums, -1)
        cellB = cellA.clone()


        # if self.args.dgi == 0:
        #     dgiA = data.dgiA
        #     dgiB = data.dgiB
        #     dgiA = dgiA.view(batch_size, -1)[:,:,None].expand(batch_size, self.genes_nums, self.in_channel)
        #     dgiB = dgiB.view(batch_size, -1)[:,:,None].expand(batch_size, self.genes_nums, self.in_channel)
        #     cellA = cellA*dgiA
        #     cellB = cellB*dgiB

        if self.args.cellencoder == 'cellTrans': 
            gene_length = 4050
            cellA = cellA[:,:gene_length,:]
            cellA = cellA.view(batch_size, self.patch, -1, x_cell.size(-1)) 
            cellA = cellA.view(batch_size, self.patch, -1)
            cellA = self.cell_linear(cellA)
            cellB = cellB[:,:gene_length,:]
            cellB = cellB.view(batch_size, self.patch, -1, x_cell.size(-1)) 
            cellB = cellB.view(batch_size, self.patch, -1) 
            cellB = self.cell_linear(cellB)
        elif self.args.cellencoder == 'cellCNNTrans': 
            cellA = self.cell_conv(cellA) 
            cellB = self.cell_conv(cellB) 
        else:
             raise ValueError('Wrong cellencoder type!!!')


        # layer1
        cellA0=cellA
        cellA, drugA, attn9, attn10 = self.drug_cell_CA(cellA, drugA, drugA_attention_mask, None)
        cellB, drugB, attn11, attn12  = self.drug_cell_CA(cellB, drugB, drugB_attention_mask, None)
        cellA1=cellA
        cellB1=cellB
        # layer2
        drugA, attn5 = self.drug_SA(drugA, drugA_attention_mask)
        drugB, attn6 = self.drug_SA(drugB, drugB_attention_mask)
        cellA, attn7 = self.cell_SA(cellA, None)
        cellB, attn8 = self.cell_SA(cellB, None)
        cellA2=cellA
        cellB2=cellB
        # layer3
        drugA, drugB, attn1, attn2 = self.drug_CA(drugA, drugB, drugA_attention_mask, drugB_attention_mask)
        cellA, cellB, attn3, attn4 = self.cell_CA(cellA, cellB, None, None)
        cellA3=cellA
        cellB3=cellB


        drugA_embed = self.drug_fc(drugA.view(-1,drugA.shape[1]*drugA.shape[2]))
        drugB_embed = self.drug_fc(drugB.view(-1,drugB.shape[1]*drugB.shape[2]))
        cellA_embed = self.cell_fc(cellA.view(-1,cellA.shape[1]*cellA.shape[2]))
        cellB_embed = self.cell_fc(cellB.view(-1,drugA.shape[1]*cellB.shape[2]))

        cell_embed = torch.cat((cellA_embed,cellB_embed),1)
        drug_embed = torch.cat((drugA_embed,drugB_embed),1)
        output = self.head(cell_embed, drug_embed)


        all_cell_embed = None
        all_attn = None
        if self.args.output_attn:
            all_cell_embed = torch.stack((cellA1.mean(axis=1).flatten(),cellB1.mean(axis=1).flatten(),cellA2.mean(axis=1).flatten(),cellB2.mean(axis=1).flatten(),cellA3.mean(axis=1).flatten(),cellB3.mean(axis=1).flatten(),cellA0.mean(axis=1).flatten()),0)
            all_attn = torch.cat((attn9,attn10,attn11,attn12,attn5,attn6,attn7,attn8,attn1,attn2,attn3,attn4),0)  
 
        return output, all_cell_embed, all_attn


    def init_weights(self):

        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        if self.args.cellencoder == 'cellCNNTrans':
            self.cell_conv.init_weights()