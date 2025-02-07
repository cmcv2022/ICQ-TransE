#!user/bin/env python
# -*- coding:utf-8 -*-

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import LxmertConfig, LxmertTokenizer, LxmertModel
from lxrt.modeling import GeLU, BertLayerNorm
from lxrt.entry import LXRTEncoder
from config import args
# from attention import MultiHeadAttention, attention
from okvqa.gumbel_softmax import gumbel_softmax

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')




class KgPreModel(nn.Module):
    def __init__(self, vocab_num):
        super(KgPreModel, self).__init__()
        self.lx = LXRTEncoder(
            args, max_seq_length=23
        )
        self.lx.load(args.load_lxmert)

        self.vocab_num = vocab_num
        self.PreLayer = self.lx
        # self.PreLayer = model
        self.linear_vision = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))
        self.linear_300 = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))
        self.linear_classifytask = nn.Linear(300, 1024)  
        self.tail_decode = nn.Embedding(vocab_num, 300)
        init.uniform_(self.tail_decode.weight.data)
        # self.sa = MultiHeadAttention(8, 768)
        self.v_att_proj = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        # self.S = nn.Linear(768, 1024)

        self.w = nn.Parameter(torch.ones(2))

    def forward(self, i, v):
        bert_output=self.PreLayer(i, v)
        language_output = bert_output[0][0]
        vision_output = bert_output[0][1]
        cls=bert_output[1]
        qid=bert_output[2]

        sum_vision = self.linear_vision(cls)

        # affinity matrix
        l_att = self.l_att_proj(language_output)
        v_att = self.v_att_proj(vision_output)
        sim_matrix_v2l = torch.matmul(v_att, l_att.transpose(1,2))  # b * v_length * l_length
        kg_output, k = torch.topk(sim_matrix_v2l, dim=-1, k=1)


        if args.hard:
            # hard attention
            hard_attention_value = gumbel_softmax(kg_output.squeeze())
            kg_output0=hard_attention_value
            head = (vision_output * hard_attention_value.unsqueeze(-1)).sum(-2)
        else:
            # soft attention
            kg_output = F.softmax(kg_output.squeeze(), dim=-1)

            vv = torch.zeros_like(kg_output)
            for i in range(len(kg_output)):

                 v=torch.sort(kg_output[i],descending=True)
                 thre=0
                 index = []
                 for ii in range(len(v[0])):
                     if thre<args.val:
                          thre=thre+v[0][ii]
                          index.append(v[1][ii])
                     else:
                          break

                 vi = index[0:len(index)]
                 for kk in vi:
                     vv[i][kk] = 1
            kg_output0=torch.mul(vv,kg_output)
            head = (vision_output * kg_output0.unsqueeze(-1)).sum(-2)

        head_300 = self.linear_300(head)
        anchor=head_300+sum_vision


        return anchor, kg_output0,qid,sim_matrix_v2l

    def decode_tail(self, most):
        most = self.tail_decode(most).squeeze()

        return most.squeeze()

    def cal_sim(self, anchor, most):
        sim_out = anchor.mm(most.t())

        return sim_out.squeeze()

    def threshold(self,x):
        if x>0.5:
            x=x
        else:
            x=0
        return x