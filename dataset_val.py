#!user/bin/env python
# -*- coding:utf-8 -*-
import collections
import pickle

from torch.utils.data import Dataset
import json
import numpy as np
import torch

from dataset import a_dic
from model import tokenizer
from config import args
from random import sample

with open('aokvqa_id_map.json', 'r') as f_six:
    id_map = json.load(f_six)

if args.dataset == 'krvqa':
    with open('data/kr-vqa/krvqa_test.json','r') as f:
         val_row = json.load(f)
    with open('data/kr-vqa/krvqa_img_feature_test.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
elif args.dataset == 'okvqa':
    with open('data/okvqa_val.json','r') as f:
        val_row = json.load(f)
    with open('data/vqa_img_feature_val.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
elif args.dataset == 'aokvqa':
    with open('data/aokvqa/aokvqa_val.json','r') as f:
        val_row = json.load(f)
    with open('data/aokvqa/aokvqa_img_feature_val.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
elif args.dataset == 'vqav2':
    with open('data/vqa_img_feature_test_dev.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    with open('data/vqa_test.json','r') as f:
        val_row = json.load(f)

def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sxo' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

image_ids = []
qids = []
questions = []
answers = []
labels = []
objects = []
answer_ids = []
answers_lists = []
question_lengths = []
most_answer = []
most_answer_ids = []
neg_answer = []



for qid, item in val_row.items():
    img_id = str(item['image_id'])
    image_ids.append(img_id)
    qids.append(qid)
    question_clean = item['question']  # + answer_sentence
    questions.append(question_clean)

    if args.dataset == 'okvqa' or args.dataset == 'aokvqa' or args.dataset == 'vqav2':
        answers.append(item['multi_answers'])
        m_ans_id = [a_dic.get(i, -1) for i in item['multi_answers']]
        most_answer_ids.append(m_ans_id)

        if args.dataset == 'okvqa':
            objects.append(item['label'])
    else:
        answers.append(item['answer'])
        most_ans_id = a_dic.get(item['answer'], -1)
        most_answer_ids.append([most_ans_id])


if args.glo_fea is True:
    if args.dataset == 'okvqa':
        with open('data/vqav2/'+'resnet_fea27.json') as f:
            vgg_feat=json.load(f)
            print(len(vgg_feat))

        with open('okvqa_val_imp.json') as f:
            kg = json.load(f)
            print(len(kg))

        with open('tools/' + 'cnn_fea_okvqa_val.json') as f:
            q_feat = json.load(f)
            print(len(q_feat))

        with open('data/ok-vqa/captions_5.json') as f:
            cap = json.load(f)
            print(len(cap))

    elif args.dataset == 'aokvqa':
        with open('data/aokvqa/' + 'res_val_aokvqa.json') as f:
            vgg_feat = json.load(f)
            print(len(vgg_feat))

        with open('data/aokvqa/' + 'cnn_val_aokvqa.json') as f:
            q_feat = json.load(f)
            print(len(q_feat))

        if args.cap:
            with open('data/aokvqa/promptcap_val_mukea.json') as f:
                cap = json.load(f)
                print(len(cap))
    else:
        with open('tools/' + 'resnet_featest_krvqa.json') as f:
            vgg_feat = json.load(f)
            print(len(vgg_feat))

        with open('data/kr-vqa/' + 'cnn_featest_krvqa.json') as f:
            q_feat = json.load(f)
            print(len(q_feat))

        with open('data/kr-vqa/' + 'dic_img_test.json') as f:
            img_ids= json.load(f)
            print(len(img_ids))

        with open('data/kr-vqa/' + 'VG_100Kvg100_caption.json') as f:
            cap = json.load(f)
            print(len(cap))
        with open('data/kr-vqa/' + 'kb_id.json') as f:
            kbid = json.load(f)
            print(len(kbid))

class KgDatasetVal(Dataset):
    def __init__(self, val=False, val_test=False):
        self.image_ids = image_ids
        self.qids = qids
        self.questions = questions
        self.answers = answers
        self.most_answer_ids = most_answer_ids

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid = self.qids[index]
        question = self.questions[index]
        answer = self.answers[index]
        image_feature = pretrain_feature[self.image_ids[index]]['feats']
        spatial_feature = pretrain_feature[self.image_ids[index]]['sp_feats']
        most_id = self.most_answer_ids[index]
        if args.dataset == 'aokvqa':
            most_id.append(int(id_map[qid]))
        else:
            try:
                most_id.append(int(qid))
                qid=qid
            except:
                most_id.append(int(qid[0:len(qid) - 1]))
                qid=qid[0:len(qid) - 1]


        if args.glo_fea is True:
            if args.dataset == 'okvqa':
                # Get vgg_feat info
                img_id = 1000000000000 + int(self.image_ids[index])
                image = "COCO_train2014_" + str(img_id)[1:13]
                image0 = "COCO_val2014_" + str(img_id)[1:13]
                try:
                    # vgg_feats = np.array(vgg_feat[image][0])
                    vgg_feats = np.array(vgg_feat[image0][0])
                except:
                    vgg_feats = np.array(vgg_feat[image][0])

                q_feats = np.array(q_feat[str(qid)])

                if args.cap:
                    caption=cap[str(self.image_ids[index])]
                    if args.pretrain is False:
                        if args.kg is True:
                            im_kg=kg[str(qid)]
                            caption =  im_kg+caption
                        else:
                            caption = caption
                    return int(qid), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats,caption
                else:
                    return int(qid), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats
            elif args.dataset == 'aokvqa':
                # Get vgg_feat info
                img_id = 1000000000000 + int(self.image_ids[index])
                image = str(img_id)[1:13]
                vgg_feats = np.array(vgg_feat[image][0])

                q_feats = np.array(q_feat[str(qid)])

                if args.cap:
                    caption = cap[qid]
                    if args.pretrain is False:
                        if args.kg is True:
                            im_kg=kg[str(qid)]
                            caption =  im_kg+caption
                        else:
                            caption = caption

                    return int(id_map[qid]), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats,caption
                else:
                    return int(id_map[qid]), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats
            else:
                img_id0 = str(img_ids[str(qid)])
                vgg_feats = np.array(torch.tensor(vgg_feat[img_id0]).squeeze())
                q_feats = np.array(q_feat[str(qid)])
                try:
                    if kbid[str(qid)]==1:
                        caption = cap[str(self.image_ids[index])]
                    else:
                        caption ='0'
                except:
                    caption ='0'
                return int(qid), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats,caption
        else:
            return int(qid), question, answer, image_feature, spatial_feature, most_id


