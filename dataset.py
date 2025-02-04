#!user/bin/env python
# -*- coding:utf-8 -*-
import collections
import json
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from config import args
from model import tokenizer
from random import sample
# from tools.cnn_text.main import glo_embedding
# from tools.resnet101 import glo_fea

with open('aokvqa_id_map.json', 'r') as f_six:
    id_map = json.load(f_six)

if args.dataset == 'krvqa':
    if args.pretrain:
        with open('data/vqa_train_filter.json','r') as f:
            vqa2 = json.load(f)
        train_row = vqa2
        with open('data/vqa_img_feature_train.pickle', 'rb') as f:
            pretrain_feature = pickle.load(f)
    else:
        with open('data/kr-vqa/krvqa_img_feature_train.pickle', 'rb') as f:
            pretrain_feature = pickle.load(f)
        with open('data/kr-vqa/krvqa_train.json','r') as f:
             train_row = json.load(f)
    if args.accumulate:
        with open('data/krvqa-pretrain_dic_all_filter.pickle', 'rb') as f:
            a_dic = pickle.load(f)
    else:
        with open('data/kr-vqa/krvqa-ans_dic.pickle', 'rb') as f:
            a_dic = pickle.load(f)
elif args.dataset == 'okvqa':
    with open('data/vqa_img_feature_train.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    if args.pretrain:
        with open('data/vqa_train_filter.json','r') as f:
            vqa2 = json.load(f)
        train_row = vqa2
    else:
        with open('data/okvqa_train.json','r') as f:
            train_row = json.load(f)
    if args.accumulate:
        with open('data/pretrain_dic_all_filter.pickle', 'rb') as f:
            a_dic = pickle.load(f)
    else:
        with open('data/ans_dic.pickle', 'rb') as f:
            a_dic = pickle.load(f)
elif args.dataset == 'aokvqa':
    if args.pretrain:
        with open('data/vqa_train_filter.json', 'r') as f:
            vqa2 = json.load(f)
        train_row = vqa2
        with open('data/vqa_img_feature_train.pickle', 'rb') as f:
            pretrain_feature = pickle.load(f)
    else:
        with open('data/aokvqa/aokvqa_img_feature_train.pickle', 'rb') as f:
            pretrain_feature = pickle.load(f)
        with open('data/aokvqa/aokvqa_train.json','r') as f:
            train_row = json.load(f)
    if args.accumulate:
        with open('data/aokvqa/pretrain_dic_all_filter.pickle', 'rb') as f:
            a_dic = pickle.load(f)
    else:
        with open('data/aokvqa/ans_dic_raw_aokvqa.pickle', 'rb') as f:
            a_dic = pickle.load(f)

elif args.dataset == 'vqav2':
    with open('data/vqa_img_feature_train.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    with open('data/vqa_train.json','r') as f:
        train_row = json.load(f)
    with open('data/vqav2/vqav2_dic_all.pickle', 'rb') as f:
        a_dic = pickle.load(f)
    with open('data/vqa_img_feature_val.pickle', 'rb') as f:
        pretrain_feature_val = pickle.load(f)
    with open('data/vqa_val.json','r') as f:
        val_row = json.load(f)
    pretrain_feature.update(pretrain_feature_val)
    train_row.update(val_row)


vocab_num = len(a_dic)
ans_all_list = a_dic.keys()
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
answers_most = []
most_answer_ids = []
neg_answer = []

n = 0


for qid, item in train_row.items():
    img_id = str(item['image_id'])
    image_ids.append(img_id)
    qids.append(qid)
    question_clean = item['question']# + answer_sentence
    questions.append(question_clean)

    # multi-answer
    if args.dataset == 'okvqa':
        answers.append(item['multi_answers'])
        m_ans_id = [a_dic.get(i, 0) for i in item['multi_answers']]
        most_answer_ids.append(m_ans_id)
    # most_answer.append(answer_embedding[0])
    elif args.dataset == 'aokvqa':
        answers.append(item['multi_answers'])
        m_ans_id = [a_dic.get(i, 0) for i in item['multi_answers']]
        most_answer_ids.append(m_ans_id)
    #single answer
    else:
        answers.append(item['answer'])
        most_ans_id = a_dic.get(item['answer'], 0)
        most_answer_ids.append([most_ans_id])


if args.glo_fea is True:
    #load q_feat
    if args.pretrain:
        with open('data/vqav2/' + 'vqa_train_filter_vggf.json') as f:
            vgg_feat = json.load(f)
            print(len(vgg_feat))

        with open('data/vqav2/' + 'vqa_train_filter_qf.json') as f:
            q_feat = json.load(f)
            print(len(q_feat))
        print("--------------------")

        # if args.cap:
        with open('data/ok-vqa/captions_5.json') as f:
            cap = json.load(f)
            print(len(cap))

    else:
        if args.dataset == 'okvqa':
            with open('data/vqav2/' + 'resnet_fea27.json') as f:
                vgg_feat = json.load(f)
                print(len(vgg_feat))

            with open('tools/' + 'cnn_fea_okvqa_train.json') as f:
                q_feat = json.load(f)
                print(len(q_feat))
            with open('okvqa_train_imp.json') as f:
                kg = json.load(f)
                print(len(kg))

            # if args.cap:
            with open('data/ok-vqa/captions_5.json') as f:
                cap = json.load(f)
                print(len(cap))

        elif args.dataset == 'aokvqa':
            with open('data/aokvqa/' + 'res_train_aokvqa.json') as f:
                vgg_feat = json.load(f)
                print(len(vgg_feat))

            with open('data/aokvqa/' + 'cnn_train_aokvqa.json') as f:
                q_feat = json.load(f)
                print(len(q_feat))

            if args.cap:
                with open('data/aokvqa/promptcap_train_mukea.json') as f:
                    cap = json.load(f)
                    print(len(cap))
        else:
            with open('tools/' + 'resnet_featrain_krvqa.json') as f:
                vgg_feat = json.load(f)
                print(len(vgg_feat))

            with open('data/kr-vqa/cnn_featrain_krvqa.json') as f:
                q_feat = json.load(f)
                print(len(q_feat))

            with open('data/kr-vqa/' + 'dic_img.json') as f:
                img_ids = json.load(f)
                print(len(img_ids))

            with open('data/kr-vqa/' + 'VG_100Kvg100_caption.json') as f:
                cap = json.load(f)
                print(len(cap))

            with open('data/kr-vqa/' + 'kb_id.json') as f:
                kbid = json.load(f)
                print(len(kbid))

class KgDataset(Dataset):
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
            most_id.append(id_map[qid])
        else:
            try:
                most_id.append(int(qid))
                qid=qid
            except:
                most_id.append(int(qid[0:len(qid) - 1]))
                qid=qid[0:len(qid) - 1]


        if args.glo_fea is True:
            # Get vgg_feat info
            if  args.dataset == 'okvqa':
                img_id = 1000000000000 + int(self.image_ids[index])
                image = "COCO_train2014_" + str(img_id)[1:13]
                image0 = "COCO_val2014_" + str(img_id)[1:13]
                try:
                    # vgg=glo_fea()
                    vgg_feats = np.array(vgg_feat[image][0])
                except:
                    vgg_feats = np.array(vgg_feat[image0][0])
                if args.pretrain:
                    # q_feats=glo_embedding(question)
                    q_feats = np.array(q_feat[str(qid)])
                else:
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
                img_id = 1000000000000 + int(self.image_ids[index])
                image = str(img_id)[1:13]
                vgg_feats = np.array(vgg_feat[image][0])
                if args.pretrain:
                    q_feats = np.array(q_feat[str(qid)])
                else:
                    q_feats = np.array(q_feat[str(qid)])

                if args.cap:
                    caption = cap[qid]
                    if args.pretrain is False:
                        if args.kg is True:
                            im_kg = kg[str(qid)]
                            caption = im_kg + caption
                        else:
                            caption = caption

                    return int(id_map[qid]), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats,caption
                else:
                    return int(id_map[qid]), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats
            else:
                img_id0 = str(img_ids[qid])
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

def my_collate(batch):
    batch = list(zip(*batch))
    res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
            'img': batch[3], 'spatial': batch[4],'mostid':batch[5]}
    del batch
    return res


class PretrainDataset(Dataset):
    def __init__(self, val=False, val_test=False):
        self.image_ids = image_ids
        self.qids = qids
        self.questions = questions
        self.length = question_lengths
        self.answers = answers
        self.most_answer_ids = most_answer_ids
        if val:
            self.qids = qids[30000:30500]
            self.questions = questions[30000:30500]
            self.answers = answers[30000:30500]
            self.most_answer_ids = most_answer_ids[30000:30500]
            self.image_ids = image_ids[30000:30500]
            self.length = question_lengths[30000:30500]


    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid = self.qids[index]
        question = self.questions[index]
        answer = self.answers[index]

        iid=self.image_ids[index]
        image_feature = pretrain_feature[self.image_ids[index]]['feats']
        spatial_feature = pretrain_feature[self.image_ids[index]]['sp_feats']

        most_id = self.most_answer_ids[index]
        if args.pretrain:
            most_id.append(int(qid[0:len(qid)-1]))
            qid=qid[0:len(qid)-1]
        else:
            most_id.append(int(qid))
            qid=qid

        if args.glo_fea is True:
            # Get vgg_feat info
            img_id = 1000000000000 + int(self.image_ids[index])
            image = "COCO_train2014_" + str(img_id)[1:13]
            image0 = "COCO_val2014_" + str(img_id)[1:13]
            try:
                vgg_feats = np.array(vgg_feat[image][0])
            except:
                vgg_feats = np.array(vgg_feat[image0][0])
            if args.pretrain:
                q_feats = np.array(q_feat[str(qid)])
            else:
                q_feats = np.array(q_feat[str(qid)])

            # most_id = self.most_answer_ids[index]

            if args.cap:
                caption=cap[str(self.image_ids[index])]

                if args.pretrain is False:
                    if args.kg is True:
                        im_kg = kg[str(qid)]
                        caption = im_kg + caption
                    else:
                        caption = caption

                return int(qid), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats,caption
            else:
                return int(qid), question, answer, image_feature, spatial_feature, most_id, vgg_feats, q_feats
        else:
            return int(qid), question, answer, image_feature, spatial_feature, most_id

def my_collate_pretrain(batch):
    batch = list(zip(*batch))
    if args.glo_fea is True:
        if args.cap is True:
            res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                    'img': batch[3], 'spatial':  batch[4],'mostid': batch[5], 'vgg_f': batch[6], 'q_f': batch[7], 'cap':batch[8]}
        else:
            res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                   'img': batch[3], 'spatial': batch[4], 'mostid': batch[5], 'vgg_f': batch[6], 'q_f': batch[7]}
    else:
        res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
               'img': batch[3], 'spatial': batch[4], 'mostid': batch[5]}
    del batch
    return res


