#!user/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import pickle
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from bisect import bisect
from math import fabs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LxmertTokenizer

from config import args
from contrastive_loss import ContrastiveLoss, l2_sim
from dataset import KgDataset, my_collate_pretrain, PretrainDataset, my_collate
from dataset import vocab_num
from dataset_val import KgDatasetVal
from model import KgPreModel, tokenizer
from transformers import get_linear_schedule_with_warmup


# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)

# torch.manual_seed(10)
# torch.cuda.manual_seed(10)
# cudnn.benchmark = False
# cudnn.deterministic = True

torch.multiprocessing.set_sharing_strategy('file_system')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate_tripleid(batch_anchor, candidate):
    # cos distance
    similarity = batch_anchor.mm(candidate.t())   # b * v

    # l2 distance
    # similarity = l2_sim(batch_anchor, candidate)   #b * v

    # cos largest:True  l2 largest:False
    prob, idx_1 = torch.topk(similarity, k=1, dim=1, largest=True)
    prob3, idx_3 = torch.topk(similarity, k=3, dim=1, largest=True)
    return idx_1.squeeze(), idx_3.squeeze()


def cal_batch_loss(target, target_true, criterion):
    target = target.view(-1, 2)
    target_true = target_true.view(-1, 1).squeeze()
    batch_loss = criterion(target, target_true)
    return batch_loss


def cal_acc_multi(ground_truth, preds, p,pd,simi,return_id = False):
    all_num = len(((np.array(ground_truth))[:,0:-1]).tolist())
    acc_num = 0
    ids = []
    dic_pre_true1 = {}
    dic_pre_true2 = {}
    dic_pre_true3 = {}
    dic_pre_false = {}
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        # ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id[0:-1]:
            if pred == aid:
                cnt += 1
        if cnt ==1:
            acc_num += 0.3
            # s=str(answer_id[-1])
            if args.logf:
                dic_pre_true1[str(answer_id[-1])] = [pred.tolist(),p[i].cpu().detach().numpy().tolist(),simi[i].cpu().detach().numpy().tolist()]
            # ids.append([int(pred), 1])
        elif cnt == 2:
            acc_num += 0.6
            if args.logf:
                dic_pre_true2[str(answer_id[-1])] = [pred.tolist(),p[i].cpu().detach().numpy().tolist(),simi[i].cpu().detach().numpy().tolist()]
            # ids.append([int(pred), 1])
        elif cnt > 2:
            acc_num += 1
            if args.logf:
                dic_pre_true3[str(answer_id[-1])] = [pred.tolist(),p[i].cpu().detach().numpy().tolist(),simi[i].cpu().detach().numpy().tolist()]
        else:
            if args.logf:
                dic_pre_false[str(answer_id[-1])] = [pred.tolist(),p[i].cpu().detach().numpy().tolist(),simi[i].cpu().detach().numpy().tolist()]
            # ids.append([int(pred), 1])
        # else:
        #     ids.append([int(pred), 0])

    if return_id:
        return acc_num / all_num, ids
    else:
        if args.logf:
            return acc_num / all_num,dic_pre_true1,dic_pre_true2,dic_pre_true3,dic_pre_false
        else:
            return acc_num / all_num

   
def cal_acc(ground_truth, preds, p,pd,simi,return_id = False):
    all_num = len(((np.array(ground_truth))[:,0:-1]).tolist())
    acc_num = 0
    ids = []
    dic_pre_true={}
    dic_pre_false={}
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        ids.append([i, int(pred)])
        for aid in answer_id[0:-1]:
            if pred == aid:
                acc_num += 1
                if args.logf:
                    dic_pre_true[str(answer_id[-1])]=[pred.tolist(),p[i].cpu().detach().numpy().tolist(),simi[i].cpu().detach().numpy().tolist()]
            else:
                if args.logf:
                    dic_pre_false[str(answer_id[-1])] = [pred.tolist(),p[i].cpu().detach().numpy().tolist(),simi[i].cpu().detach().numpy().tolist()]

    if return_id:
        return acc_num / all_num, ids
    else:
        if args.logf:
            return acc_num / all_num,dic_pre_true,dic_pre_false
        else:
            return acc_num / all_num


def train():
    if not args.pretrain:
        train_dataset = KgDataset(val=False)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, collate_fn=my_collate_pretrain)
        if args.validate:
            test_dataset = KgDatasetVal(val=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, collate_fn=my_collate_pretrain)
    else:
        train_dataset = PretrainDataset(val=False)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      num_workers=4, collate_fn=my_collate_pretrain, shuffle=True)#sampler=train_sampler)
        if args.validate:
            test_dataset = KgDatasetVal(val=False)
            # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            # test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
            #                              num_workers=8, collate_fn=my_collate, shuffle=False)#sampler=test_sampler)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                         num_workers=4, collate_fn=my_collate_pretrain, shuffle=False)#sampler=test_sampler)
    model= KgPreModel(vocab_num)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)


    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    
    # warm up 
    total_steps = (len(train_dataset) // (args.batch_size / torch.cuda.device_count())) * args.num_epochs \
        if len(train_dataset) % args.batch_size == 0 \
        else (len(train_dataset) // (args.batch_size / torch.cuda.device_count()) + 1) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.01 * total_steps,
                                                num_training_steps=total_steps)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_graph = ContrastiveLoss(measure='dot', margin=1.0, max_violation=False)

    print(args.load_pthpath)
    if args.load_pthpath == "":
        start_epoch = 0
    else:
        print('load model')
        # "path/to/checkpoint_xx.pth" -> xx
        if args.checkpoint:
            start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1
        else:
            start_epoch=0
        # model.module.load_state_dict(torch.load(args.load_pthpath))
        model.load_state_dict(torch.load(args.load_pthpath))


    best_acc = 0
    best_epoch = 0
    best_acc_t = 0
    best_epoch_t = 0
    best_acc_t3 = 0
    ok_a1=[]
    ok_a2=[]
    ok_a3=[]
    ok_f=[]
    kr_a=[]
    kr_f=[]
    if args.embedding:
        answer_candidate_tensor = torch.arange(0, vocab_num).view(-1, 1).long().cuda()
    for epoch in range(start_epoch, args.num_epochs):
        train_answers = []
        train_preds = []
        train_preds_trip = []
        train_preds_trip_3 = []
        train_answers_trip = []
        train_kq=[]
        train_iid=[]
        train_simi=[]
        for batch_data in train_dataloader:
            visual_faetures = torch.from_numpy(np.array(batch_data['img'], dtype=float)).float().to(device)
            source_seq = tokenizer(batch_data['ques'], padding=True, return_tensors="pt",
                                   add_special_tokens=True)
            input_id = source_seq['input_ids'].to(device)
            attention_mask = source_seq['attention_mask'].to(device)
            token_type_ids = source_seq['token_type_ids'].to(device)
            #------------cap-----------------------------------
            if args.cap is True:
                source_seq_cap = tokenizer(batch_data['cap'], padding=True, return_tensors="pt",
                                       add_special_tokens=True)
                input_id_cap = source_seq_cap['input_ids'].to(device)
                attention_mask_cap = source_seq_cap['attention_mask'].to(device)
                token_type_ids_cap = source_seq_cap['token_type_ids'].to(device)

            spatial_feature = torch.tensor(batch_data['spatial']).float().to(device)
            most_id = batch_data['mostid']
            most_id_tensor = torch.tensor(most_id).long()
            qid = batch_data['id']


            if args.glo_fea is True:
                vgg_fea=torch.tensor(batch_data['vgg_f']).float().to(device)
                q_fea=torch.tensor(batch_data['q_f']).float().to(device)

            model.zero_grad()
            if args.glo_fea is True:
                if args.cap:
                    anchor,p,qd,sim_matrix_v2l = model((input_id, attention_mask, token_type_ids,qid,input_id_cap, attention_mask_cap,token_type_ids_cap), (visual_faetures, spatial_feature,vgg_fea,q_fea))
                else:
                    anchor, p, qd,sim_matrix_v2l = model((input_id, attention_mask, token_type_ids, qid), (visual_faetures, spatial_feature, vgg_fea, q_fea))
            else:
                anchor,p,qd,sim_matrix_v2l  = model((input_id, attention_mask, token_type_ids,qid),(visual_faetures, spatial_feature))


            if args.embedding:
                most_id_tensor = torch.tensor(tuple((np.array(most_id)[:,0:-1]).tolist())).view(anchor.shape[0], -1).long().cuda()
                if torch.cuda.device_count() > 1:
                    most = model.module.decode_tail(most_id_tensor)
                else:
                    most = model.decode_tail(most_id_tensor)
            else:
                most = torch.tensor(batch_data['most']).float().to(device)
            most = F.normalize(most, dim=-1, p=2)

            if args.embedding:
                if torch.cuda.device_count() > 1:
                    answer_candidate_tensor_train = model.module.decode_tail(answer_candidate_tensor)
                    cls = model.module.cal_sim(anchor, answer_candidate_tensor_train)
                else:
                    answer_candidate_tensor_train = model.decode_tail(answer_candidate_tensor)
                    cls = model.cal_sim(anchor, answer_candidate_tensor_train)
            anchor = F.normalize(anchor, dim=1, p=2)
            optimizer.zero_grad()

            most_id_tensor = most_id_tensor[:,0].squeeze()
            loss_cl = criterion_cls(cls, most_id_tensor)
            if args.dataset == 'okvqa' or args.dataset == 'aokvqa':
                loss = 0
                for i in range(10):
                    most_i = most[:,i,:]
                    loss_mse = criterion_mse(anchor, most_i)
                    loss_graph = criterion_graph(anchor, most_i)
                    loss = loss + loss_mse + loss_graph+loss_cl
            else:
                loss_mse = criterion_mse(anchor, most)
                loss_graph = criterion_graph(anchor, most)
                loss = loss_mse + loss_graph+loss_cl

            loss_stat = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if args.embedding:
                    if torch.cuda.device_count() > 1:
                        answer_candidate_tensor_train = model.module.decode_tail(answer_candidate_tensor)
                    else:
                        answer_candidate_tensor_train = model.decode_tail(answer_candidate_tensor)
                    answer_candidate_tensor_train = F.normalize(answer_candidate_tensor_train, dim=1, p=2)
                    trip_predict, trip_predict_3 = generate_tripleid(anchor.float(), answer_candidate_tensor_train)
                else:
                    trip_predict, trip_predict_3 = generate_tripleid(anchor.float(), answer_candidate_tensor)

                for i, pre in enumerate(most_id):
                    train_answers.append(most_id[i])
                    train_preds_trip.append(trip_predict[i])
                    train_preds_trip_3.append(trip_predict_3[i])
                    train_answers_trip.append(most_id[i])

                    train_kq.append(p[i])
                    train_iid.append(qd[i])
                    train_simi.append(sim_matrix_v2l[i])

        if args.dataset == 'krvqa':
            train_acc_1_trip = cal_acc(train_answers_trip, train_preds_trip,train_kq,train_iid,train_simi)
            print('epoch %d train_loss = %.1f, acc_trip = %.4f' % (epoch, loss_stat,
                                                                          train_acc_1_trip))
        else:
            train_acc_1_trip = cal_acc_multi(train_answers_trip, train_preds_trip,train_kq,train_iid,train_simi)
            print('epoch %d train_loss = %.1f, acc_trip = %.4f' % (epoch, loss_stat,
                                                                          train_acc_1_trip))
        if args.validate:
            args.logs = True
            model.eval()
            answers = []  # [batch_answers,...]
            preds = []  # [batch_preds,...]
            preds_trip = []
            preds_trip_3 = []
            answers_trip = []
            k_q=[]
            iid=[]
            simi=[]
            print(f"\nValidation after epoch {epoch}:")
            for i, batch_data in enumerate(test_dataloader):
                with torch.no_grad():
                    visual_faetures = torch.tensor(batch_data['img']).float().to(device)
                    source_seq = tokenizer(batch_data['ques'], padding=True, return_tensors="pt",
                                           add_special_tokens=True).to(device)
                    input_id = source_seq['input_ids'].to(device)
                    attention_mask = source_seq['attention_mask'].to(device)
                    token_type_ids = source_seq['token_type_ids'].to(device)

                    # ------------cap-----------------------------------
                    if args.cap is True:
                        source_seq_cap = tokenizer(batch_data['cap'], padding=True, return_tensors="pt",
                                                   add_special_tokens=True)
                        input_id_cap = source_seq_cap['input_ids'].to(device)
                        attention_mask_cap = source_seq_cap['attention_mask'].to(device)
                        token_type_ids_cap = source_seq_cap['token_type_ids'].to(device)

                    spatial_feature = torch.tensor(batch_data['spatial']).float().to(device)
                    most_id = batch_data['mostid']
                    qid = batch_data['id']

                    if args.glo_fea is True:
                        vgg_fea = torch.tensor(batch_data['vgg_f']).float().to(device)
                        q_fea = torch.tensor(batch_data['q_f']).float().to(device)

                    if args.glo_fea is True:
                        if args.cap:
                            anchor, p, qd,sim_matrix_v2l = model((input_id, attention_mask, token_type_ids, qid, input_id_cap,attention_mask_cap, token_type_ids_cap),(visual_faetures, spatial_feature, vgg_fea, q_fea))
                        else:
                            anchor, p, qd,sim_matrix_v2l = model((input_id, attention_mask, token_type_ids, qid),(visual_faetures, spatial_feature, vgg_fea, q_fea))
                    else:
                        anchor,p,qd,sim_matrix_v2l = model((input_id, attention_mask, token_type_ids,qid), (visual_faetures, spatial_feature))

                    anchor = F.normalize(anchor, dim=1, p=2)
                    if args.embedding:
                        if torch.cuda.device_count() > 1:
                            answer_candidate_tensor_test = model.module.decode_tail(answer_candidate_tensor)
                        else:
                            answer_candidate_tensor_test = model.decode_tail(answer_candidate_tensor)
                        answer_candidate_tensor_test = F.normalize(answer_candidate_tensor_test, dim=1, p=2)
                        trip_predict, trip_predict_3 = generate_tripleid(anchor, answer_candidate_tensor_test)
                    else:
                        trip_predict, trip_predict_3 = generate_tripleid(anchor, answer_candidate_tensor)

                    for i, pre in enumerate(most_id):
                        answers.append(most_id[i])
                        try:
                            preds_trip.append(trip_predict[i])
                        except Exception as e:
                            preds_trip.append(trip_predict[i])
                            print(e)

                        preds_trip_3.append(trip_predict_3[i])
                        answers_trip.append(most_id[i])
                        k_q.append(p[i])
                        iid.append(qd[i])
                        simi.append(sim_matrix_v2l[i])

            if args.dataset == 'krvqa':
                args.logf=True
                acc_1_trip,dic_pre_true,dic_pre_false = cal_acc(answers_trip, preds_trip,k_q,iid,simi )
                print('epoch %d ,  acc_trip = %.4f' % (
                    epoch, acc_1_trip))

            else:
                args.logf = True
                acc_1_trip,dic_pre_true1,dic_pre_true2,dic_pre_true3,dic_pre_false = cal_acc_multi(answers_trip, preds_trip,k_q,iid,simi )
                print('epoch %d ,  acc_trip = %.4f' % (
                    epoch, acc_1_trip))


            if acc_1_trip > best_acc_t:
                best_acc_t = acc_1_trip
                best_epoch_t = epoch
            print("best_acc@1t={:.2%}, epoch{}".format(best_acc_t, best_epoch_t))
            if torch.cuda.device_count() > 1:
                if epoch == 0:
                    torch.save(model.module.state_dict(), args.model_dir + 'model_for_epoch_%d.pth' % epoch)
                else:
                    files_list = os.listdir(args.model_dir)
                    for fn in files_list:
                        if int(fn.split("_")[-1][:-4]) != best_epoch_t:
                            os.remove(args.model_dir + fn)
                    torch.save(model.module.state_dict(), args.model_dir + 'model_for_epoch_%d.pth' % epoch)
            else:
                if epoch == 0:
                    torch.save(model.state_dict(), args.model_dir + 'model_for_epoch_%d.pth' % epoch)
                else:
                    files_list = os.listdir(args.model_dir)
                    for fn in files_list:
                        if int(fn.split("_")[-1][:-4]) != best_epoch_t:
                            os.remove(args.model_dir + fn)
                    torch.save(model.state_dict(), args.model_dir + 'model_for_epoch_%d.pth' % epoch)
                    if epoch - best_epoch_t == 15:
                        break
            if args.dataset == 'fvqa':
                print("best_acc@3t={:.2%}".format(best_acc_t3))

            if args.pretrain is False and args.dataset == 'krvqa' and args.logf:
                kr_a.append(dic_pre_true)
                kr_f.append(dic_pre_false)

            if args.pretrain is False:
                if epoch-best_epoch_t==10 and args.dataset == 'krvqa':
                    if not os.path.exists(args.model_dir):
                        os.makedirs(args.model_dir)
                    torch.save(model.state_dict(),
                           os.path.join(args.model_dir, 'model_for_best5_epoch_%d.pth' % best_epoch_t))
                    return kr_a, kr_f
                    break
                else:
                    model.train()
            else:
                    model.train()

        if args.pretrain is False and (args.dataset == 'okvqa' or args.dataset == 'aokvqa') and args.logf:
            ok_a1.append(dic_pre_true1)
            ok_a2.append(dic_pre_true2)
            ok_a3.append(dic_pre_true3)
            ok_f.append(dic_pre_false)

        args.logf = False
        args.logs = False

    if args.pretrain is False and (args.dataset == 'okvqa' or args.dataset == 'aokvqa'):
        return ok_a1,ok_a2,ok_a3,ok_f
    elif args.pretrain is False and args.dataset == 'krvqa':
        return kr_a, kr_f


if __name__ == "__main__":
    if args.pretrain:
        train()
    else:
        if args.dataset == 'aokvqa':
            o1,o2,o3,o4=train()
            with open(args.model_dir_save+"/pre_result_true1.json", "a") as json_file:
                json.dump(o1, json_file)
            with open(args.model_dir_save+"/pre_result_true2.json", "a") as json_file2:
                json.dump(o2, json_file2)
            with open(args.model_dir_save+"/pre_result_true3.json", "a") as json_file3:
                json.dump(o3, json_file3)
            with open(args.model_dir_save+"/pre_result_false.json", "a" ) as json_file4:
                json.dump(o4, json_file4)
        else:
            k1,k2 = train()
            with open(
                     args.model_dir_save+"/pre_result_true.json", "a"
            ) as json_file5:
                json.dump(k1, json_file5)
            with open(
                     args.model_dir_save+"/pre_result_false.json", "a"
            ) as json_file6:
                json.dump(k2, json_file6)