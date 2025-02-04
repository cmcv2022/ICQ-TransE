#!user/bin/env python
# -*- coding:utf-8 -*-
import argparse
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument("--inference", action="store_true", help='complete dataset or not')
parser.add_argument("--pretrain", action="store_true", default=True, help='use vqa2.0 or not')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--val', type=float, default=0.75,
                    help='number of epochs')
parser.add_argument('--model_dir', type=str, default='model_save_dir/20250115_procap_75_llm_cap/train/',
                    help='model file path')
parser.add_argument('--model_dir_save', type=str, default='model_save_dir/20250115_procap_75_llm_cap/',
                    help='model file path')
parser.add_argument('--data_train', type=str, default='data/kr-vqa/krvqa_train.json',
                    help='model file path')
parser.add_argument('--data_val', type=str, default='data/kr-vqa/krvqa_test.json',
                    help='model file path')
parser.add_argument("--hard", dest='hard', action='store_const', default=False, const=True)
parser.add_argument("--logf", dest='logf', action='store_const', default=False, const=True)
parser.add_argument("--logs", dest='logs', action='store_const', default=False, const=True)
parser.add_argument("--load_pthpath",
                    default="",
                    help="the fine-tuned model.")
parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default='data/pre_model/model',
                    help='Load the pre-trained LXMERT model.')
parser.add_argument("--validate", action="store_true", default=True,
                    help="Whether to validate on val split after every epoch.")
parser.add_argument("--embedding", action="store_true", default=True, help="Whether to train tail embedding.")
parser.add_argument("--accumulate", action="store_true", default=True, help="Whether to fine-tune.")
parser.add_argument("--dataset", default="aokvqa", help="dataset that model training on")
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument("--checkpoint", dest='checkpoint', action='store_const', default=False, const=True,
                    help="To continue training, path to .pth file of saved checkpoint.")

# LXRT Model Config
# Note: LXRT = L, X, R (three encoders), Transformer
parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

parser.add_argument('--load', type=str, default=None,
                    help='Load the model (usually the fine-tuned model).')
parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default='',
                    help='Load the pre-trained LXMERT model with QA answer head.')
parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                    help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                         'the model would be trained from scratch. If --fromScratch is'
                         ' not specified, the model would load BERT-pre-trained weights by'
                         ' default. ')

# Use global fea
parser.add_argument("--globalfea", dest='glo_fea', action='store_const', default=True, const=True)
parser.add_argument("--cap", dest='cap', action='store_const', default=True, const=True)
parser.add_argument("--cat", dest='cat', action='store_const', default=False, const=True)
parser.add_argument("--kg", dest='kg', action='store_const', default=False, const=True)
# Use kg
# parser.add_argument("--kgfea", dest='kg_fea', action='store_const', default=False, const=True)

# Set seeds
torch.manual_seed(parser.parse_args().seed)
torch.cuda.manual_seed(parser.parse_args().seed)
random.seed(parser.parse_args().seed)
np.random.seed(parser.parse_args().seed)

args = parser.parse_args()

print("------arguments/parameters-------")
for k, v in vars(args).items():
    print(k + ': ' + str(v))
print("---------------------------------")