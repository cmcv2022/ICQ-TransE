### LIBRARIES ###
# Global libraries
import os
import codecs
import ast

import _pickle
import json

from termcolor import colored
from tqdm import tqdm

import requests
import io
import zipfile
import gzip
import shutil
from config import args

import numpy as np

from transformers import LxmertConfig, LxmertTokenizer, LxmertModel
tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
import torch
# from pytorch_pretrained_bert.tokenization import BertTokenizer

### FUNCTION DEFINITIONS ###
def download_file(source_url, dest_path, source_path=""):
    """
        Downloads the given archive and extracts it
        Currently works for:
            - `zip` files
            - `tar.gz` files

        Inputs:
            - `source_url` (str): URL to download the ZIP file
            - `source_path` (str): path of the file in the ZIP file
            - `dest_path` (str): path of the extracted file
    """
    # Initiate the request
    r = requests.get(source_url, stream=True)

    # Measure the total size of the ZIP file
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True)

    file_extension = source_url.split(".")[-1]

    if file_extension == "zip":
        # Save the ZIP file in a temporary ZIP file
        with open(os.path.join("data", "raw", "temp.zip"), "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print(
                colored(
                    "ERROR: Something went wrong while downloading the ZIP file", "red"
                )
            )

        z = zipfile.ZipFile(os.path.join("data", "raw", "temp.zip"))
        # Extract the file from the temporary file
        if source_path != "":
            z.extract(source_path, os.path.dirname(dest_path))
            os.rename(os.path.join(os.path.dirname(dest_path), source_path), dest_path)
        else:
            z.extractall(os.path.dirname(dest_path))
            # z.extractall(dest_path.split(os.path.sep)[:-1])

        # Remove the temporary file
        os.remove(os.path.join("data", "raw", "temp.zip"))

    elif file_extension == "gz":
        # Save the GZ file in a temporary GZ file
        with open(os.path.join("data", "raw", "temp.gz"), "wb") as temp_file:
            for data in r.iter_content(block_size):
                t.update(len(data))
                temp_file.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print(
                colored(
                    "ERROR: Something went wrong while downloading the GZ file", "red"
                )
            )

        with gzip.open(os.path.join("data", "raw", "temp.gz"), "rb") as file_in:
            with open(dest_path, "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)

        # Remove the temporary file
        os.remove(os.path.join("data", "raw", "temp.gz"))


def download_and_save_file(downloaded_file, message_name, cfg, dataset):
    """
        Downloads the specified file and saves it to the given path.

        Inputs:
            - `downloaded_file` (str): name of the downloaded file (cf. `configs/main.yml`)
            - `message_name` (str): name of the downloaded file that will be displayed
    """
    if not os.path.exists(cfg["paths"][dataset][downloaded_file]):
        try:
            print(colored("Downloading {}...".format(message_name), "yellow"))
            download_file(
                cfg["download_links"][dataset][downloaded_file],
                cfg["paths"][dataset][downloaded_file],
                cfg["paths"][dataset][downloaded_file + "_raw"],
            )
            print(colored("{} was successfully saved.".format(message_name), "green"))

        except Exception as e:
            print(colored("ERROR: {}".format(e), "red"))



def initiate_embeddings():
    """
        Saves the embedding dictionary in a file
    """
    # Check if the raw file exists
    if not os.path.exists("data/conceptnet/raw/embeddings.txt"):
        # Download the embeddings file
        download_file(
            "https://ttic.uchicago.edu/~kgimpel/comsense_resources/embeddings.txt.gz",
            "data/conceptnet/raw/embeddings.txt",
        )

    dict_embedding = {}
    with open("data/conceptnet/raw/embeddings.txt", "r") as raw_file:
        for entry in tqdm(raw_file, desc="Saving the node embeddings"):
            entry.strip()
            if entry:
                # word = s.split(" ")[0]
                # embedding=s.split(" ")[1:]
                # embedding=embedding.replace(" ", ",")
                embedding_split = entry.replace(" \n", "").split(" ")
                word = embedding_split[0]
                if word=='douse':
                    dict_embedding['douse']=[-0.036697,-0.022913,0.116233,0.047123,0.309197,0.298409,0.013085,-0.438115,0.173056,0.177791,-0.256662,-0.206903,-0.070952,0.202392,-0.275014,-0.079623,0.085338,0.126511,-0.327852,0.132575,-0.151003,0.041332,0.098347,0.123907,0.146914,-0.242280,0.263716,0.422949,0.285988,0.400306,0.200617,0.024604,0.094795,-0.215312,-0.308863,0.079667,0.075841,-0.234782,0.309248,-0.111984,-0.073682,0.028721,0.205369,0.166565,0.030564,-0.092426,-0.198365,-0.150006,0.210103,-0.124619,-0.076932,0.109750,-0.005759,0.141053,-0.036725,-0.151439,0.065561,0.171034,-0.169113,-0.033487,-0.013065,-0.152867,0.005810,0.039536,-0.256188,0.208498,-0.006843,0.096500,-0.136612,0.069631,-0.058705,0.017920,0.207801,-0.036526,0.086026,0.153917,0.167286,-0.044397,0.177769,-0.292242,0.231229,0.157654,-0.002559,-0.081928,0.144668,0.028379,-0.130566,0.184153,-0.057037,-0.187306,-0.089486,0.019623,0.147888,-0.112225,0.050660,-0.083019,-0.086881,-0.063039,-0.122365,0.161222,-0.158677,0.265578,0.090776,-0.040379,-0.050569,-0.118057,0.168614,-0.123679,-0.098062,-0.074090,-0.230502,0.197713,-0.347361,0.099742,-0.224144,0.329711,0.107113,0.273713,0.030895,0.045708,-0.026106,0.036205,-0.067876,-0.222802,-0.073343,0.230056,-0.062293,-0.033074,-0.218427,0.276834,-0.023408,0.006742,0.265278,0.128735,-0.066952,-0.047572,-0.073425,0.273172,0.105305,-0.187800,0.133159,0.071221,0.089881,0.190069,0.035059,0.338253,0.185213,0.047214,-0.019977,-0.137299,0.248599,-0.041086,-0.039352,0.037320,-0.004359,-0.199487,-0.126885,-0.083292,-0.045520,-0.037537,0.262288,-0.036051,-0.010844,-0.230896,-0.017650,-0.135975,-0.129410,-0.115469,-0.218849,0.104275,-0.233273,-0.017917,-0.002001,0.178321,0.126867,-0.025928,-0.249387,-0.026973,0.044996 -0.250451,0.031431,0.253847,0.091361,0.060627,0.068059,0.100037,0.177616,0.105489,0.195213,0.303896,0.002710,-0.348195,0.330642,-0.240489,-0.210553,-0.176066,0.147762,-0.018938,0.060202,0.171078]
                    continue
                else:
                    embedding = list(map(float, embedding_split[1:]))
                    dict_embedding[word]=embedding
    print(len(dict_embedding))
    d_e=json.dumps(dict_embedding)
    # Save in JSON file
    with open(
        "data/conceptnet/processed/embeddings.json", "a"
    ) as pkl_file:
        # _pickle.dump(dict_embedding, pkl_file)
        pkl_file.write(d_e)

def load_embeddings():
    """
        Loads the embeddings from ConceptNet
    """
    if not os.path.exists(
        "data/conceptnet/processed/embeddings.json"
    ):
        initiate_embeddings()

    with open(
        "data/conceptnet/processed/embeddings.json", "r"
    ) as pkl_file:
        dict_embedding = json.load(pkl_file)
    print(dict_embedding['to'])
    return dict_embedding


def get_txt_questions(split):
    """
        Returns the text of the questions
    """

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    if args.pretrain:
        question_path ='data/vqa_train_filter.json'
    else:
        question_path ='data/okvqa_train.json'
    # question_path = (
    #     "data/ok-vqa/OpenEnded_mscoco_"
    #     + str(split)
    #     + "2014_questions.json"
    # )
    # questions = sorted(
    #     json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
    # )
    # questions_ordered = {}
    # for question in questions:
    #     questions_ordered[question["question_id"]] = question["question"]
    with open(question_path,'r') as f:
        qf=json.load(f)

    dict_tokens = {}
    for _, question in qf.items():
        tokens = tokenizer.tokenize(question["question"])
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        for token in tokens:
            token_emb = tokenizer.vocab.get(token, tokenizer.vocab["[UNK]"])

            if token_emb not in dict_tokens:
                dict_tokens[token_emb] = token

    return dict_tokens


# if __name__=='__main__':
#     initiate_embeddings()