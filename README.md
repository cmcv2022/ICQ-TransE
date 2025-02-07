## Requirement
torch==1.8.0

transformers == 4.18.0

## Training       
1. Create model_save_dir 
```                           
mkdir model_save_dir
```

2. Preprocessing   
```
$ mkdir data
$ cd data
```

We reorganized the storage structure of image features as:

```
vqa_img_feature_train.pickle{
"image_id":{'feats': features, 'sp_feats': spatial features}
}
```

The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to 1

The image features are provided by and downloaded from the original bottom-up attention' [repo](https://github.com/peteanderson80/bottom-up-attention#pretrained-features),  then follow the [script](https://github.com/AndersonStra/MuKEA/blob/main/vqa_v2_pretrain/tsv2feature.py) to process the feature.

```
python tsv2feature.py
```

### Optional download link
The image features with **objects' label** are provided by and downloaded from the origin LXMERT' [repo](https://github.com/airsplay/lxmert#google-drive), then follow the [script](https://github.com/AndersonStra/MuKEA/blob/main/vqa_v2_pretrain/tsv2feature_objects.py) to process the feature.

```
python tsv2feature_objects.py
```

### Image features for KRVQA
The image features for KRVQA are generated based on the code in this [repo](https://github.com/violetteshev/bottom-up-features), and can be downloaded form 

[google drive](https://drive.google.com/file/d/1YUhqLLXGouBsy6C-i8SIQ86VXIkclrm9/view?usp=sharing)

unzip the file and put it under `/data/kr-vqa`

### Pre-training on VQAv2
```
python train.py --embedding --model_dir model_save_dir --dataset finetune-dataset/okvqa/krvqa/vqav2 --pretrain --accumulate --validate
```       

note: `--dataset` parameter is to set the dataset for finetune


### Fine-tuning     
```
python train.py --embedding --model_dir model_save_dir --dataset okvqa/krvqa/vqav2 --load_pthpath model_save_dir/checkpoint --accumulate --validate
```

