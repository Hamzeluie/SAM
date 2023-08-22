# Segment Anything Model(SAM)
SAM segmentation is used for object detection. For more details about the theory of the model read [Facebook](https://github.com/facebookresearch/segment-anything) and for details about codes read [bhpfelix](https://github.com/bhpfelix/segment-anything-finetuner).


# Installing
install this repo with git 

    git clone git@github.com:NaserFaryad/ImagePro_SAM.git
    cd ImagePro_SAM; pip install -r requirements.txt
 torch-cudnn and tochvision requierment .
 
 **install necessary**
    
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# Project Structure
    ├── dataset/
    ├── exps/
    ├── model/
    ├── finetune.py
    ├── inference.py
    
# Dataset
dataset structure.Coco format

    ├── dataset/
    │   ├── train/
    │   │   ├── _annotations.coco.json # COCO format    annotation
    │   │   ├── 000001.png             # Images
    │   │   ├── 000002.png
    │   │   ├── ...
    │   ├── val/
    │   │   ├── _annotations.coco.json # COCO format annotation
    │   │   ├── xxxxxx.png             # Images
    │   │   ├── ...
# exps
in each running finetune.py it would build an experiment with the name of 'exp' + integer id

example: exp1, exp2, ...

every exp contains finetune details text file, last and best checkpoint.

structure name of every best checkpoints are "{model_version}-{step}-{loss:.5f}"

- model_version: base on {"hit_h": 0, "vit_l": 1, "vit_b": 2}
- step: number of steps that the model trained
- loss: minimal loss which is best in all epochs

this directory also has necessary model checkpoints which you should download and put in this directory.

**Notice**: when you input arguments. argument model type should match to the version of the checkpoint which you downloaded.

Three model versions of the model are available with different backbone sizes. 
Click the links below to download the checkpoint for the corresponding model type.

- default or vit_h: [ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) SAM model.
- vit_l: [ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) SAM model
- vit_b: [ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) SAM model.

**directory structure**

    ├── exps/                       # the SAM pretrained checkpoints
    │   ├── exp1/
    │   │   ├── finetune/
    │   │   │   ├── details.txt
    │   │   │   ├── last.ckpt
    │   │   │   ├── model_version=0-step=188-loss=0.33437.ckpt
    │   │   ├── inference/
    |   │   │   │   ├── <name>.jpg
    |   │   │   │   ├── ...
    │   ├── ...
    │   ├── sam_vit_h_4b8939.pth
    │   ├── ...

# model
this directory contains two files:
- SAM.py: SAMFinetuner class which is our model is in this file.
- utils.py: there is some functions and dataset class in this file.


        ├── model/.
        │   ├── SAM.py
        │   ├── utils.py

# FineTuning
you can train the model based on parameters that you input.
also, you can resume one trained model

**Notice:** if you want to resume, first check **epochs**.it would be more than previous **epochs** and set checkpoint path that you want to resume to **output_dir** .

parameters:
- data_root_train: root of train dataset
- data_root_val: root of val dataset
- model_type: should match with checkpoint ['hit_h', 'vit_l', 'vit_b']
- checkpoint_path:downloaded checkpoint path
- output_dir:output directory path to write checkpoints (./exps)
- yaml_path:path yaml file to write detail of training on yaml 
- freeze_image_encoder:boolean for traine or freeze layer
- freeze_prompt_encoder:boolean for traine or freeze layer
- freeze_mask_decoder:boolean for traine or freeze layer
- metrics_interval:int apply evaluation every N step
- batch_size:batch size to train
- image_size:image size to use for model type
- learning_rate:optimization LR
- weight_decay:optimization WD
- model_name: a name for trained best checkpoint
- device: 'cpu' or 'cuda'

for example:
~~~~
cd ImagePro_SAM
~~~~
~~~~
python finetune.py \
    --data_root_train ./dataset/train \
    --data_root_val ./dataset/val \
    --model_type vit_h \
    --checkpoint_path ./weights/sam_vit_h_4b8939.pth \
    --output_dir ./exps/exp<n>
    --yaml_path ./path/to/*.yaml
    --freeze_image_encoder \
    --freeze_prompt_encoder \
    --freeze_mask_decoder \
    --metrics_interval 100
    --batch_size 2 \
    --image_size 1024 \
    --epochs 10 \
    --learning_rate 1.e-5 \
    --weight_decay 0.01\
    --resume False
    --model_name str
    --device cuda
~~~~
# Inference
parameters:
- image_path: path of an image or directory of images ('.jpg' or '.png') to predict
- read_exp: path to exp< id > for using checkpoints
- checkpoint_path: path of trained checkpoint by finetune.py.
- device: 'cpu' or 'gpu'

for example:
~~~~
cd ImagePro_SAM
~~~~
~~~~
python inference.py \
    --image_path path/to/ *.jpg | *.png \
    --read_exp ./exps/exp1 \
    --checkpoint_path ./path/to/*.pth \
    --device cuda
~~~~

**Note** : if using boolean arguments like (ckpt, freeze_image_encoder, ...).only for setting them to 'True' define them in cli. otherwise do not use that. 