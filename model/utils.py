import os
import sys
import pickle
import random
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import yaml
from sahi.utils.coco import Coco
import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv
# Add the SAM directory to the system path
sys.path.append("./ImagePro_SAM")
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam
import time


class Yaml_writer:
    def __init__(self, ds_len, batch_size, epochs: int, yaml_path:str) -> None:
        self.yaml_path = yaml_path
        assert os.path.isfile(self.yaml_path), f"there is no any yanl file in {self.yaml_path}"
        self.ds_steps = ds_len / batch_size
        self.epochs = epochs
        self.max_steps = int(self.ds_steps * self.epochs)
        self.one_percent_batch = self.max_steps / 100
        self.list_keys = ["progress_percent", "accuracy", "loss", "status"]
        
    def __write(self, data: str):
        with open(self.yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def write_yaml(self, epoch, current_step, model_output):
        progress_percent = int((current_step + (epoch * self.ds_steps)) / self.one_percent_batch)
        data = {}
        if model_output['status'] != "done":
            data["progress_percent"] = progress_percent
            data["accuracy"] = float(model_output["train_per_mask_iou"])
            data["loss"] = float(model_output['loss'])
            data["status"] = model_output['status']
        else:
            data["status"] = model_output['status']
            self.list_keys = ["status"]
        yaml_text = {}
        with open(self.yaml_path, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
                for k, v in parsed_yaml.items():
                    if k not in self.list_keys:
                        yaml_text[f"{k}"] = v
            except yaml.YAMLError as exc:
                print(exc)
        self.__write({**yaml_text, **data})


def write_yaml(arguments:argparse.ArgumentParser, base_path:str):
    """ A function to write YAML file"""
    assert os.path.isdir(base_path), "CHECK PATH TO WRITE YAML."
    path = os.join(base_path, 'detailes.yml')
    
    for k, v in arguments.items():
        data = f"{k} : {v} \n"
    
    with open(path, 'w') as f:
        yaml.dump(data, f)
    

def get_model_type_version(model_type: str) -> int:
    """
    model_type: should be one of ['vit_h', 'vit_l', 'vit_b']
    """
    if model_type == "vit_h":
        model_version = 0
    elif model_type == "vit_l":
        model_version = 1
    elif model_type == "vit_b":
        model_version = 2
    else:
        model_version = -1
    return model_version


def get_model_version_type(model_version: int) -> str:
    """
    model_type: should be one of [0, 1, 2]
    """
    if model_version == 0:
        model_type = "vit_h"
    elif model_version == 1:
        model_type = "vit_l"
    elif model_version == 2:
        model_type = "vit_b"
    else:
        model_type = ""
    return model_type


def get_exam_id(path: str)-> int:
    """
    path: str path 'exam<id>'
    retutn id + 1
    
    exam 1:
    |examples
    --exp1
    --exp2
    get_exam_id("./examples") -> 3
    
    exam 2:
    
    |examples
    --test.py
    get_exam_id("./examples") -> 1
    exam 3:
    
    |examples
    --test.py
    get_exam_id("./exm") -> 0 
    0 means error .your path is not valid
    """
    if os.path.isdir(path):
        max_exp_id = 0
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                try:
                    last_file_id = int(d.replace("exp", ""))
                    if max_exp_id < last_file_id:
                        max_exp_id = last_file_id
                except ValueError:
                    continue
        return max_exp_id + 1   
    else:
        return 0
       
       
def get_points(bbox:torch.tensor, classes:torch.tensor, DEVICE:str)-> torch.tensor:
    label = []
    center_points = []
    classes = classes.cpu().detach().numpy()
    for b, c in zip(bbox, classes):
        xmin, ymin, xmax, ymax = list(b.cpu().detach().numpy())
        w, h = (xmax - xmin), (ymax - ymin)
        cente_x, center_y = int((w // 2) + xmin), int((h // 2) + ymin)
        label.append(c)
        center_points.append([cente_x, center_y])

    center_points = np.array(center_points)
    center_points = torch.Tensor(center_points).to(DEVICE).reshape([center_points.shape[0], 1,center_points.shape[1]])
    label = torch.Tensor(label).to(DEVICE)
    return (center_points, label)


def get_masks(masks:torch.tensor, DEVICE:str)-> torch.tensor:
    res = []
    masks = masks.cpu().detach().numpy()
    for mask in masks:
        label_msk = cv2.resize(mask.astype("float32"), (256,256), interpolation=cv2.INTER_LINEAR)
        res.append(label_msk)
    res = np.stack(res, axis=0)
    res = torch.Tensor(res)[None]
    label_msk = res.permute([1,0,2,3]).to(device=DEVICE, dtype=torch.float32)
    return label_msk


def inference_preview(sam_model, images_base_dir:str, write_base_dir:str, checkpoint:str, model_type:str, epoch:int, DEVICE:str):
    # SELECT RANDOM
    paths = Path(images_base_dir)
    paths = list(paths.glob("**/mask_*.png"))
    paths = paths[random.randint(0, len(paths) - 1)]

    IMAGE_PATH = paths.as_posix().replace("mask_", "")
    
    # IMAGE_PATH = "/home/ubuntu/hamze/sam_github/segment-anything-finetuner-main/magnet_taile/val/mask_000158.png"


    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_annotator = sv.MaskAnnotator()
    
    write_dir_tuned = os.path.join(write_base_dir, paths.stem.replace("mask_", "") + f"_e_{epoch}_tuned.jpg")
    write_dir_orig = os.path.join(write_base_dir,  paths.stem.replace("mask_", "") + f"_org.jpg")
    
    list_imgs = [i for i in os.listdir(write_base_dir) if i[-3:] == "jpg"]
    es = [int(j.split("/")[-1][9:][:-10]) for j in list_imgs if j.find("_e_") != -1]
    if epoch not in es:
        sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
        sam_model_orig.to(DEVICE);
        sam_model_orig = SamAutomaticMaskGenerator(sam_model_orig)
        orig_result = sam_model_orig.generate(image_rgb)
        if orig_result.__len__() > 0:
            detections = sv.Detections.from_sam(orig_result)
            annotated_image = mask_annotator.annotate(image_bgr, detections)
            cv2.imwrite(write_dir_orig, annotated_image)
            print(f"file wrote {write_dir_orig}")
        else:
            print("has not reasult ORIG")

    if epoch not in es:
        sam_model_tuned = SamAutomaticMaskGenerator(sam_model)
        tuned_result = sam_model_tuned.generate(image_rgb)
        
        if tuned_result.__len__() > 0:
            detections = sv.Detections.from_sam(tuned_result)
            annotated_image = mask_annotator.annotate(image_bgr, detections)
            
            cv2.imwrite(write_dir_tuned, annotated_image)
            print(f"file wrote {write_dir_tuned}")
        else:
            print("has not reasult TUNED")


def build_points_grid(x_pixel_distance, y_pixel_distance, image_shape):
    w, h = image_shape
    coordinates = []
    for i in range(0, w, x_pixel_distance):
        for j in range(0, h, y_pixel_distance):
            coordinates.append(np.array([i / w, j / h]))
    return [coordinates]


def inference(sam_model, image_path:str, write_base_dir:str, x_y_distance: list=None)-> dict:
    
    assert os.path.isfile(image_path), f"image '{image_path}' not found"
    assert os.path.isdir(write_base_dir), f"there is no any  directory '{write_base_dir}' to write"
    img_name = image_path.split("/")[-1]
    write_dir = os.path.join(write_base_dir, img_name[:-3] + "jpg")

    image_bgr = cv2.imread(image_path)
    image_bgr = cv2.resize(image_bgr, (1024,1024))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    assert (x_y_distance == None) or (type(x_y_distance) == list) , "x_y_distance should be 'None' or list"
        
    if x_y_distance is not None:
        assert (type(x_y_distance[0]) == int) and (type(x_y_distance[1]) == int), "x_y_distance sould be int"
        assert (x_y_distance[0] <= image_rgb.shape[0] // 2) and (x_y_distance[1] <= image_rgb.shape[1] // 2), "x_y_distance sould be half of image shape"
        gride = build_points_grid(x_y_distance[0], x_y_distance[1], list(image_bgr.shape[:2]))  
        sam_model = SamAutomaticMaskGenerator(sam_model, points_per_side=None, point_grids=gride)
    else:
        sam_model = SamAutomaticMaskGenerator(sam_model)
    
    mask_annotator = sv.MaskAnnotator() 
    start_time = time.time()   
    reasult = sam_model.generate(image_rgb)
    if reasult.__len__() > 0:
        detections = sv.Detections.from_sam(reasult)
        annotated_image = mask_annotator.annotate(image_bgr, detections)
        print("time spend", time.time() - start_time)
        cv2.imwrite(write_dir, annotated_image)
        print(f"file wrote {write_dir}")
        # write out put json file
        if reasult is not None:
            selected_keys = ['bbox', 'area', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']
            res = {}
            bbox = ""
            for idx, d in enumerate(reasult):
                tmp_dict = {}
                for k, v in d.items():
                    if k in selected_keys:
                        if k == "bbox":
                            for n in v:
                                bbox += str(n) + ","
                            bbox = bbox[:-1]
                            bbox += ";"
                            
                        tmp_dict[k] = v
                res[idx] = tmp_dict
            
            image_name = image_path.split("/")[-1]
            image_name = image_name.split(".")[0]
            js_write_path = os.path.join(write_base_dir, image_name + ".json")
            with open(js_write_path, 'w') as f:
                json.dump(res, f, indent=1)
            print(f"json resaults wrote in '{js_write_path}'")
        return reasult
    else:
        print("has not reasult TUNED")
        return None


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


# coco mask style dataloader
class Coco2MaskDataset(Dataset):
    def __init__(self, data_root, split, image_size):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        # annotation = os.path.join(data_root, split, "_annotations.coco.json")
        annotation = os.path.join(data_root, "_annotations.coco.json")
        self.coco = Coco.from_coco_dict_or_path(annotation)

        # TODO: use ResizeLongestSide and pad to square
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_resize = transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR)

    def __len__(self):
        return len(self.coco.images)

    def __get_binary_mask(self, path, category):
        # file_path = path.as_posix()[:-3] + "png"
        gt_grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        gt_grayscale = cv2.resize(gt_grayscale, (self.image_size , self.image_size), interpolation=cv2.INTER_LINEAR)
        gt_grayscale[gt_grayscale > 0] = 255
        if category == "backgroung":
            gt_grayscale = (~(gt_grayscale == 255)) * 1
            return gt_grayscale
        else:
            gt_grayscale = (gt_grayscale == 255) * 1
            return gt_grayscale
        

    def __getitem__(self, index):
        coco_image = self.coco.images[index]
        image_path = os.path.join(self.data_root, coco_image.file_name)
        # image_path = os.path.join(self.data_root, self.split, coco_image.file_name)
        assert os.path.isfile(image_path), "check root path!"
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.width, image.height
        ratio_h = self.image_size / image.height
        ratio_w = self.image_size / image.width
        image = self.image_resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        bboxes = []
        masks = []
        labels = []
        for annotation in coco_image.annotations:
            x, y, w, h = annotation.bbox
            # get scaled bbox in xyxy format
            bbox = [x * ratio_w, y * ratio_h, (x + w) * ratio_w, (y + h) * ratio_h]

            # mask_file_path = os.path.join(self.data_root, self.split, "mask_" + coco_image.file_name)
            mask_file_path = os.path.join(self.data_root, "mask_" + coco_image.file_name)
            mask = self.__get_binary_mask(mask_file_path, annotation.category_name)
            # mask = get_bool_mask_from_coco_segmentation(annotation.segmentation, original_width, original_height)
            # mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
            label = annotation.category_id
            bboxes.append(bbox)
            masks.append(mask)
            labels.append(label)
        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        labels = np.stack(labels, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).long(), torch.tensor(labels)
    
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, labels

