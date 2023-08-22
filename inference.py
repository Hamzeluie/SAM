import os
import yaml
import argparse
from collections import OrderedDict
from pathlib import Path
import torch
from sahi.utils.file import save_json
from segment_anything import build_sam_vit_l, build_sam_vit_h, build_sam_vit_b
from model.utils import inference, get_model_type_version, get_model_version_type
from segment_anything import sam_model_registry

def main(params):
    # assert os.path.isfile(args.yaml_path), "check YAML PATH"
    # with open(args.yaml_path, "r") as stream:
    #     try:
    #         params = {}
    #         for i, j in yaml.safe_load(stream).items():
    #             params[i] = j
    #         print(params)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    
    # check image path
    assert os.path.isfile(params["image_path"]) or os.path.isdir(params["image_path"]), "image path is invalid."
    # checkpoint read_exp  structure
    assert int(params["read_exp"].split("/")[-1][3:]), "read path structure should be like this 'path/to/exps/exp<id>'"
    # assert params["read_exp"].split("/")[-2] == "exps" and int(params["read_exp"].split("/")[-1][3:]), "read path structure should be like this 'path/to/exps/exp<id>'"
    assert os.path.isdir(os.path.join(params["read_exp"], "finetune")), f"should have sub directory finetune"
    exp_finetune = os.path.join(params["read_exp"], "finetune")
    
    # check checkpoint_path
    assert os.path.isfile(params["checkpoint_path"]) , f"there is no any file in {params['checkpoint_path']}"
    checkpoint_path = params["checkpoint_path"]
    sam_model_orig = sam_model_registry["vit_h"](checkpoint="/home/ubuntu/hamze/ImagePro_SAM/exps/sam_vit_h_4b8939.pth")
    sam_model_orig.to("cuda");
    
    # if params["ckpt_last"]:
    #     checkpoint_path = os.path.join(exp_finetune, "last.ckpt")
    #     assert os.listdir(exp_finetune).__len__() > 0, "there is no any ckpt file in this exp"
    #     best_name = [f for f in os.listdir(exp_finetune) if f[-4:] == "ckpt" and f != "last.ckpt"][0]
    #     assert best_name.__len__() > 0, "both last and best checkpoint need."
    # else:    
    #     # read best
    #     assert os.listdir(exp_finetune).__len__() > 0, f"{exp_finetune} is empty."
    #     best_name = [f for f in os.listdir(exp_finetune) if f[-4:] == "ckpt" and f != "last.ckpt"][0]
    #     checkpoint_path = os.path.join(exp_finetune, best_name)
    # assert os.path.isfile(checkpoint_path), "there is no any ckpt file in this exp"
    
    # make dire inference
    output_dir = os.path.join(params["read_exp"], "inference")
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    # catch model type and version checkpoint
    best_name = [f for f in os.listdir(exp_finetune) if f[-4:] == "ckpt" and f != "last.ckpt"][0]
    assert best_name.__len__() > 0, "both last and best checkpoint need."
    best_name = best_name[best_name.index("model_version="):]
    best_name = best_name[:best_name.index("-step")]
    best_name = best_name[len("model_version="):]
    model_version = int(best_name)
    model_type = get_model_version_type(model_version)


    if model_type == "vit_h":
        model = build_sam_vit_h().to(params["device"])
        # model.eval()      
    elif model_type == "vit_b":
        model = build_sam_vit_b().to(params["device"])
        model.eval()      
    elif model_type == "vit_l":
        model = build_sam_vit_l().to(params["device"])
        model.eval()      
    else:
        assert False, "Model type is should be one of ['vit_h', 'vit_b', 'vit_l']"

    loaded_statedict = torch.load(checkpoint_path, map_location=torch.device(params["device"]))["state_dict"]

    state_dict = OrderedDict([(k.replace("model.", ""), v) for k, v in loaded_statedict.items()])

    model.load_state_dict(state_dict)
    
    
    
    
    # x = torch.randn(3, 224, 224, requires_grad=True)
    # torch_out = model(x, multimask_output=False)
    # torch.onnx.export(model,               # model being run
    #               x,                         # model input (or a tuple for multiple inputs)
    #               "/home/ubuntu/hamze/onnx/sam.onnx",   # where to save the model (can be a file or file-like object)
    #               export_params=True,        # store the trained parameter weights inside the model file
    #               opset_version=10,          # the ONNX version to export the model to
    #               do_constant_folding=True,  # whether to execute constant folding for optimization
    #               input_names = ['input'],   # the model's input names
    #               output_names = ['output'], # the model's output names
    #               dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                             'output' : {0 : 'batch_size'}})
    
    
    
    if params["xy_distance"].__len__() == 0:
        xy_distance = None
    else:
        xy_distance = list((int(xy_distance[0]), int(xy_distance[1])))  
    
    if os.path.isfile(params["image_path"]):
        inference(sam_model=model, image_path= params["image_path"], x_y_distance=xy_distance, write_base_dir= output_dir)
    elif os.path.isdir(params["image_path"]):
        path_list = Path(params["image_path"])
        jpg_files = list(path_list.glob("*.jpg"))
        png_files = list(path_list.glob("*.png"))
        bmp_files = list(path_list.glob("*.bmp"))
        for f in jpg_files + png_files + bmp_files:
            inference(sam_model=model, image_path= f.as_posix(), x_y_distance=xy_distance, write_base_dir= output_dir)
    else:
        print("Please check --image_path.it should be a file path or a directory contain '*.jpg' or '*.png'")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_path", type=str, required=True, help="path to the images")
#     parser.add_argument("--read_exp", type=str, required=True, help="path to the experiment 'exp<id>'")
#     parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
#     parser.add_argument("--xy_distance", default=[], action='append', help="set list of xy_distance to set custom point grid")
#     parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda'])
    
#     args = parser.parse_args()
#     params = {
#     "image_path": args.image_path,
#     "read_exp": args.read_exp,
#     "checkpoint_path": args.checkpoint_path,
#     "xy_distance": args.xy_distance,
#     "device": args.device,
#     }
#     main(params=params)

# exit()

params = {
"image_path": "/home/ubuntu/hamze/ImagePro_SAM/dataset/screw",
"read_exp": "/home/ubuntu/hamze/ImagePro_SAM/exps/exp5",
"checkpoint_path": "/home/ubuntu/hamze/ImagePro_SAM/exps/exp5/finetune/ali-model_version=0-step=188-loss=0.08243.ckpt",
"xy_distance":[],
"device": "cuda",
}

main(params=params)