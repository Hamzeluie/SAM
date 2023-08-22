import os
import argparse
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# Add the SAM directory to the system path
sys.path.append("./segment-anything")
from model.SAM import SAMFinetuner
from model.utils import Coco2MaskDataset, get_model_type_version, get_model_version_type, get_exam_id, Yaml_writer


NUM_WORKERS = 0  # https://github.com/pytorch/pytorch/issues/42518
NUM_GPUS = torch.cuda.device_count()


def main(params):
    # READ YAML
    # assert os.path.isfile(args.yaml_path), "check YAML PATH"
    
    # with open(args.yaml_path, "r") as stream:
    #     try:
    #         params = {}
    #         for i, j in yaml.safe_load(stream).items():
    #             params[i] = j
    #         print(params)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # ---------------------------------------------------------
    # check checkpoint path
    assert os.path.isfile(params["checkpoint_path"]), "please check --checkpoint_path"
    # set output dir

    if params["resume"]:
        assert os.path.isdir(params["output_dir"]), "output dir is not valid please check it."
        output_dir = os.path.join(params["output_dir"], "finetune")
        assert os.path.isfile(os.path.join(output_dir, "last.ckpt")), "last.ckpt is not in this directory"
    else:
        exam_id = get_exam_id(params["output_dir"])
        assert exam_id != 0, "output dir is not valid please check it."
        # assert params["output_dir"].split("/")[-1] == "exps", "output_dir should be exps."
        if exam_id - 1 != 0:
            if os.listdir(os.path.join(params["output_dir"], f"exp{exam_id - 1}", "finetune")).__len__() == 0:
                output_dir = os.path.join(params["output_dir"], f"exp{exam_id - 1}", "finetune")
            else:
                output_dir = os.path.join(params["output_dir"], f"exp{exam_id}", "finetune")
        else:
            output_dir = os.path.join(params["output_dir"], f"exp{exam_id}", "finetune")
        if os.path.isdir(output_dir) == False:
            os.makedirs(output_dir)

    # set model version
    if params["resume"]:
        # assert os.path.isfile(output_dir), "output_dir should be path of an exp<n>"
        best_chckpoint = [f for f in os.listdir(output_dir) if f[-4:] == "ckpt" and f != "last.ckpt"][0]
        best_chckpoint = best_chckpoint[len("model_version="):]
        best_chckpoint = best_chckpoint[:best_chckpoint.index("-step")]
        model_version = int(best_chckpoint)
        assert get_model_version_type(model_version) == params["model_type"], f"model version id and model type not matched, checkpoint version is {get_model_version_type(model_version)} and model type is {args.model_type}"
    else:
        assert get_model_type_version(params["model_type"]) >= 0, "model type is wrong please check it."
        model_version = get_model_type_version(params["model_type"])

    # load the dataset
    train_dataset = Coco2MaskDataset(data_root=params["data_root_train"], split="train", image_size=params["image_size"])
    val_dataset = Coco2MaskDataset(data_root=params["data_root_val"], split="val", image_size=params["image_size"])
 
    callback_yaml = Yaml_writer(ds_len=len(train_dataset), batch_size=params["batch_size"], epochs=params["epochs"], yaml_path=params["yaml_path"])

    # create the model
    model = SAMFinetuner(
        params["model_type"],
        params["checkpoint_path"],
        yaml_writer = callback_yaml,
        freeze_image_encoder=params["freeze_image_encoder"],
        freeze_prompt_encoder=params["freeze_prompt_encoder"],
        freeze_mask_decoder=params["freeze_mask_decoder"],
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        metrics_interval=params["metrics_interval"],
    )
    
    model_name = params["model_name"]
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=output_dir,
            # filename='{step}-{loss:.5f}',
            # filename='{model_name}',
            filename='{model_version}-{step}-{loss:.5f}',
            # filename='{step}-{val_per_mask_iou:.5f}',
            save_last=True,
            save_top_k=1,
            monitor="loss",
            mode="min",
            # monitor="val_per_mask_iou",
            # mode="max",
            save_weights_only=False,
            save_on_train_epoch_end=True,
            every_n_epochs= 1,
            # every_n_train_steps=metrics_interval,
        ),
        ]


    trainer = pl.Trainer(
        strategy='ddp' if NUM_GPUS > 1 else None,
        accelerator=params["device"],
        devices=NUM_GPUS,
        precision=32,
        callbacks=callbacks,
        max_epochs=params["epochs"],
        max_steps=-1,
        val_check_interval=params["metrics_interval"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )


    if params["resume"]:
        last_point = os.path.join(output_dir, "last.ckpt")
        print(last_point)
        assert os.path.isfile(last_point), f"{output_dir} file should contain last.ckpt file."
        
        trainer.fit(model, ckpt_path=last_point)
        callback_yaml.write_yaml(params["epochs"] - 1, callback_yaml.max_steps, {**{"loss":0, "train_per_mask_iou":0}, **{"status": "done"}})
        # write_yaml(args, output_dir)
    else:
        
        trainer.fit(model)
        
        best_real_name = [i for i in os.listdir(output_dir) if i != "last.ckpt"][0]
        best_new_name = params["model_name"] + "-" + best_real_name
        os.rename(os.path.join(output_dir, best_real_name), os.path.join(output_dir, best_new_name))
        callback_yaml.write_yaml(params["epochs"] - 1, callback_yaml.max_steps, {**{"loss":0, "train_per_mask_iou":0}, **{"status": "done"}})
        # write_yaml(args, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_train", type=str, required=True, help="path to the dataset root train")
    parser.add_argument("--data_root_val", type=str, required=True, help="path to the dataset root val")
    parser.add_argument("--model_type", type=str, required=True, help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--output_dir", type=str, default="./exps", help="path to save the model")
    parser.add_argument("--yaml_path", type=str, help="path to save the log in yaml")


    parser.add_argument("--freeze_image_encoder", action="store_true", help="freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", action="store_true", help="freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", action="store_true", help="freeze mask decoder")
    parser.add_argument("--metrics_interval", type=int, default=100, help="interval for logging metrics")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--epochs", type=int, default=1, help="number epoch to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--resume", type=bool, default=False, help="resume checkpoint")
    parser.add_argument("--model_name", type=str, help="best checkpoint name")
    parser.add_argument("--device", type=str, default="cpu", help="device to proccess", choices=["cuda", "cpu"])
                        
    args = parser.parse_args()
    params = {
    "data_root_train": args.data_root_train,
    "data_root_val": args.data_root_val,
    "model_type": args.model_type,
    "checkpoint_path": args.checkpoint_path,
    "output_dir": args.output_dir,
    "yaml_path": args.yaml_path,
    "freeze_image_encoder": args.freeze_image_encoder,
    "freeze_prompt_encoder": args.freeze_prompt_encoder,
    "freeze_mask_decoder": args.freeze_mask_decoder, 
    "metrics_interval": args.metrics_interval,
    "batch_size": args.batch_size,
    "image_size": args.image_size,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "weight_decay": args.weight_decay,
    "resume": args.resume,
    "model_name": args.model_name,
    "device": args.device
    }
    
    main(params=params)
    
# params = {'data_root_train': '/home/ubuntu/hamze/ImagePro_SAM/dataset/train', 
#           'data_root_val': '/home/ubuntu/hamze/ImagePro_SAM/dataset/val', 
#           'model_type': 'vit_h', 
#           'checkpoint_path': '/home/ubuntu/hamze/ImagePro_SAM/exps/sam_vit_h_4b8939.pth', 
#           'output_dir': '/home/ubuntu/hamze/ImagePro_SAM/exps',
#           'yaml_path': '/home/ubuntu/hamze/fineturne.yaml',
#           'freeze_image_encoder': True, 
#           'freeze_prompt_encoder': True, 
#           'freeze_mask_decoder': False, 
#           'metrics_interval': 500, 
#           'batch_size': 2, 
#           'image_size': 1024, 
#           'epochs': 2, 
#           'learning_rate': 0.0001, 
#           'weight_decay': 0.01, 
#           'resume': False, 
#           'model_name': 'ali', 
#           'device': 'cuda'}
# main(params=params)
