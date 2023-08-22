import sys
from collections import defaultdict, deque
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
import torch.nn.functional as F
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry
from model.utils import get_masks, get_points, inference_preview, all_gather


class SAMFinetuner(pl.LightningModule):

    def __init__(
            self,
            model_type,
            checkpoint_path,
            yaml_writer,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            train_dataset=None,
            val_dataset=None,
            metrics_interval=10
        ):
        super(SAMFinetuner, self).__init__()
        self.num_worker = 0
        self.num_gpus = torch.cuda.device_count()
        self.status = "waiting"
        self.yaml_writer = yaml_writer
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.freeze_image_encoder = freeze_image_encoder
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))

        self.metrics_interval = metrics_interval


    def forward(self, imgs, bboxes, msks, labels):

        # inference_preview(sam_model=self.model, 
        #           images_base_dir= "./hamze/sam_github/segment-anything-finetuner-main/magnet_taile/val", 
        #           write_base_dir= "/home/ubuntu/hamze/sam_git_tuned_output_report", 
        #           checkpoint=self.checkpoint_path, 
        #           model_type=self.model_type, 
        #           epoch=2)

        _, _, H, W = imgs.shape
        try:
            self.status = "training"
            features = self.model.image_encoder(imgs)
            num_masks = sum([len(b) for b in bboxes])
            loss_focal = loss_dice = loss_iou = 0.
            predictions = []
            tp, fp, fn, tn = [], [], [], []
            for feature, bbox, b_mask, clss in zip(features, bboxes, msks, labels):
                # Embed prompts
                # *********BY MEHDI
                # points = get_points(bbox)
                # *****************
                points = get_points(bbox, clss, DEVICE=self.device)
                label_msk = get_masks(b_mask, DEVICE=self.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points,boxes=bbox,masks=label_msk)
                # sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None,boxes=bbox,masks=None,)
                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=feature.unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # Upscale the masks to the original image resolution
                masks = F.interpolate(
                    low_res_masks,
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                predictions.append(masks)
                # Compute the iou between the predicted masks and the ground truth masks
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    masks,
                    b_mask.unsqueeze(1),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
                # Compute the loss
                masks = masks.squeeze(1).flatten(1)
                b_mask = b_mask.flatten(1)
                loss_focal += sigmoid_focal_loss(masks, b_mask.float(), num_masks)
                loss_dice += dice_loss(masks, b_mask.float(), num_masks)
                loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
                tp.append(batch_tp)
                fp.append(batch_fp)
                fn.append(batch_fn)
                tn.append(batch_tn)
            return {
                'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
                'loss_focal': loss_focal,
                'loss_dice': loss_dice,
                'loss_iou': loss_iou,
                'predictions': predictions,
                'tp': torch.cat(tp),
                'fp': torch.cat(fp),
                'fn': torch.cat(fn),
                'tn': torch.cat(tn),
            }
        except Exception as e:
            self.status = f"Error: {e}"
    
    def training_step(self, batch, batch_nb):
        try:
            imgs, bboxes, masks, labels = batch
            outputs = self(imgs, bboxes, masks, labels)

            for metric in ['tp', 'fp', 'fn', 'tn']:
                self.train_metric[metric].append(outputs[metric])

            # aggregate step metics
            step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
            per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
            metrics = {
                "loss": outputs["loss"],
                "loss_focal": outputs["loss_focal"],
                "loss_dice": outputs["loss_dice"],
                "loss_iou": outputs["loss_iou"],
                "train_per_mask_iou": per_mask_iou,
            }
            self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
            if batch_nb % 50 == 0 :
                self.yaml_writer.write_yaml(self.current_epoch, batch_nb, {**metrics, **{"status": self.status}})
            return metrics
        except Exception as e:
            self.status = f"Error: {e}"
            self.yaml_writer.write_yaml(self.current_epoch, batch_nb, {**{"loss":0, "train_per_mask_iou":0}, **{"status": self.status}})
            exit()
    
    def validation_step(self, batch, batch_nb):
        imgs, bboxes, masks, labels = batch
        outputs = self(imgs, bboxes, masks, labels)
        outputs.pop("predictions")
        return outputs
    
    def validation_epoch_end(self, outputs):
    # def on_validation_epoch_end(self):
        # *******BY MEHDI
        # inference_preview(sam_model=self.model, 
        #           images_base_dir= "./dataset/val", 
        #           write_base_dir= f"./exp/exp_{}", 
        #           checkpoint=self.checkpoint_path, 
        #           model_type=self.model_type, 
        #           epoch=self.current_epoch,
        #           DEVICE=self.device)
        # *********************************
        if self.num_gpus > 1:
            outputs = all_gather(outputs)
            # the outputs are a list of lists, so flatten it
            outputs = [item for sublist in outputs for item in sublist]
        # aggregate step metics
        step_metrics = [
            torch.cat(list([x[metric].to(self.device) for x in outputs]))
            for metric in ['tp', 'fp', 'fn', 'tn']]
        # per mask IoU means that we first calculate IoU score for each mask
        # and then compute mean over these scores
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")

        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log_dict(metrics)
        return metrics
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale
            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=False)
        return val_loader

