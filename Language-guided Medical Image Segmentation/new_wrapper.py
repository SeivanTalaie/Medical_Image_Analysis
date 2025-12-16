##################### Libraries #####################
from monai.losses import DiceCELoss
from torchmetrics import Accuracy
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
import config
import torch
import torch.nn as nn
import lightning as L
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
import importlib
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning
)
torch.cuda.empty_cache()

##################### Main Class ####################

class LanGuideMedSegWrapper(L.LightningModule):
    def __init__(self):
        super(LanGuideMedSegWrapper, self).__init__()
        
        model_module = importlib.import_module(config.model_module)  # model_module='utils.githubUNET.github_UNET'
        model_class = getattr(model_module, config.model_name) 
        self.model = model_class()  

        self.lr = config.lr
        self.epoch = config.max_epochs
        
        self.loss_fn = DiceCELoss()
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        self.train_dice = DiceScore(num_classes=2, average="macro")
        self.val_dice = DiceScore(num_classes=2, average="macro")
        self.test_dice = DiceScore(num_classes=2, average="macro")

        self.train_miou = BinaryJaccardIndex()
        self.val_miou = BinaryJaccardIndex()
        self.test_miou = BinaryJaccardIndex()
       
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    

    def forward(self, x):
        return self.model.forward(x)
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.epoch, T_mult=1)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    
    def training_step(self, batch, batch_idx):
        output = self(batch)
        if len(output) == 4:
            preds, gt, generated, generated_gt = output
            loss = self.loss_fn(preds,gt) + 0.1 * nn.L1Loss()(generated,generated_gt)
        elif len(output) == 2:
            preds, gt = output
            loss = self.loss_fn(preds,gt)
        else:
            ValueError('wrong output')
        
        self.train_acc.update(preds, gt)
        self.train_dice.update((preds > 0.5).long().squeeze(1), (gt > 0.5).long().squeeze(1))
        self.train_miou.update(preds, gt)

        self.log("acc", self.train_acc, prog_bar=True, on_step=True)
        self.log("dice", self.train_dice, prog_bar=True, on_step=True)
        self.log("MIoU", self.train_miou, prog_bar=True, on_step=True)
        self.training_step_outputs.append(loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self(batch)
        if len(output) == 4:
            preds, gt, generated, generated_gt = output
            loss = self.loss_fn(preds,gt) + 0.1 * nn.L1Loss()(generated,generated_gt)
        elif len(output) == 2:
            preds, gt = output
            loss = self.loss_fn(preds,gt)
        else:
            ValueError('wrong output')
        
        self.val_acc.update(preds, gt)
        self.val_dice.update((preds > 0.5).long().squeeze(1), (gt > 0.5).long().squeeze(1))
        self.val_miou.update(preds, gt)
        self.validation_step_outputs.append(loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        output = self(batch)
        if len(output) == 4:
            preds, gt, generated, generated_gt = output
            loss = self.loss_fn(preds,gt) + 0.1 * nn.L1Loss()(generated,generated_gt)
        elif len(output) == 2:
            preds, gt = output
            loss = self.loss_fn(preds,gt)
        else:
            ValueError('wrong output')
        
        self.test_acc.update(preds, gt)
        self.test_dice.update((preds > 0.5).long().squeeze(1), (gt > 0.5).long().squeeze(1))
        self.test_miou.update(preds, gt)
        self.validation_step_outputs.append(loss)

        return loss


    def on_train_epoch_end(self):
        mean_train_loss = torch.stack(self.training_step_outputs).mean().item()
        self.training_step_outputs.clear()
        current_epoch = self.trainer.current_epoch

        mean_train_acc = self.train_acc.compute().item()
        self.train_acc.reset()
        mean_train_dice = self.train_dice.compute().item()
        self.train_dice.reset()
        mean_train_miou = self.train_miou.compute().item()
        self.train_miou.reset()

        dic = {"Epoch": current_epoch,
               "train_loss": round(mean_train_loss, 4),
               "train_acc": round(mean_train_acc, 4),
               "train_dice": round(mean_train_dice, 4),
               "train_MIoU": round(mean_train_miou, 4)}
        
        self.log_dict(dic, logger=True, prog_bar=True)


    def on_validation_epoch_end(self):
        mean_val_loss = torch.stack(self.validation_step_outputs).mean().item()
        self.validation_step_outputs.clear()
        current_epoch = self.trainer.current_epoch

        mean_val_acc = self.val_acc.compute().item()
        self.val_acc.reset()
        mean_val_dice = self.val_dice.compute().item()
        self.val_dice.reset()
        mean_val_miou = self.val_miou.compute().item()
        self.val_miou.reset()

        dic = {"Epoch": current_epoch,
               "val_loss": round(mean_val_loss, 4),
               "val_acc": round(mean_val_acc, 4),
               "val_dice": round(mean_val_dice, 4),
               "val_MIoU": round(mean_val_miou, 4)}
        
        self.log_dict(dic, logger=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        mean_test_loss = torch.stack(self.test_step_outputs).mean().item()
        self.test_step_outputs.clear()
        current_epoch = self.trainer.current_epoch

        mean_test_acc = self.test_acc.compute().item()
        self.test_acc.reset()
        mean_test_dice = self.test_dice.compute().item()
        self.test_dice.reset()
        mean_test_miou = self.test_miou.compute().item()
        self.test_miou.reset()

        dic = {"Epoch": current_epoch,
               "test_loss": round(mean_test_loss, 4),
               "test_acc": round(mean_test_acc, 4),
               "test_dice": round(mean_test_dice, 4),
               "test_MIoU": round(mean_test_miou, 4)}
        
        self.log_dict(dic, logger=True, prog_bar=True)


    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0]), batch[1]
        else:
            return self(batch)
        




