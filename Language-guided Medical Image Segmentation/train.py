##################### Libraries #####################
import torch
from torch.utils.data import DataLoader
from utils.new_dataset import ImageTextMaskDataset
import config
from new_wrapper import LanGuideMedSegWrapper
import lightning as L
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import torch.multiprocessing
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning)
torch.multiprocessing.set_sharing_strategy('file_system') # used when num_workers>0
torch.cuda.empty_cache()

########### Model Training Configuration ############

dataset = config.dataset_name
model_module = config.model_module
model_name = config.model_name
saving_dir = config.saving_dir
train_batch_size = config.train_batch_size
val_batch_size = config.valid_batch_size
lr = config.lr
max_epochs = config.max_epochs


if __name__ == '__main__':

    if dataset == 'kvasir':
        print("-"*50)
        print('using dataset: kvasir')
        print('using dataset: kvasir')
        print("-"*50)

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='C:/Datasets/Kvasir_prepared/train/image',
            masks_dir='C:/Datasets/Kvasir_prepared/train/mask',
            caps_file='F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/kvasir_polyp/anns/train.json'

        )

        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='C:/Datasets/Kvasir_prepared/val/image',
            masks_dir='C:/Datasets/Kvasir_prepared/val/mask',
            caps_file='F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/kvasir_polyp/anns/val.json'
        )
        
    elif dataset == 'isic':
        print("-"*50)
        print('using dataset: ClinicDB')
        print('using dataset: ClinicDB')
        print("-"*50)

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/clinicdb_polyp/anns/train.json'
        )

        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ClinicDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/clinicdb_polyp/anns/val.json'
        )
        
    elif dataset == 'covid':
        print("-"*50)
        print('using dataset: colondb_polyp')
        print('using dataset: colondb_polyp')
        print("-"*50)

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p6',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/colondb_polyp/anns/train.json'
        )

        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p6',
            images_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/images',
            masks_dir='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/Medvlsm/CVC-ColonDB/masks',
            caps_file='/media/iipl/35f051be-def5-48dd-b3a9-6db9e762c2d6/early_fusion/code/text_data/colondb_polyp/anns/val.json'
        )

    else:
        ValueError('No dataset')

    dl_train = DataLoader(ds_train, batch_size=train_batch_size, shuffle=True, num_workers=3, persistent_workers=True)
    dl_valid = DataLoader(ds_valid, batch_size=val_batch_size, shuffle=False, num_workers=3, persistent_workers=True)
    
    for i in range(config.num_exps):  # Repeat the experiment x times
        print("-"*50)
        print(f"Experiment {i+1} started !!!")
        print("-"*50)
        
        start_time = datetime.now()

        # Reinitialize model
        model = LanGuideMedSegWrapper()

        # Setting checkpoint and early stopping callbacks
        model_ckpt = ModelCheckpoint(
            dirpath=os.path.join(saving_dir + dataset + "/", model_name + "/" + model_module.split(".")[-1] + f"_EXP{i+1}"),
            filename=f"{model_module.split('.')[-1]}_{{val_loss:.4f}}",
            monitor='val_dice',
            save_top_k=3,
            mode='max',
            verbose=True,
        )

        # Initialize trainer
        trainer = L.Trainer(
            logger=True,
            max_epochs=max_epochs,
            accelerator='gpu',
            devices=1,
            callbacks = [model_ckpt],
            enable_progress_bar=True,
        )

        # Set random seed for reproducibility
        L.seed_everything(42 + i)

        # Start training
        print('Start training')
        print("-"*50)
        trainer.fit(model, dl_train, dl_valid)
        
        end_time = datetime.now()
        full_end_time = start_time.strftime("%d/%m/%Y, %H:%M:%S")
        print(f"Training Experiment {i+1} Finished!!!")
        print(f"Experiment {i+1} Duration: {full_end_time}")
        print("-"*50)

# CUDA_VISIBLE_DEVICES=1  python train.py