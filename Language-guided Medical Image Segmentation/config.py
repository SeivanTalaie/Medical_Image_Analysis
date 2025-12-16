############# training configuration 

# Model_Modules = [
#     "utils.githubUNET.github_UNET",
#     "utils.githubUNET.github_UNET_aug",
#     "utils.githubUNET.github_UNET_text_gene",
#     "utils.githubUNET.github_UNET_aug_text_gene",
    
#     "utils.missformer.missformer",
#     "utils.missformer.missformer_aug",
#     "utils.missformer.missformer_text_gene",
#     "utils.missformer.missformer_aug_text_gene",
    
#     "utils.transunet.transunet",
#     "utils.transunet.transunet_aug",
#     "utils.transunet.transunet_text_gene",
#     "utils.transunet.transunet_aug_text_gene",
    
#     "utils.unet_plus.unet_plus",
#     "utils.unet_plus.unet_plus_aug",
#     "utils.unet_plus.unet_plus_text_gene",
#     "utils.unet_plus.unet_plus_aug_text_gene"]

model_module = "utils.githubUNET.github_UNET"

## Model_names = ["UNet", "MISSFormer", "VisionTransformer", "NestedUNet"]
model_name = "UNet"

## Dataset_names = ["kvasir", "isic", "covid"]
dataset_name = "kvasir"

num_exps = 5

train_batch_size = 8  #8
valid_batch_size = 8  #8

lr = 0.0001

max_epochs = 100

saving_dir = "F:/Visual Language Grounding (VLG)/Article Repo/Clean Code/savings/"

