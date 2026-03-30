import os
import pathlib
import argparse
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from net.moce_ir import MoCEIR
from options import train_options
from utils.test_utils import save_img
from data.dataset_utils import IRBenchmarks, CDD11

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Test')


# args = parser.parse_args()



####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.net = MoCEIR(
            dim=opt.dim, 
            num_blocks=opt.num_blocks, 
            num_dec_blocks=opt.num_dec_blocks, 
            levels=len(opt.num_blocks),
            heads=opt.heads, 
            num_refinement_blocks=opt.num_refinement_blocks, 
            topk=opt.topk, 
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity, 
            depth_type=opt.depth_type, 
            stage_depth=opt.stage_depth, 
            rank_type=opt.rank_type, 
            complexity_scale=opt.complexity_scale,)
    
    def forward(self,x):
        return self.net(x)


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    lr_img = crop_img(np.array(Image.open(path).convert('RGB')), base=16)
    img_lq = ToTensor()(lr_img).unsqueeze(0).to(device)
    img_gt = None

    posofrMoCEIR = args.model_path.rfind('MoCE_IR')
    posofr = args.model_path.rfind('/')
    if posofrMoCEIR != -1:
        imgname = f"output_{args.model_path[posofrMoCEIR:posofr]}"
    else:
        raise ValueError("Model path does not contain 'MoCE_IR' substring.")

    

    return imgname, img_lq, img_gt
from PIL import Image
from utils.image_utils import random_augmentation, crop_img

from torchvision.transforms import ToTensor

def test_img(net, lr, save_path):
    
    with torch.no_grad():
        # Forward pass
        restored = net(lr)
        if isinstance(restored, List) and len(restored) == 2:
            restored , _ = restored
         # save output images
        restored = torch.clamp(restored,0,1)
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        return restored
import requests
def test_img_api(path,args):

    data = {
        "lq_path": path,
        "output_path": args.save_dir
    }

    url = f"http://127.0.0.1:{args.server_port}/predict"
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("API Response:", result)
    else:
        print("API Request Failed with status code:", response.status_code)

####################################################################################################
## main
import glob
def main(opt):
    if opt.server_port is not None:
        folder = opt.folder_lq
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            test_img_api(path,opt)
    else:

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        
        model_path = opt.model_path
        # Load model
        net = PLTestModel.load_from_checkpoint(
            model_path, opt=opt).cuda()
        net.eval()

        folder = opt.folder_lq

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            imgname, img_lq, img_gt = get_image_pair(opt, path)  # image to HWC-BGR, float32

            restored = test_img(net, img_lq, imgname+".png")

            output_path = opt.save_dir
            save_path = os.path.join(output_path, imgname+".png")
            save_img(save_path, img_as_ubyte(restored))

        

    

def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    # train_opt = train_options()
from options import base_parser, moce_ir, moce_ir_s
base_args = base_parser().parse_known_args()[0]
    
if base_args.model == "MoCE_IR_S":
    parser = moce_ir_s(base_parser())
elif base_args.model == "MoCE_IR":
    parser = moce_ir(base_parser())
else:
    raise NotImplementedError(f"Model '{base_args.model}' not found.")


parser.add_argument('--model_path', type=str, default=
    './',
    help='training loss')

parser.add_argument('--task', type=str, required=True, help='restoration type')
parser.add_argument('--folder_lq', type=str, required=True, help='input test dataset path')

parser.add_argument('--save_dir', type=str,  required=True, help='output save path')

parser.add_argument('--server_port', type=int, default=None, help='server port')
options = parser.parse_args()


    # Adjust batch size if gradient accumulation is used
if options.accum_grad > 1:
    options.batch_size = options.batch_size // options.accum_grad

main(options)  
# if __name__ == '__main__':
#     # train_opt = train_options()
#     from options import base_parser, moce_ir, moce_ir_s
#     base_args = base_parser().parse_known_args()[0]
    
#     if base_args.model == "MoCE_IR_S":
#         parser = moce_ir_s(base_parser())
#     elif base_args.model == "MoCE_IR":
#         parser = moce_ir(base_parser())
#     else:
#         raise NotImplementedError(f"Model '{base_args.model}' not found.")


#     parser.add_argument('--model_path', type=str, default=
#         './',
#         help='training loss')

#     parser.add_argument('--task', type=str, required=True, help='restoration type')
#     parser.add_argument('--folder_lq', type=str, required=True, help='input test dataset path')

#     parser.add_argument('--save_dir', type=str,  required=True, help='output save path')
#     options = parser.parse_args()


#     # Adjust batch size if gradient accumulation is used
#     if options.accum_grad > 1:
#         options.batch_size = options.batch_size // options.accum_grad

#     main(options)