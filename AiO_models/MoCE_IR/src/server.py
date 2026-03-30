

from flask import Flask, request, jsonify
import torch
import os
import argparse
import subprocess
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

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

from utils.image_io import save_image_tensor

# 初始化 Flask 应用
app = Flask(__name__)

# 全局变量
model = None
Args = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_MOCEIR():

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # 1. 先在 CPU 上实例化
    print("=> loading model to CPU first...")
    # 2. 加载权重
    model = PLTestModel.load_from_checkpoint(
        Args.model_path, opt=Args)
    model.eval()
    

    
    # 4. 移动到 GPU
    model.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return model
from utils.image_utils import random_augmentation, crop_img
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

@app.route("/predict", methods=["POST"])
def predict():
    """推理接口"""
    # 获取 JSON 数据
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    output_path = data.get("output_path", None)
    lq_path = data.get("lq_path", "")

    if output_path is None:
        return jsonify({"error": "output_path must be provided"}), 400

    try:
        # 创建输出目录
        subprocess.check_output(['mkdir', '-p', output_path])
        # 获取图像对
        imgname, lr, _ = get_image_pair(Args, lq_path)
        save_path = os.path.join(output_path, imgname + '.png')
        # to do
        with torch.no_grad():
        # Forward pass
            restored = model(lr)
            if isinstance(restored, List) and len(restored) == 2:
                restored , _ = restored
            # save output images
            restored = torch.clamp(restored,0,1)
            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            
        print(f'Saving image to {save_path}')
        save_img(save_path, img_as_ubyte(restored))
        return jsonify({"result": f"Image saved to {save_path}"})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # 参数解析
    
    # train_opt = train_options()
    from options import base_parser, moce_ir, moce_ir_s
    base_args = base_parser().parse_known_args()[0]
        
    if base_args.model == "MoCE_IR_S":
        parser = moce_ir_s(base_parser())
    elif base_args.model == "MoCE_IR":
        parser = moce_ir(base_parser())
    else:
        raise NotImplementedError(f"Model '{base_args.model}' not found.")
    
    parser.add_argument('--model_path', type=str, default='./', help='path to model')
    parser.add_argument('--task', type=str, required=True, help='restoration type')
    parser.add_argument('--server_port', type=int, default=8000, help='server port')
    
    args = parser.parse_args()
    Args = args # 设置全局 Args 供 get_image_pair 使用
    print("Loading models...")
    
    # 1. 加载主模型
    model = load_MOCEIR()
        
    print(f"Starting Flask server on port {args.server_port}...")
    
    # 启动 Flask 服务
    # host='0.0.0.0' 允许外部访问
    app.run(host="0.0.0.0", port=args.server_port, debug=False)