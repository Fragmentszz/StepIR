from pathlib import Path
import cv2
from sklearn import base
from tqdm import tqdm

from .add_single_degradation import *

def degrade(img, degradation, idx):
    router = {
        "low resolution": lr,
        "dark": darken,
        "noise": add_noise,
        "jpeg compression artifact": add_jpeg_comp_artifacts,
        "haze": add_haze,
        "motion blur": add_motion_blur,
        "defocus blur": add_defocus_blur,
        "rain": add_rain,



    }
    # if degradation == "haze":
    #     return add_haze(img, idx=idx)


    return router[degradation](img)


base_dir = Path("/home/wanglixin/datasets/LSDIR/")
hq_dir = base_dir / "selected"
degras_path = '/home/wanglixin/StepIR/fintune_AiO/datasets/degradations.txt'

lq_dir = base_dir / "LQ"
lq_dir.mkdir(exist_ok=True)

with open(degras_path) as f:
    lines = f.readlines()
combs = []
for line in lines:
    items = [i.strip() for i in line.strip().split('+')]
    degras = [i for i in items if i]
    if degras:
        combs.append(degras)
print(combs)
hq_json_paths = base_dir / 'train_selected.json'

import json
with open(hq_json_paths) as f:
    hq_json = json.load(f)
# for comb in combs:
#     n_degra = len(comb)
#     comb_dir = lq_dir / f"d{n_degra}" / "+".join(comb)
#     comb_dir.mkdir(parents=True)
#     for hq_path in tqdm(hq_paths, desc=" + ".join(comb), unit='img'):
#         img = cv2.imread(str(hq_path))
#         for degra in comb:
#             img = degrade(img, degra, idx=hq_path.stem)
#         cv2.imwrite(str(comb_dir / hq_path.name), img)
import os
import numpy as np
haze_img_dir = base_dir / "haze_imgs" / "ITS" / "train" / "hazy"
haze_imgs = os.listdir(haze_img_dir)


def random_choose_from_haze_img():
    # 如果为空则返回None
    if not haze_imgs:
        return None
    choosed = np.random.choice(haze_imgs)
    # haze_imgs.remove(choosed)
    return choosed

for hq_item in tqdm(hq_json, desc="Degrading images", unit='img'):

    hq_rel_path = hq_item["path"]

    hq_path = os.path.join(base_dir, hq_rel_path)
    
    # random choose a degradation combination
    comb_idx = np.random.choice(len(combs))
    comb = combs[comb_idx]

    # print(f'Processing {hq_rel_path} with degradations: {comb}')

    # random shuffle the degradation order
    np.random.shuffle(comb)

    # if haze in comb:
    if "haze" in comb:
        # img = add_haze(img, idx=hq_name.split('.')[0])
        hq_name = random_choose_from_haze_img()
        if hq_name is None:
            continue
        hq_path = os.path.join(str(haze_img_dir).replace('hazy',"gt"), hq_name)
        hq_rel_path = os.path.relpath(hq_path, base_dir)
        # comb.remove("haze")
        comb = ["haze"] + [degra for degra in comb if degra != "haze"]
        img = cv2.imread(os.path.join(str(haze_img_dir), hq_name))
    else:
        hq_name = os.path.basename(hq_path)
        img = cv2.imread(hq_path)
    for degra in comb:
        if degra == "blur":
            degra = np.random.choice(["motion blur", "defocus blur"])
        if degra == "haze":
            continue
        img = degrade(img, degra, idx=hq_name.split('.')[0])
    
    item_dict = {
        "hq_path": hq_rel_path,
        "lq_path": 'LQ/' + f"d{len(comb)}/" + hq_name,
        "degradations": "+".join(comb)
    }
    cv2.imwrite(str(base_dir / item_dict["lq_path"]), img)
    with open(base_dir / "degradation_info.json", 'a') as f:
        json.dump(item_dict, f)
        f.write('\n')
    # if "haze" in comb:
    #     print(item_dict)
        # break




