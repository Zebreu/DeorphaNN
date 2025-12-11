import os
import glob
import torch
import requests

from .model import DeorphaNN

def load_models(model_dir="pretrained_models", device="cpu"):
    os.makedirs(model_dir, exist_ok=True)

    base = ("https://huggingface.co/datasets/lariferg/DeorphaNN/"
            "resolve/main/pretrained_models/")
    files = ["pretrained_0.pth", "pretrained_1.pth", "pretrained_2.pth", "pretrained_3.pth", "pretrained_4.pth", "pretrained_5.pth", "pretrained_6.pth", "pretrained_7.pth", "pretrained_8.pth", "pretrained_9.pth"]

    for f in files:
        path = os.path.join(model_dir, f)
        if not os.path.exists(path):
            r = requests.get(base + f)
            with open(path, "wb") as out:
                out.write(r.content)

    models = []
    weight_paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    for path in weight_paths:
        w = torch.load(path, map_location="cpu")
        dim = w["lin.weight"].shape[1]
        model = DeorphaNN(dim)
        model.load_state_dict(w)
        model.to(device)
        model.eval()
        models.append(model)

    return models
