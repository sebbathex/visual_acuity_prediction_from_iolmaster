"""
Step 1 – Create image feature embeddings (In the manuscript: "Feature Extraction").

Input:  images_{task}.csv with one row per eye: patient_id, eye (OD/OS), path_FR, path_AS, path_MC.
        The paths point to the three image regions rendered from the IOLMaster 700 PDF report of the
        examination used for the task (same-day: day of the VA measurement; postop: preoperative biometry).
Output: embeddings_{task}_{embedder}.parquet with patient_id, eye and one column per feature
        (e.g. AS_000 ... AS_255 for YOLO11n-cls).

All three networks are used with their pretrained weights only (no fine-tuning).

Usage: python extract_embeddings.py --task sameday|postop --embedder yolo|clip|dinov2
"""


import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import config as C

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def crop(path: str, img: str) -> Image.Image:
    """Removes labels and borders around an image region (boxes in config.CROP_BOXES)."""
    return Image.open(path).crop(C.CROP_BOXES[img]).convert("RGB")


def yolo_embedder():
    """YOLO11n-cls via the built-in embed() function of Ultralytics.
    embed() returns the globally average-pooled output of the last backbone layer before the
    classification head (layer 9 of 11), i.e. a 256-dimensional vector per image.
    """
    from ultralytics import YOLO
    model = YOLO("yolo11n-cls.pt")
    return lambda im: model.embed(im, device=DEVICE, verbose=False)[0].cpu().numpy()


def clip_embedder():
    """CLIP ViT-B/32 image encoder; 512-dimensional, L2-normalized image features."""
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model, proc = CLIPModel.from_pretrained(name).to(DEVICE).eval(), CLIPProcessor.from_pretrained(name)

    @torch.inference_mode()
    def embed(im):
        e = model.get_image_features(**proc(images=im, return_tensors="pt").to(DEVICE)).float()
        return (e / e.norm(dim=-1, keepdim=True))[0].cpu().numpy()
    return embed


def dinov2_embedder():
    """DINOv2 ViT-S/14 (self-supervised); 384-dimensional class-token embedding."""
    from torchvision import transforms as T
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE).eval()
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   

    @torch.inference_mode()
    def embed(im):
        return model(tf(im).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
    return embed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sameday", "postop"])
    ap.add_argument("--embedder", required=True, choices=C.EMBEDDERS)
    a = ap.parse_args()
    task, emb_name = a.task, a.embedder
    embed = {"yolo": yolo_embedder, "clip": clip_embedder, "dinov2": dinov2_embedder}[emb_name]()

    eyes = pd.read_csv(C.INPUT_DIR / f"images_{task}.csv")
    rows = []
    for _, r in eyes.iterrows():
        vec = {"patient_id": r.patient_id, "eye": r.eye}
        for img in C.IMAGES[task]:
            e = np.asarray(embed(crop(r[f"path_{img}"], img)), dtype=np.float32)
            assert e.shape == (C.EMB_DIM[emb_name],), e.shape
            vec.update({f"{img}_{i:03d}": v for i, v in enumerate(e)})
        rows.append(vec)

    C.WORK_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(C.WORK_DIR / f"embeddings_{task}_{emb_name}.parquet", index=False)
    print(f"{task}/{emb_name}: {len(rows)} eyes embedded")
