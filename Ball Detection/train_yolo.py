"""
Fine-tune YOLO26n for squash player detection on GPU.

Setup (once):
    pip install -U ultralytics torch --break-system-packages
    # Make sure your torch build actually has CUDA:
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    # If that prints False, reinstall torch with the correct CUDA wheel from
    # https://pytorch.org/get-started/locally/ before doing anything else -
    # ultralytics will silently fall back to CPU otherwise and training will crawl.

Run:
    python train.py
"""

import torch
from ultralytics import YOLO

def main():
    assert torch.cuda.is_available(), "CUDA not available - fix your torch install first"
    device = 0  # first GPU. Use "0,1" for multi-GPU, "cpu" to force CPU (not recommended)
    print(f"Training on: {torch.cuda.get_device_name(device)}")

    # Load pretrained COCO weights, we fine-tune the head + backbone from here
    model = YOLO(r"C:\Users\light\Downloads\Ball Detection\runs\fine_tune\weights\last.pt")

    results = model.train(
        resume = True,
        data=r"C:\Users\light\Downloads\Squash Ball Detection.v4i.yolo26\data.yaml",
        epochs=100,
        patience=12,          # early stop if val metric doesn't improve for 20 epochs
        imgsz=640,             # bump to 960/1280 if players are small/far from camera
        batch=8,              # drop to 8 if you hit CUDA OOM on the nano model
        device=device,
        workers=8,
        amp=True,              # mixed precision - faster + less VRAM, default on GPU
        optimizer="auto",      # YOLO26 defaults to MuSGD, leave this alone
        lr0=0.01,
        cos_lr=True,
        mosaic=1.0,            # strong aug helps since your dataset is likely small
        mixup=0.1,
        degrees=5.0,           # squash court camera angle doesn't vary wildly, keep rotation mild
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        project="runs/",
        name="fine_tune",
        exist_ok=True,
        val=True,
        plots=True,
    )

    # Validate against your held-out validation split (from data.yaml's `val:`)
    metrics = model.val()
    print(metrics.box.map, metrics.box.map50)

    # Evaluate against your separate held-out TEST set explicitly
    test_metrics = model.val(data="data.yaml", split="test")
    print("Test mAP50-95:", test_metrics.box.map)
    print("Test mAP50:", test_metrics.box.map50)

if __name__ == "__main__":
    main()