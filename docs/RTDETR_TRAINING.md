# RT-DETR Cash Training - Usage Instructions

Since `roboflow` package installation has issues, use these alternative approaches:

## Option 1: Manual Roboflow Download (Recommended)

1. Go to your Roboflow project: https://app.roboflow.com/cashcountingx
2. Click "Download Dataset"
3. Select format: **YOLOv8**
4. Click "Show download code" and copy the snippet
5. Run the snippet to download to `datasets/cash/`

Then train with:
```bash
python3 scripts/train_rtdetr_cash.py \
    --api-key YOUR_KEY \
    --workspace cashcountingx \
    --project YOUR_PROJECT \
    --version 1 \
    --skip-download \
    --epochs 100 \
    --batch 8 \
    --device mps
```

## Option 2: Direct Training (if dataset already downloaded)

If you already have the dataset with `data.yaml`:
```python
from ultralytics import RTDETR

# Load pre-trained RT-DETR
model = RTDETR('rtdetr-l.pt')

# Train on your cash dataset
results = model.train(
    data='datasets/cash/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device='mps',  # or 'cuda' for NVIDIA GPU
    project='runs/rtdetr',
    name='cash_detector',
    patience=20,
)

# Use the trained model
best_model = RTDETR('runs/rtdetr/cash_detector/weights/best.pt')
results = best_model('resources/video_WLpx1XJLSKY.webm')
```

## Option 3: Quick Test Training
```bash
# If dataset is at datasets/cash with data.yaml
cd /Users/ebowwa/apps/caringmind-project/edge-training

python3 -c "
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
model.train(
    data='datasets/cash/data.yaml',
    epochs=5,  # Quick test
    imgsz=640,
    batch=4,
    device='mps',
)
"
```

**Expected training time:**
- 100 epochs on MPS (Apple Silicon): ~2-4 hours
- With CUDA GPU: ~30-60 minutes
