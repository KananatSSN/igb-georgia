from ultralytics import YOLO

model = YOLO('yolo12m.pt')  # Load the classification model

# Start training
results = model.train(
    data=r"D:\Kananat_Arm\Data\data-delfi\2_verifiedDataset\Iteration2\data.yaml",
    epochs=500,
    imgsz=640,
    batch=32,
    device='0'  # Use '0' for first GPU, 'cpu' for CPU training
)