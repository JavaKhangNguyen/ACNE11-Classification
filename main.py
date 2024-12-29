import subprocess
import ultralytics

"""## Test installation"""
ultralytics.checks()

"""## Train model"""
subprocess.run(['yolo','task=classify', 'mode=train', 'model=weight/yolov8s-cls.pt','data=/home/datpham/datpham/ACNE11-Classification/datasets','epochs=200', 'batch=16', 'imgsz=800', 'save=True', 'save_period=10', 'patience=0'])

"""## Validating model"""
with open('valid.txt', 'w') as f:
    result = subprocess.run(['yolo', 'task=classify', 'mode=val', 'model=runs/classify/train/weights/best.pt', 'imgsz=800', 'data=/home/datpham/datpham/ACNE11-Classification/datasets'], capture_output=True, text=True)
    f.write(result.stdout)
    f.write(result.stderr)
