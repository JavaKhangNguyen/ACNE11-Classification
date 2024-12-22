import subprocess
import ultralytics

"""## Test installation"""
ultralytics.checks()

"""## Train model"""
subprocess.run(['yolo','task=classify', 'mode=train', 'model=weight/yolo11m-cls.pt','data=data.yaml','epochs=200', 'batch=16', 'imgsz=640', 'save=True', 'save_period=10', 'patience=0'])

"""## Validating model"""
with open('valid.txt', 'w') as f:
    result = subprocess.run(['yolo', 'task=classify', 'mode=val', 'model=runs/classify/train/weights/best.pt', 'imgsz=640', 'data=data.yaml'], capture_output=True, text=True)
    f.write(result.stdout)
    f.write(result.stderr)
