from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

# Load file data.yaml trên laptop của bạn
data_info = check_det_dataset("dataset/processed/detection/data.yaml")

# Tạo cache cho tập train và val
YOLODataset(img_path=data_info['train'], data=data_info, cache=True)
YOLODataset(img_path=data_info['val'], data=data_info, cache=True)
