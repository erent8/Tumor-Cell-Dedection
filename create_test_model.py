import torch
from tumor_detection_app import UNet

# Test modeli oluştur
model = UNet()
torch.save(model.state_dict(), 'tumor_model.pth') 