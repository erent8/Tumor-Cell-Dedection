import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image
from tumor_detection_app import UNet

class TumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        print(f"Bulunan görüntü sayısı: {len(self.images)}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.png'))
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Dosya bulunamadı: {img_path} veya {mask_path}")
                return None
            
            # Görüntü ve maskeyi yükle
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Görüntü ve maskeyi aynı boyuta getir
            image = image.resize((256, 256))
            mask = mask.resize((256, 256))
            
            if self.transform:
                image = self.transform(image)
                mask = transforms.ToTensor()(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return None

def train_model():
    try:
        print("Eğitim başlıyor...")
        
        # Klasörlerin varlığını kontrol et
        if not os.path.exists('dataset/images') or not os.path.exists('dataset/masks'):
            print("Dataset klasörleri bulunamadı!")
            return
        
        # Cihaz seçimi
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Kullanılan cihaz: {device}")
        
        # Model oluştur
        model = UNet().to(device)
        print("Model oluşturuldu")
        
        # Veri dönüşümleri
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Veri setini yükle
        dataset = TumorDataset(
            image_dir='dataset/images',
            mask_dir='dataset/masks',
            transform=transform
        )
        
        if len(dataset) == 0:
            print("Dataset boş!")
            return
            
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"Dataloader oluşturuldu, batch sayısı: {len(dataloader)}")
        
        # Kayıp fonksiyonu ve optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Eğitim döngüsü
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for images, masks in dataloader:
                if images is None or masks is None:
                    continue
                    
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                print(f'Batch {batch_count}/{len(dataloader)} işlendi')
            
            avg_loss = total_loss/batch_count if batch_count > 0 else 0
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Modeli kaydet
        torch.save(model.state_dict(), 'tumor_model.pth')
        print("Model kaydedildi: tumor_model.pth")
        
    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {str(e)}")

if __name__ == "__main__":
    train_model() 