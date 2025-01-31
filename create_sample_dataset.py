import os
import numpy as np
from PIL import Image

def create_sample_dataset():
    # Klasörleri oluştur
    os.makedirs('dataset/images', exist_ok=True)
    os.makedirs('dataset/masks', exist_ok=True)
    
    # Örnek görüntüler oluştur
    for i in range(10):
        # Rastgele görüntü (256x256 boyutunda)
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f'dataset/images/sample_{i}.jpg')
        
        # Rastgele maske (aynı boyutta)
        mask_array = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        mask = Image.fromarray(mask_array)
        mask.save(f'dataset/masks/sample_{i}_mask.png')

if __name__ == "__main__":
    create_sample_dataset()
    print("Örnek dataset oluşturuldu!") 