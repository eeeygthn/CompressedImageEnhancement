import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

# Bu kodu görüntüleri yönetmek için yazdım.
# İlk olarak yüksek ve alçak çözünürlüklü görüntülerin olduğu klasörleri belirledim.
# Klasördeki görüntü dosyalarını bulup saklıyorum. Sonra yüksek ve alçak çözünürlüklü görüntülerin sayısını kontrol ediyorum.
# Eğer sayılar eşleşmiyorsa kod hata veriyor.
# Belirli bir indeksteki görüntüleri okuyup renk formatını değiştiriyor.
# Görüntüleri 0 ile 1 arasına ölçeklendiriyor.
# Eğer bir dönüşüm varsa bunu uyguluyor.
# Sonunda alçak ve yüksek çözünürlüklü görüntüleri döndürüyor.
# Böylece modelin eğitiminde kullanılacak görüntü çiftlerini hazırlıyorum.

# Bu class görüntü veri setini temsil ediyor
class ImageDataset(Dataset):
    # Classın başlangıç metodu yüksek ve alçak çözünürlüklü görüntülerin bulunduğu dosya ile başlıyor
    def __init__(self, high_res_folder, low_res_folder, transform=None):
        # Yüksek çözünürlüklü görüntülerin bulunduğu klasör
        self.high_res_folder = high_res_folder
        # Alçak çözünürlüklü görüntülerin bulunduğu klasör
        self.low_res_folder = low_res_folder
        # Transform varsa burada saklıyoruz
        self.transform = transform
        
        # Yüksek çözünürlüklü görüntü dosyalarını buluyoruz
        self.high_res_images = glob.glob(os.path.join(high_res_folder, "*.png"))
        # Alçak çözünürlüklü görüntü dosyalarını bulıyoruz
        self.low_res_images = glob.glob(os.path.join(low_res_folder, "*.png"))
        
        # Yüksek ve alçak çözünürlüklü görüntülerin sayısını kontrol ediyoruz
        assert len(self.high_res_images) == len(self.low_res_images), "Yüksek ve alçak çözünürlüklü görüntü sayısı eşleşmiyor"
        print("Yüksek çözünürlüklü görüntüler yüklendi")
        print("Alçak çözünürlüklü görüntüler yüklendi")
        print(f"Toplam yüksek çözünürlüklü görüntü sayısı: {len(self.high_res_images)}")
        print(f"Toplam alçak çözünürlüklü görüntü sayısı: {len(self.low_res_images)}")

    # Bu metot veri setinde kaç tane görüntü olduğunu veriyor 
    def __len__(self):
        return len(self.high_res_images)

    # Bu metot belirtilen indeksteki görüntüleri veriyor
    def __getitem__(self, index):
        # Yüksek çözünürlüklü görüntüyü okuyoruz
        high_res_image = cv2.imread(self.high_res_images[index])
        # Alçak çözünürlüklü görüntüyü okuyoruz
        low_res_image = cv2.imread(self.low_res_images[index])
        
        # Yüksek çözünürlüklü görüntüyü BGR'den RGB'ye çeviriyoruz
        high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB)
        # Alçak çözünürlüklü görüntüyü BGR'den RGB'ye çeviriyoruz
        low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)
        
        # Yüksek çözünürlüklü görüntüyü ölçeklendiriyoruz
        high_res_image = high_res_image / 255.0
        # Alçak çözünürlüklü görüntüyü ölçeklendiriyoruz
        low_res_image = low_res_image / 255.0

        # Eğer transform varsa bunu uyguluyoruz
        if self.transform:
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)

        # Alçak çözünürlüklü ve yüksek çözünürlüklü görüntüleri döndürüyoruz
        return low_res_image, high_res_image
