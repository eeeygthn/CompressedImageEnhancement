import os  
import glob  # Dosya yollarını işlemek için kullanılıyor
import cv2  # Görüntü işleme için OpenCV kütüphanesi
import numpy as np  # Sayısal işlemler için NumPy kütüphanesi
import torch  # PyTorch kütüphanesi
import torch.nn as nn  
import torch.optim as optim  # Optimizasyon teknikleri için PyTorch modülü
import torchvision.transforms as transforms  # Görüntü dönüşümleri için
from torch.utils.data import Dataset, DataLoader  # Veri yükleme ve yönetimi için
import matplotlib.pyplot as plt  # Görselleştirme için Matplotlib kütüphanesi


# Bu sınıfı yüksek çözünürlüklü ve alçak çözünürlüklü görüntüleri yüklemek için kullandık
class ImageDataset(Dataset):
    def __init__(self, high_res_folder, low_res_folder):
        # Yüksek ve alçak çözünürlüklü görüntü dosyaları
        self.high_res_images = glob.glob(os.path.join(high_res_folder, "*.png"))
        self.low_res_images = glob.glob(os.path.join(low_res_folder, "*.png"))
        # Yüksek ve alçak çözünürlüklü görüntü sayılarının eşleştiğini kontrol ediyoruz
        assert len(self.high_res_images) == len(self.low_res_images), "Yüksek ve alçak çözünürlüklü görüntü sayısı eşleşmiyor"

    def __len__(self):
        # Veri setinin uzunluğunu
        return len(self.high_res_images)

    def __getitem__(self, index):
        # Belirtilen indeksin yüksek ve alçak çözünürlüklü görüntülerini burada yükledim
        high_res_img = cv2.imread(self.high_res_images[index])  # Yüksek çözünürlüklü görüntü
        low_res_img = cv2.imread(self.low_res_images[index])  # Alçak çözünürlüklü görüntü
        
        # Görüntüleri RGB formatına çevirip [0 1] aralığına çeviriyorum
        high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB) / 255.0
        low_res_img = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Görüntüleri tensör formatına çevirip döndürüyorum
        return transforms.ToTensor()(low_res_img), transforms.ToTensor()(high_res_img)

# Basit CNN Modeli
# Görüntü iyileştirmesi için kullanacağım basit bir sinir ağı modeli tanımlıyorum
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # İlk katman Girişten 64 filtre çıkaran katman
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        # İkinci katman 64 filtreden 128 filtre çıkaran katman
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        # Üçüncü katman 128 filtreden 64 filtre çıkaran katman
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        # Dördüncü katman 64 filtreden 3 filtre çıkaran katman (sonuç)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()  # Aktivasyon fonksiyonu olarak ReLU kullanıyorum
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling katmanı

    def forward(self, input_tensor):
        # Önce birinci konvolüsyon sonra ReLU ve son olarak MaxPooling uyguluyorum
        x = self.relu(self.conv1(input_tensor))
        x = self.pool(x)
        # Aynı işlemleri ikinci ve üçüncü katmanlar için tekrarlıyorum
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        # Sonuçları çıkartıyorum
        x = self.conv4(x)
        return x

# Modeli Eğitme Fonksiyonu
# Modelin eğitimini gerçekleştiren bir fonksiyon tanımlıyorum
def train_model(model, dataloader, loss_function, optimizer, total_epochs):
    model.train()  # Modeli eğitim moduna alıyorum
    for epoch in range(total_epochs):
        for low_res_images, high_res_images in dataloader:
            optimizer.zero_grad()  # Gradienten sıfırlıyorum
            outputs = model(low_res_images)  # Modelin çıktısını alıyorum
            loss = loss_function(outputs, high_res_images)  # Kaybı hesaplıyorum
            loss.backward()  # Gerileyen gradientleri hesaplıyorum
            optimizer.step()  # Optimizasyonu güncelliyorum
        # Her epoch sonunda kaybı yazdırıyorum
        print(f"Epoch [{epoch+1}/{total_epochs}] Loss: {loss.item():.4f}")

# Görüntülerin Görselleştirilmesi
# Eğitimden sonra sonuçları görselleştiren bir fonksiyon tanımlıyorum
def visualize_results(model, dataloader):
    model.eval()  # Modeli değerlendirme moduna alıyorum
    with torch.no_grad():  # Gradienten hesaplamayacak şekilde modelimi kullanıyorum
        for low_res_images, high_res_images in dataloader:
            outputs = model(low_res_images)  # Modelden tahmin edilen çıktıyı alıyorum
            break  # İlk batch ile işlemi sonlandırıyorum

    # Görselleştirme için
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(low_res_images[0].permute(1, 2, 0).numpy())  # Alçak çözünürlüklü görüntü
    axes[0].set_title("Alçak Çözünürlük")
    axes[0].axis('off')  # Eksenleri kapatıyorum

    axes[1].imshow(outputs[0].permute(1, 2, 0).numpy())  # Tahmin edilen yüksek çözünürlük
    axes[1].set_title("Tahmin Edilen Yüksek Çözünürlük")
    axes[1].axis('off')  # Eksenleri kapatıyorum

    axes[2].imshow(high_res_images[0].permute(1, 2, 0).numpy())  # Gerçek yüksek çözünürlük
    axes[2].set_title("Gerçek Yüksek Çözünürlük")
    axes[2].axis('off')  # Eksenleri kapatıyorum

    plt.show()  # Görselleştirme
# Ana Fonksiyon
def main():
    # Yüksek ve alçak çözünürlüklü görüntülerin tanımlanması
    high_res_folder = "path_to_high_res_images"  # Yüksek çözünürlüklü görüntüler
    low_res_folder = "path_to_low_res_images"  # Alçak çözünürlüklü görüntüler

    # Veri setini oluşturuyorum ve DataLoader ile yükleme işlemi gerçekleştiriyorum
    dataset = ImageDataset(high_res_folder, low_res_folder)  # Veri setini başlatıyorum
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Veri yükleyici oluşturuyorum

    # Modelimin kaybını ve optimizasyonumu tanımlıyorum
    model = SimpleCNN()  # Modelimi başlatıyorum
    loss_function = nn.MSELoss()  # Kaybı hesaplamak için MSE kullanıyorum
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizasyonunu seçiyorum

    # Modeli eğitiyorum
    train_model(model, dataloader, loss_function, optimizer, total_epochs=10)  # Modeli eğitiyorum
    # sonuçları görselleştiriyorum
    visualize_results(model, dataloader)

if __name__ == "__main__":
    main()
