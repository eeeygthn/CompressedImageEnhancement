import os  # işletim sistemi ile ilgili fonksiyonları kullanmak için os kütüphanesini ekliyorum
import glob  # dosya yolunu kullanarak dosyaları bulmamı sağlayan glob kütüphanesini ekliyorum
import cv2  # görüntü işleme için OpenCV kütüphanesini ekliyorum
import numpy as np 
import torch  # PyTorch kütüphanesini ekliyorum
import torch.nn as nn
import torch.optim as optim  # optimizasyon işlemleri için optim modülünü ekliyorum
import torchvision.transforms as transforms  # görüntü dönüştürmeleri için torchvision kütüphanesini ekliyorum
from torch.utils.data import Dataset, DataLoader  # veri seti ve veri yükleyici için gerekli bileşenleri ekliyorum
import matplotlib.pyplot as plt  # görselleştirme için matplotlib kütüphanesini ekliyorum

# Görüntü Veri Seti Sınıfı
class ResimVeriSeti(Dataset):  # Dataset sınıfından türeyen ResimVeriSeti adında bir sınıf oluşturuyorum
    def __init__(self, klasor):  # sınıfın yapıcısı
        self.resimler = glob.glob(os.path.join(klasor, "*.png"))  # klasördeki png dosyalarını buluyorum

    def __len__(self):  # veri setinin uzunluğunu döndüren fonksiyon
        return len(self.resimler)  # toplam resim sayısı

    def __getitem__(self, index):  # belirli bir indekse göre öğe döndüren fonksiyon
        resim = cv2.imread(self.resimler[index])  # belirtilen resim dosyasını okuyoruz
        resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB) / 255.0  # resmi RGB formatına çevirip 0-1 aralığına getiriyorum
        return transforms.ToTensor()(resim)  # resmi tensöre çevirip döndürüyoruz

# Basit Bir CNN Modeli
class BasitCNN(nn.Module):  # nn.Module sınıfından türeyen BasitCNN adında bir sınıf oluşturuyoruz
    def __init__(self):  # fonksiyon
        super(BasitCNN, self).__init__() 
        self.konv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)  # ilk katman
        self.konv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # ikinci katman
        self.konv3 = nn.Conv2d(128, 64, kernel_size=5, padding=2)  # üçüncü katman
        self.konv4 = nn.Conv2d(64, 3, kernel_size=5, padding=2)  # sonucu elde ettiğim katman
        self.relu = nn.ReLU()  # fonksiyon olarak ReLU kullanıyorum
        self.maxpool = nn.MaxPool2d(2, 2)  # max pooling katmanı

    def forward(self, x):  # fonksiyon
        x = self.relu(self.konv1(x))  # ilk aktivasyon
        x = self.maxpool(x)  # pooling işlemi
        x = self.relu(self.konv2(x))  # ikinci aktivasyon
        x = self.maxpool(x)  # pooling işlemi
        x = self.relu(self.konv3(x))  # üçüncü aktivasyon
        x = self.konv4(x)
        return x  # sonlandırıyoruz

# PSNR Hesaplama Fonksiyonu
def psnr_hesapla(gercek, tahmin):  # PSNR hesaplayan fonksiyon
    mse = np.mean((gercek - tahmin) ** 2)  # ortalama kare hatası hesaplıyorum
    if mse == 0:  # sıfır hata olma durumu
        return float('inf')  # sonsuz döndürüyorux
    return 20 * np.log10(1.0 / np.sqrt(mse))  # PSNR değerini döndürüyoruz

# SSIM Hesaplama Fonksiyonu
def ssim_hesapla(gercek, tahmin):  # SSIM hesaplayan fonksiyon
    C1 = 6.5025  # sabit değer
    C2 = 58.5225  # sabit değer
    gercek = gercek.astype(np.float64)  # gerçek resmi float64 formatına çeviriyor
    tahmin = tahmin.astype(np.float64)  # tahmin edilen resmi float64 formatına çeviriyor
    mu1 = gercek.mean()  # gerçek resmin ortalamasını alıyor
    mu2 = tahmin.mean()  # tahmin edilen resmin ortalamasını alıyor
    sigma1_sq = gercek.var()  # gerçek resmin varyansı
    sigma2_sq = tahmin.var()  # tahmin edilen resmin varyansı
    sigma12 = np.cov(gercek.flatten(), tahmin.flatten())[0, 1]

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))  # SSIM hesaplıyorum
    return ssim  # SSIM değerini döndürüyorum

# Modellerin Performansını Değerlendirme
def modelleri_degerlendir(modeller, veri_seti):  # modellerin performansını değerlendiren fonksiyon
    psnr_sonuc = {} 
    ssim_sonuc = {}  

    for model_ismi, model in modeller.items():  # her bir model için
        model.eval()  # modeli değerlendirme moduna alıyorum
        psnr_puanlar = []  
        ssim_puanlar = []  

        with torch.no_grad():  # gradyan hesaplamalarını kapatıyorum
            for resim in veri_seti:  # veri setindeki her resim için
                tahmin = model(resim.unsqueeze(0))  # modeli tahmin ettiriyorum
                tahmin_np = tahmin.squeeze().permute(1, 2, 0).numpy()  # tahmin edilen resmi numpy dizisine çeviriyorum
                gercek_np = resim.permute(1, 2, 0).numpy()  # gerçek resmi numpy dizisine çeviriyorum

                psnr = psnr_hesapla(gercek_np, tahmin_np)
                ssim = ssim_hesapla(gercek_np, tahmin_np)

                psnr_puanlar.append(psnr)
                ssim_puanlar.append(ssim)

        psnr_sonuc[model_ismi] = np.mean(psnr_puanlar)
        ssim_sonuc[model_ismi] = np.mean(ssim_puanlar)

    return psnr_sonuc, ssim_sonuc

# Modeli Eğitme Fonksiyonu
def modeli_egit(model, veri_yukleyici, kriter, optimizer, epoch_sayisi):  # modeli eğiten fonksiyon
    model.train()  # modeli eğitim moduna alıyoruz
    for epoch in range(epoch_sayisi):  
        for resim in veri_yukleyici:  
            optimizer.zero_grad()  # gradyanları sıfırladık
            tahmin = model(resim)  # tahmini aldık
            kayip = kriter(tahmin, resim)  # kaybı hesapladık
            kayip.backward()  # geriye yayılım ile gradyanı hesapladık
            optimizer.step()  # ağırlıkları güncelledik
        print(f"Eğitim Epoch: {epoch + 1}/{epoch_sayisi}, Kayb: {kayip.item():.4f}")  # her epoch sonrası kaybı yazdırıyorum

# Sonuçların Görselleştirilmesi
def sonuclari_gorsellestir(modeller, veri_seti):  # sonuçları görselleştiren fonksiyon
    model = modeller["Model1"]
    model.eval()
    with torch.no_grad():
        for resim in veri_seti:
            tahmin = model(resim.unsqueeze(0))
            break

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 3 sütunlu bir figür oluşturuyorum
    axes[0].imshow(resim.permute(1, 2, 0).numpy())  # gerçek resmi görselleştiriyorum
    axes[0].set_title("Gerçek Yüksek Çözünürlük")
    axes[0].axis('off')  # eksenleri gizliyorum

    tahmin_np = tahmin.squeeze().permute(1, 2, 0).numpy()  # tahmin edilen resmi numpy dizisine çeviriyorum
    axes[1].imshow(tahmin_np)  # tahmin edilen resmi görselleştiriyorum
    axes[1].set_title("Tahmin Edilen Yüksek Çözünürlük")
    axes[1].axis('off')  # eksenleri gizli
    tahmin_np = tahmin_np * 255  # tahmin edilen resmi 0-255 aralığına getiriyorum
    tahmin_np = tahmin_np.astype(np.uint8)  # resmi uint8 formatına çeviriyorum
    axes[2].imshow(tahmin_np)  # tahmin edilen resmi yeniden görselleştiriyorum
    axes[2].set_title("Yeniden Yapılandırılmış Görüntü")
    axes[2].axis('off')  # eksenleri gizli

    plt.show()