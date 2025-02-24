# Sıkıştırılmış Görüntü İyileştirme

## Kullanılan Veri Kaynakları
- **DIV2K**: Yüksek çözünürlüklü görüntü seti
- **Flickr2K**: Çeşitli görüntüleri içeren veri seti
- **COCO-Stuff**: Segmantasyon ve nesne tanıma için kullanılan veri seti

## Kod Başlıkları
- **ImageEnhancement.py**: Sıkıştırılmış görüntüleri iyileştirmek için basit bir CNN modeli kullanılıyor
- **Framework.py**: Farklı modellerin performansını değerlendiren bir benchmark frameworkü oluşturuyor
- **PyTorchSpeedUp.py**: ONNX ve TensorRT kullanarak modelin inference hızını artırmayı hedefliyor
- **DataPipeline.py**: Veri ön işleme ve artırma işlemlerini yönetiyor

