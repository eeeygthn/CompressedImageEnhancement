import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import onnx
import onnxruntime as ort
# Gerekli kütüphaneleri import ettim burada PyTorch ile modelimi eğitmek için ve ONNX ile modelimi kaydedip çalıştırmak için kullanacağım

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Basit bir CNN modeli tanımlıyorum burada iki tane konvolüsyon katmanı var ardından iki tane connected katmanı ekliyorum

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # Modelin geçişini tanımlıyorum burada girdi verisi konvolüsyon katmanlarından geçiyor daha sonra havuzlama işlemi yapılıyor

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Modelimi oluşturuyorum ayrıca kayıpları hesaplamak için Cross Entropy Loss kullanıyorum ve modelin parametrelerini optimize etmek için Adam algoritmasını seçiyorum

dummy_data = torch.randn(16, 3, 32, 32)  # 16 tane 32x32 RGB görüntü
dummy_labels = torch.randint(0, 10, (16,))  # 16 tane etiket
# Eğitim için kullanacağım dummy verileri ve etiketleri oluşturuyorum burada rastgele 16 tane 32x32 boyutunda RGB görüntü ve bu görüntülere karşılık gelen rastgele etiketler oluşturuyor

model.train()
for epoch in range(5):  # 5 epoch boyunca eğittim
    optimizer.zero_grad()
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
# Modeli eğitmeye başlıyorum burada 5 epoch boyunca modelimin eğitimini yapıyorum her epoch başında gradyanları sıfır

onnx_model_path = "simple_cnn.onnx"
torch.onnx.export(model, dummy_data, onnx_model_path, 
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
# Modeli ONNX formatına çevirmek için bu kısmı kullanıyorum burada modelin yapısını ve giriş çıkışlarını belirtiyor

ort_session = ort.InferenceSession(onnx_model_path)
# ONNX modelimi yüklemek için ONNX runtime oturumu açıyorum bu sayede modelimin inference işlemini gerçekleştirebileceğim

dummy_input = dummy_data.numpy()  # NumPy array'e çevirdim
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
# Dummy verileri NumPy array formatına çeviriyorum ve ONNX modeline input olarak vermek için uygun bir şekilde hazırlıyorum

ort_outs = ort_session.run(None, ort_inputs)
# Burada inference işlemini gerçekleştiriyorum modelin tahmin ettiği çıktıları alıyorum

try:
    import tensorrt as trt
    from tensorrt import InferenceEngine
    # TensorRT kütüphanesini import ediyorum bu kütüphane modelin daha hızlı çalışmasını sağlıyor
except ImportError:
    print("TensorRT kütüphanesi bulunamadı lütfen yükleyin")
    # Eğer TensorRT yüklenmemişse kullanıcıya bilgi veriyorum

def convert_to_tensorrt(onnx_model_path):
    with open(onnx_model_path, 'rb') as f:
        onnx_model = f.read()
        # ONNX modelini TensorRT'ye çevirmek için bir fonksiyon tanımlıyorum burada ONNX dosyasını okudu

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    # TensorRT için gerekli bileşenleri oluşturuyorum burada bir logger oluşturuyorum ve modelin yapısını tanımlamak için bir network oluşturuyorum

    if not parser.parse(onnx_model):
        print("ONNX modelini çevirirken hata")
        return None
    # ONNX modelini TensorRT'ye çevirmeye çalışıyorum eğer bir hata olursa kullanıcıya bilgi veriyor

    engine = builder.build_cuda_engine(network)
    return engine
    # Son olarak CUDA motorunu oluşturuyor bu motor TensorRT'nin optimizasyonlarından faydalanarak çalışacak şekilde ayarlanıyor

tensorrt_engine = convert_to_tensorrt(onnx_model_path)
if tensorrt_engine:
    print("Model TensorRT formatına çevrildi ve hazır")
    # ONNX modelini TensorRT motoruna dönüştürüyor

context = tensorrt_engine.create_execution_context()
inputs = np.array(dummy_input, dtype=np.float32)
outputs = np.empty((16, 10), dtype=np.float32)  # Çıktının şekli
bindings = [inputs.ctypes.data, outputs.ctypes.data]
# Inference işlemi için gerekli olan bağlamı oluşturuyorum burada motorun çalışması için gerekli olan girişleri ve çıkışları tanımlıyorum

context.execute(batch_size=16, bindings=bindings)
print("TensorRT Inference sonucu:", outputs)
# Son olarak TensorRT motorunu kullanarak inference işlemini gerçekleştiriyorum burada dummy girdi verisini kullanarak çıkış alıyorum ve sonucu yazdırıyorumm
