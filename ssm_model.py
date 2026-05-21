import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalSelectiveSSMBlock(nn.Module):
    """
    Karmaşık fiziksel gösterimlerden arındırılmış, 
    yalnızca Seçici Tarama (Selective Scan) mantığına odaklanan Minimal S-SSM Bloğu.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Seçici (Selective) parametreler için doğrusal katmanlar.
        # Gelen veriye (x) bağlı olarak B, C ve dt (adım boyutu) değerlerini dinamik üretir.
        self.proj_b = nn.Linear(d_model, d_state)
        self.proj_c = nn.Linear(d_model, d_state)
        self.proj_dt = nn.Linear(d_model, d_model)
        
        # A matrisi (Durum geçişi) genellikle sabit başlatılır ve eğitilir
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x boyutu: (Batch, Sequence_Length, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Seçici parametrelerin (B, C, dt) üretilmesi
        dt = F.softplus(self.proj_dt(x)) # Adım boyutu daima pozitif olmalı
        B = self.proj_b(x)
        C = self.proj_c(x)
        
        # Çıktı tensörünü hazırlama
        y = torch.zeros_like(x)
        
        # Başlangıç durumu (Gizli bellek / Hidden state)
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        
        # Dizilim (Sequence) üzerinde zaman adımlı tarama (Minimal Recurrent Loop)
        for t in range(seq_len):
            xt = x[:, t, :] # (Batch, d_model)
            dt_t = dt[:, t, :].unsqueeze(-1) # (Batch, d_model, 1)
            Bt = B[:, t, :].unsqueeze(1) # (Batch, 1, d_state)
            Ct = C[:, t, :].unsqueeze(1) # (Batch, 1, d_state)
            
            # Euler ayrıklaştırması (Discretization) - Mantıksal temel
            # h_t = h_{t-1} * exp(A * dt) + (B * dt) * x_t
            delta_A = torch.exp(self.A * dt_t) 
            delta_B = Bt * dt_t * xt.unsqueeze(-1)
            
            h = h * delta_A + delta_B
            
            # Çıktı hesaplama: y_t = C * h_t + D * x_t
            yt = torch.sum(h * Ct, dim=-1) + self.D * xt
            y[:, t, :] = yt
            
        return y

class ScaledSSM(nn.Module):
    """
    ExDark (veya benzeri) görüntü sınıflandırma görevleri için
    Minimal S-SSM bloklarını kullanan kapsayıcı (Wrapper) model.
    """
    def __init__(self, num_classes=12, d_model=128, num_blocks=6):
        super().__init__()
        self.d_model = d_model
        
        # 2D Görüntüyü S-SSM'in anlayabileceği 1D diziye (sequence) çeviren katman
        # 224x224 görüntü -> 16x16 patch'ler -> 196 uzunluğunda dizi
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        
        # S-SSM Bloklarını üst üste ekleme
        self.blocks = nn.ModuleList([
            MinimalSelectiveSSMBlock(d_model=d_model) 
            for _ in range(num_blocks)
        ])
        
        # Normalizasyon ve Sınıflandırma başlığı
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Görüntüyü parçalara böl ve düzleştir: (B, C, H, W) -> (B, Sequence_Length, d_model)
        x = self.patch_embed(x) 
        x = x.flatten(2).transpose(1, 2) 
        
        # S-SSM bloklarından geçiş
        for block in self.blocks:
            # Residual connection (Artık bağlantı) ile gradyan akışını güçlendir
            x = x + block(x)
            
        x = self.norm(x)
        
        # Dizinin ortalamasını alıp (Global Average Pooling) sınıflandırma başlığına gönder
        x_pooled = x.mean(dim=1) 
        logits = self.head(x_pooled)
        return logits
