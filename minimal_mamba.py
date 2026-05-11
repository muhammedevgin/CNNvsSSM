import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis

class MinimalSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Selective projections: B, C, and Delta are input-dependent
        self.proj_b = nn.Linear(d_model, d_state, bias=False)
        self.proj_c = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=False)
        
        # Static parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        out = []
        
        # Unrolled sequential scan (Useful for FLOP tracing)
        for t in range(seq_len):
            xt = x[:, t, :] # Current token: (batch, d_model)
            
            # 1. Compute selective parameters
            bt = self.proj_b(xt) # (batch, d_state)
            ct = self.proj_c(xt) # (batch, d_state)
            dt = F.softplus(self.proj_delta(xt)) # (batch, d_model)
            
            # 2. Discretization (Simplified Euler approximation for profiling)
            # dA shape: (batch, d_model, d_state)
            dA = torch.exp(torch.einsum('bd,ds->bds', dt, self.A))
            dB = torch.einsum('bd,bs->bds', dt, bt)
            
            # 3. State Update
            h = dA * h + torch.einsum('bds,bd->bds', dB, xt)
            
            # 4. Compute Output
            yt = torch.einsum('bds,bs->bd', h, ct) + self.D * xt
            out.append(yt)
            
        return torch.stack(out, dim=1)

class UAVVisionMamba(nn.Module):
    def __init__(self, image_size=224, patch_size=16, d_model=64, d_state=16, num_classes=12):
        super().__init__()
        
        # 1. Patch Embedding: Converts 2D image into 1D sequence of patches
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # 2. The S-SSM Backbone
        self.mamba_block = MinimalSelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        
        # 3. Classification Head (for ExDark's 12 classes)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.patch_embed(x)             # Output: (batch, d_model, 14, 14)
        x = x.flatten(2).transpose(1, 2)    # Output: (batch, 196, d_model)
        
        # Process the sequence of patches
        x = self.mamba_block(x)
        x = self.norm(x)
        
        # Global Average Pooling across the sequence
        x = x.mean(dim=1)                   
        return self.head(x)

if __name__ == "__main__":
    # Create the model and dummy UAV low-light input
    model = UAVVisionMamba(d_model=64, d_state=16)
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224)

    # Calculate Theoretical S-SSM Target
    flops = FlopCountAnalysis(model, dummy_image)
    gflops = flops.total() / 1e9
    
    print("------------------------------------------------")
    print(f"UAV Vision Mamba Architecture")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")
    print(f"Theoretical Complexity: {gflops:.4f} GFLOPs")
    print("------------------------------------------------")