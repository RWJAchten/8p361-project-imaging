import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64, num_heads=4, depth=6, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4),
            num_layers=depth
        )
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = x[:, 0]  # CLS token
        x = self.mlp_head(x)
        return x

# Test het model
model = ViT()
x = torch.randn(1, 3, 32, 32)  # Simuleer een afbeelding (batch, channels, height, width)
out = model(x)
print(out.shape)  # Verwachte output: torch.Size([1, 10])
