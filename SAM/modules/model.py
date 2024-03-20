import torch.nn as nn
from torch.optim import AdamW
from transformers import SamModel

class SAM(nn.Module):
    def __init__(self):
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")

        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        
        self.optimizer = AdamW(self.model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    
    def fit(dataloader, epochs, lr):
        self.model
        