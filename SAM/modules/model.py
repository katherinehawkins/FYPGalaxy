from statistics import mean
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW

import numpy as np
from transformers import SamModel
from sklearn.metrics import jaccard_score


class SAM(nn.Module):
    def __init__(self):
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")
        self.__freezeBackbone__()
    
    def fit(self, cfg):
        trainloader, valloader = cfg['trainloader'], cfg['valloader']
        epochs, save_path, self.device = cfg['epochs'], cfg['save_path'], cfg['device']
        
        self.optimizer = AdamW(self.model.mask_decoder.parameters(), lr=cfg['lr'])
        self.loss = nn.CrossEntropyLoss()

        bestLoss, bestScores = 1000, []
        with tqdm(range(epochs), desc='Training') as tepoch:
            self.model.train(True)
            tLoss, tScores = self.__step__(trainloader, update=True)

            self.model.eval()
            with torch.no_grad():
                vLoss, vScores = self.__step__(valloader, update=False)

            if vLoss < bestLoss:
                torch.save(self.model.state_dict(), save_path)
                bestLoss, bestScores = vLoss, vScores
            
            tepoch.set_postfix(tLoss=tLoss, tScores=tScores, vLoss=vLoss, vScores=vScores)

        self.recover(save_path)
        return bestLoss, bestScores

    def __step__(self, dataloader, update=False):
        meanLoss, meanScores = [], []
        for batch in dataloader:
            pixVal, inBox = batch['pixel_values'], batch['input_boxes']
            pixVal, inBox = pixVal.to(self.device), inBox.to(self.device)

            outputs = self.model(pixel_values=pixVal, 
                                 input_boxes=inBox, 
                                 multimask_output=False)
            
            gT = batch['ground_truth_mask']
            gT = gT.to(self.device)
            
            loss = self.loss(outputs.pred_masks, gT)
            scores = self.__metrics__(gT, outputs.pred_masks)
            meanScores.append(scores)

            if update:
                self.__updateWeights__(loss)

        meanScores = np.stack(meanScores)
        meanScores = np.mean(meanScores, axis=0)
        meanLoss = mean(meanLoss)
        return meanLoss, meanScores
    
    def __updateWeights__(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return None
    
    def __metrics__(self, gT, pred):
        iou = jaccard_score(gT, pred, average='micro')
        return np.array([iou])
    
    def __freezeBackbone__(self):
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        return None
    
    def recover(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        return None






        