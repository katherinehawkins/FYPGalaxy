from statistics import mean
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import numpy as np
from torchmetrics.functional.classification import binary_jaccard_index, binary_accuracy

class SAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        self.__freezeBackbone__()
    
    def fit(self, cfg):
        trainloader, valloader = cfg['trainloader'], cfg['valloader']
        epochs, save_path, self.device = cfg['epochs'], cfg['save_path'], cfg['device']
        
        self.optimizer = AdamW(self.model.mask_decoder.parameters(), lr=cfg['lr'])
        self.loss = nn.BCEWithLogitsLoss()
        self.model = self.model.to(self.device)

        bestLoss, bestScores = 1000, []
        with tqdm(range(epochs), desc='Training') as tepoch:
            self.model.train(True)
            tLoss, tScores = self.epoch(trainloader, update=True)

            self.model.eval()
            with torch.no_grad():
                vLoss, vScores = self.epoch(valloader, update=False)

            if vLoss < bestLoss:
                self.save(save_path)
                bestLoss, bestScores = vLoss, vScores
            
            tepoch.set_postfix(tLoss=tLoss, tScores=tScores, vLoss=vLoss, vScores=vScores)

        self.recover(save_path)
        return bestLoss, bestScores

    def epoch(self, dataloader, update=False):
        meanLoss, meanScores = [], []
        for batch in dataloader:
            pixVal, inBox = batch['pixel_values'], batch['input_boxes']
            labels, gT = batch['input_labels'], batch['ground_truth_mask']
            
            pixVal, inBox = pixVal.to(self.device), inBox.to(self.device)
            labels, gT = labels.to(self.device), gT.to(self.device)

            outputs = self.model(pixel_values=pixVal, 
                                 input_boxes=inBox, 
                                 input_labels=labels,
                                 multimask_output=False)

            gT = gT[labels != -10]
            logit = outputs.pred_masks[labels != -10]
            
            h, w = gT.shape[-1], gT.shape[-2]
            logit = F.interpolate(logit, (h,w), mode='bilinear', align_corners=False)
            logit = torch.squeeze(logit)
            
            loss = self.loss(logit, gT)
            if update:
                self.__updateWeights__(loss)
            
            pred = F.sigmoid(logit)
            scores = self.__metrics__(gT, pred)
            meanScores.append(scores)
            meanLoss.append(loss)

        meanScores = np.stack(meanScores)
        return mean(meanLoss), np.mean(meanScores, axis=0)
    
    def __updateWeights__(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return None
    
    def __metrics__(self, gT, pred):
        gT, pred = gT.cpu(), pred.detach().cpu()
        iou = binary_jaccard_index(pred, gT)
        acc = binary_accuracy(pred, gT)
        return np.array([iou, acc])
    
    def __freezeBackbone__(self):
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        return None
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        return None

    def recover(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        return None






        