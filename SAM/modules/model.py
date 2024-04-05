from statistics import mean
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import numpy as np
from torchmetrics.functional.classification import binary_jaccard_index, binary_accuracy

def freezeBackbone(model):
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    return model

def metrics(gT, pred):
    gT, pred = gT.cpu(), pred.detach().cpu()
    iou = binary_jaccard_index(pred, gT)
    acc = binary_accuracy(pred, gT)

    res = [iou.item(),
           acc.item()]
    return np.array(res)

class SAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = freezeBackbone(model)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values, input_boxes, input_labels, ground_truth):
        outputs = self.model(pixel_values=pixel_values, 
                             input_boxes=input_boxes, 
                             input_labels=input_labels,
                             multimask_output=False)

        ground_truth = ground_truth[input_labels != -10]
        logit = outputs.pred_masks[input_labels != -10]
        
        h, w = ground_truth.shape[-1], ground_truth.shape[-2]
        logit = F.interpolate(logit, (h,w), mode='bilinear', align_corners=False)
        return torch.squeeze(logit), ground_truth
    
    def fit(self, cfg):
        trainloader, valloader = cfg['trainloader'], cfg['valloader']
        epochs, save_path = cfg['epochs'], cfg['save_path']

        self.setDevice(cfg['device'])
        self.configureOptimizer(cfg['lr'])

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
            pixVal = batch['pixel_values'].to(self.device)
            inBox = batch['input_boxes'].to(self.device)
            labels = batch['input_labels'].to(self.device)
            gT = batch['ground_truth_mask'].to(self.device)

            logit, gT = self.__forward__(pixVal, inBox, labels, gT)
            
            loss = self.loss(logit, gT)
            if update:
                self.__updateWeights__(loss)
            
            scores = metrics(gT, logit) # not true average across all data in a dataset
            meanScores.append(scores)
            meanLoss.append(loss)

        meanScores = np.stack(meanScores)
        return mean(meanLoss), np.mean(meanScores, axis=0)
    
    def __updateWeights__(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None
    
    def configureOptimizer(self, lr):
        self.optimizer = AdamW(self.model.mask_decoder.parameters(), lr=lr)
    
    def setDevice(self, device):
        self.device = device
        self.model = self.model.to(device)
        return None
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        return None

    def recover(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        return None






        