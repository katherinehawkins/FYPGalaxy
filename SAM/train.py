import sys
sys.path.append('/home/msuleman/ml20_scratch/fyp_galaxy')

import numpy as np
import random
import torch

from torch.utils.data import DataLoader

from modules.dataset import SAMDataset
from modules.model import SAMWrapper

from transformers import SamProcessor, SamModel

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)

reset_seed(n=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, processor)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, processor)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': args.epochs, 
       'lr': args.lr,
       'save_path': args.save_path, 
       'device': device}

model = SAMWrapper(model)
vLoss, vScores = model.fit(cfg)

print(vLoss)
print(vScores)