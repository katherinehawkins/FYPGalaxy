import numpy as np
import random
import torch

from torch.utils.data import DataLoader

from modules.dataset import SAMDataset
from modules.model import SAM

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

root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

cfg = {'trainloader': trainloader, 
       'valloader': valloader, 
       'epochs': args.epochs, 
       'save_path': args.save_path, 
       'device': device}

vLoss, vScores = SAM().fit(cfg)

print(vLoss)
print(vScores)