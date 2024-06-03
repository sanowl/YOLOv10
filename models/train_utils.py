import torch
import numpy as np
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(data_loader):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        optimizer.zero_grad()
        cls_pred, reg_pred = model(images)
        loss = criterion(cls_pred, reg_pred, targets['labels'], targets['boxes'])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate_one_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            cls_pred, reg_pred = model(images)
            loss = criterion(cls_pred, reg_pred, targets['labels'], targets['boxes'])
            
            total_loss += loss.item()
    return total_loss / len(data_loader)
