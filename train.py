import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os

from dataset import PokemonDataset
from model import get_pokemon_model, unfreeze_all_layers
from utils import load_label_map, save_checkpoint

# --- 1. Hyperparameters and Settings ---
# Path settings
DATA_PATH = "pokemon/pokemon_train/train_data.npy"
LABEL_PATH = "pokemon/pokemon_train/train_labels.npy"
LABEL_MAP_PATH = "pokemon/types2label.txt"
MODEL_SAVE_DIR = "saved_models"
PRETRAINED_WEIGHTS_PATH = "resnet50-11ad3fa6.pth"

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
VAL_SPLIT = 0.15
WEIGHT_DECAY = 1e-4
UNFREEZE_EPOCH = 5

# --- 2. Data Preprocessing and Loading ---
_, label_to_type = load_label_map(LABEL_MAP_PATH)
NUM_CLASSES = len(label_to_type)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize,
])
val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    normalize,
])

def train_one_epoch(loader, model, optimizer, criterion, device):
    model.train()
    loop = tqdm(loader, desc="Training...")
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(loader)
    print(f"Train epoch average loss: {avg_loss:.4f}")

def check_accuracy(loader, model, criterion, device):
    model.eval()
    num_correct = 0
    num_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating..."):
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = criterion(scores, y)
            total_loss += loss.item()
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    acc = (num_correct / num_samples) * 100
    avg_loss = total_loss / len(loader)
    print(f"Validation accuracy: {acc:.2f}%")
    print(f"Validation average loss: {avg_loss:.4f}")
    return acc, avg_loss

def main():
    print(f"Device: {DEVICE}")
    
    # --- Dataset and Dataloaders ---
    full_dataset = PokemonDataset(data_path=DATA_PATH, labels_path=LABEL_PATH, transform=None)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # --- Model, Loss function, Optimizer ---
    model = get_pokemon_model(
        num_classes=NUM_CLASSES, 
        use_pretrained_weights=True, 
        weights_path=PRETRAINED_WEIGHTS_PATH
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_acc = 0.0
    
    # --- Training loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        if epoch == UNFREEZE_EPOCH:
            model = unfreeze_all_layers(model)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE / 10, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)
            print(f"Model unfrozen. Learning rate adjusted to {LEARNING_RATE / 10}")

        train_one_epoch(train_loader, model, optimizer, criterion, DEVICE)
        val_acc, val_loss = check_accuracy(val_loader, model, criterion, DEVICE)
        scheduler.step(val_acc)
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }, is_best, folder=MODEL_SAVE_DIR)

if __name__ == "__main__":
    main()