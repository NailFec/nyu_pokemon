# utils.py
import torch
import json
import os

def load_label_map(filepath="pokemon/types2label.txt"):
    """
    Load label mapping file.
    Returns two dicts: type_string -> label_int and label_int -> type_string.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    json_content = content.replace("'", '"')
    type_to_label = json.loads(json_content)
    label_to_type = {v: k for k, v in type_to_label.items()}
    print(f"Label mapping loaded successfully, {len(type_to_label)} classes found.")
    return type_to_label, label_to_type

def save_checkpoint(state, is_best, folder="saved_models", filename="checkpoint.pth.tar"):
    """Save model checkpoint"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(folder, "best_model.pth.tar")
        torch.save(state, best_filepath)
        print("=> New best model saved.")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Optimizer state loaded.")
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    print(f"=> Checkpoint loaded. Resume training from epoch {start_epoch}, best accuracy so far: {best_acc:.2f}%")
    return model, optimizer, start_epoch, best_acc