import torch
import torch.nn as nn
import torchvision.models as models
import os

def get_pokemon_model(num_classes, use_pretrained_weights=True, weights_path=None):
    """
    Get a ResNet-50 model.
    If use_pretrained_weights is True, tries to load weights.
    If weights_path is provided and exists, loads from local file.
    Otherwise, tries to download from torchvision default URL.
    """
    weights = None
    if use_pretrained_weights and weights_path and os.path.exists(weights_path):
        print(f"Local weights file detected: '{weights_path}'. Loading from file.")
        model = models.resnet50(weights=None)
        try:
            model.load_state_dict(torch.load(weights_path))
            print("Local pretrained weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load local weights: {e}. Will try online download or use random weights.")
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
    elif use_pretrained_weights:
        print("No valid local weights path provided, trying to load online pretrained weights...")
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        print("Not using pretrained weights, model will start from random weights.")
        model = models.resnet50(weights=None)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the fully connected layer
    num_ftrs = model.fc.in_features

    # Replace the original fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    print("Model structure created (ResNet-50 base).")
    print(f"Final layer replaced for {num_classes}-class classification.")
        
    return model

def unfreeze_all_layers(model):
    """Unfreeze all layers of the model for fine-tuning"""
    for param in model.parameters():
        param.requires_grad = True
    print("All model layers unfrozen, ready for end-to-end fine-tuning.")
    return model