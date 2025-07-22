# predict.py
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

from model import get_pokemon_model
from utils import load_label_map, load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/best_model.pth.tar"
LABEL_MAP_PATH = "pokemon/types2label.txt"

def predict_image(image_path, model, label_map, device):
    """Predict the type of a single Pokemon image."""
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return

    # Load label map
    _, label_to_type = label_map

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    except Exception as e:
        print(f"Failed to load or process image: {e}")
        return

    # Model inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)

    predicted_label = predicted_idx.item()
    predicted_type = label_to_type[predicted_label]

    print(f"\nPrediction for '{os.path.basename(image_path)}':")
    print(f"  -> Pokemon type: {predicted_type} (class index: {predicted_label})")

if __name__ == "__main__":
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model '{MODEL_PATH}' not found. Please run train.py first.")
        sys.exit(1)
        
    # Load the model
    print("Loading model...")
    label_map = load_label_map(LABEL_MAP_PATH)
    num_classes = len(label_map[1])
    
    model = get_pokemon_model(num_classes, pretrained=False).to(DEVICE) # pretrained=False because we are loading our own weights
    model, _, _, _ = load_checkpoint(MODEL_PATH, model)
    print("Model loaded.")

    # Get the image path from command line arguments, or use a sample path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(image_path, model, label_map, DEVICE)
    else:
        print("\nUsage: python predict.py <your_image_path>")
        print("Example: python predict.py my_pokemon_image.jpg")
        # You can place a test image in the root directory, named test.jpg, and uncomment the following lines to test
        # if os.path.exists("test.jpg"):
        #     predict_image("test.jpg", model, label_map, DEVICE)