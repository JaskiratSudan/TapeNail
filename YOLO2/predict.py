import torch
from torchvision import transforms
from PIL import Image
import os
from train_model import PatternClassifier

def load_model(model_path, num_classes):
    model = PatternClassifier(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_path, device):
    # Define the same transform as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        
    return predicted.item() + 1  # Convert back to 1-based indexing

def predict_folder(model_path, folder_path, num_classes):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, num_classes).to(device)
    
    # Process all images in the folder
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            predicted_label = predict_image(model, image_path, device)
            results.append((filename, predicted_label))
            print(f"{filename}: Predicted Label {predicted_label}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict patterns in images')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images to predict')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes in the model')
    
    args = parser.parse_args()
    
    results = predict_folder(args.model_path, args.folder_path, args.num_classes)
    
    # Print summary
    print("\nPrediction Summary:")
    print("-" * 50)
    for filename, label in results:
        print(f"File: {filename:<30} Label: {label}") 