import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

class PatternDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.augment:
            image = self.aug_transform(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def analyze_dataset(folder_path):
    labels = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                label = int(filename.split('_')[1])
                labels.append(label)
            except (IndexError, ValueError):
                continue
    
    label_counts = Counter(labels)
    print("\nDataset Analysis:")
    print("-" * 50)
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} images")
    print("-" * 50)
    return label_counts

def load_dataset(folder_path):
    image_paths = []
    labels = []
    
    # Get all image files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                label = int(filename.split('_')[1])  # Get the label number
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(label - 1)  # Convert to 0-based indexing
            except (IndexError, ValueError):
                print(f"Skipping {filename}: doesn't match expected format")
    
    return image_paths, labels

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    if val_accuracies:
        plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model(folder_path, num_epochs=50, batch_size=32, learning_rate=0.001, min_samples_per_class=2):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Analyze dataset
    label_counts = analyze_dataset(folder_path)
    min_count = min(label_counts.values())
    
    # Load dataset
    image_paths, labels = load_dataset(folder_path)
    num_classes = len(set(labels))
    print(f"Found {len(image_paths)} images with {num_classes} different classes")
    
    # Determine if we should use validation split
    use_validation = min_count >= min_samples_per_class
    if not use_validation:
        print(f"\nWarning: Some classes have fewer than {min_samples_per_class} samples.")
        print("Training without validation split and using data augmentation.")
    
    # Split dataset if using validation
    if use_validation:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
    else:
        train_paths, train_labels = image_paths, labels
        val_paths, val_labels = [], []
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with augmentation for small datasets
    train_dataset = PatternDataset(train_paths, train_labels, transform, augment=not use_validation)
    if use_validation:
        val_dataset = PatternDataset(val_paths, val_labels, transform, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if use_validation:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = PatternClassifier(num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation if available
        if use_validation:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total
            
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), 'best_model.pth')
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        if use_validation:
            print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 50)
    
    # Plot training history
    plot_training_history(train_losses, val_losses if use_validation else [], 
                         train_accuracies, val_accuracies if use_validation else [])
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Final model saved as 'final_model.pth'")
    if use_validation:
        print("Best model saved as 'best_model.pth'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pattern classifier')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing labeled images')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min-samples', type=int, default=2, help='Minimum samples per class for validation split')
    
    args = parser.parse_args()
    
    train_model(
        args.folder_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_samples_per_class=args.min_samples
    ) 