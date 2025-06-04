import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_fscore_support
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

# Detect available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset parameters
data_dir = './dataset'
input_size = 224
batch_size = 32
num_epochs = 100
learning_rate = 1e-4

# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),  # Random translation
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(data_dir)
class_names = full_dataset.classes
num_classes = len(class_names)

# Split dataset (80% train, 10% validation, 10% test)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Apply different transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load a pre-trained network (e.g., ResNet18, AlexNet, VGG, GoogLeNet, etc.)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Parameters except the last fully connected layer
base_params = [param for name, param in model.named_parameters() if "fc" not in name]

# Parameters of the last fully connected layer
fc_params = model.fc.parameters()

optimizer = optim.SGD([
    {'params': base_params, 'lr': learning_rate},
    {'params': fc_params, 'lr': learning_rate * 10}
], momentum=0.9)

# Training function
def train_model():
    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate statistics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_acc = correct / total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Accuracy: {epoch_acc:.4f}')

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.savefig('training_metrics.png')
    plt.show()

    return train_losses, val_losses, accuracies

# Train the model
train_metrics = train_model()

# Test the model and calculate various metrics
def test_model():
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)  # Get prediction probabilities
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Calculate accuracy, precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None,
                                                               labels=range(num_classes))
    accuracy = np.sum(all_labels == all_preds) / len(all_labels)

    print(f"Overall Test Accuracy: {accuracy:.4f}")
    for i, cls in enumerate(class_names):
        print(f"Class {cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1[i]:.4f}")

    # Calculate Specificity (per class in multiclass scenario)
    print("Specificity per class:")
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        print(f"Class {class_names[i]} Specificity: {specificity:.4f}")

    # Calculate AUC (multiclass scenario, one-vs-rest approach)
    auc_scores = []
    for i in range(num_classes):
        # Convert true labels to binary format
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')  # Skip AUC calculation if a class has no positive examples in test set
        auc_scores.append(auc)
        print(f"Class {class_names[i]} AUC: {auc:.4f}")

    # Save results
    np.savez('test_results.npz',
             labels=all_labels,
             preds=all_preds,
             probs=all_probs,
             class_names=class_names,
             confusion_matrix=cm,
             accuracy=accuracy,
             precision=precision,
             recall=recall,
             f1=f1,
             auc=auc_scores,
             specificity=specificity)

    return accuracy, cm

# Training call omitted
test_acc, conf_matrix = test_model()

# Define feature extraction and dimensionality reduction function
def extract_features_and_tsne(model, data_loader, device):

    # Define the layers to extract features from — both the selection and number of layers can be flexibly adjusted based on the model architecture
    layers = {
        'conv1': model.conv1,
        'layer1_0_conv1': model.layer1[0].conv1,
        'layer1_0_conv2': model.layer1[0].conv2,
        'layer1_1_conv1': model.layer1[1].conv1,
        'layer1_1_conv2': model.layer1[1].conv2,
        'layer2_0_conv1': model.layer2[0].conv1,
        'layer2_0_conv2': model.layer2[0].conv2,
        'layer2_1_conv1': model.layer2[1].conv1,
        'layer2_1_conv2': model.layer2[1].conv2,
        'layer3_0_conv1': model.layer3[0].conv1,
        'layer3_0_conv2': model.layer3[0].conv2,
        'layer3_1_conv1': model.layer3[1].conv1,
        'layer3_1_conv2': model.layer3[1].conv2,
        'layer4_0_conv1': model.layer4[0].conv1,
        'layer4_0_conv2': model.layer4[0].conv2,
        'layer4_1_conv1': model.layer4[1].conv1,
        'layer4_1_conv2': model.layer4[1].conv2,
        'fc': model.fc
    }

    # Create storage space for each layer
    all_features = {name: [] for name in layers}
    labels_list = []

    # Hook function to store features
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Register forward hooks
    hooks = []
    for name, layer in layers.items():
        hook = layer.register_forward_hook(get_features(name))
        hooks.append(hook)

    # Set model to evaluation mode
    model.eval()
    total_samples = 0

    # Disable gradient computation
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Extracting features"):

            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass (trigger hooks)

            # Process current batch of features
            batch_size = inputs.size(0)

            for name in layers:
                # Get features and apply global average pooling
                feat = features[name][:batch_size]

                if feat.ndim == 4:  # Convolutional features [B, C, H, W]
                    feat = feat.mean(dim=[2, 3])  # Global average pooling
                elif feat.ndim == 2:  # Fully connected features [B, F]
                    pass  # Use directly
                else:
                    # Handle other dimensions
                    feat = feat.view(feat.size(0), -1)

                all_features[name].append(feat.cpu().numpy())

            labels_list.append(labels[:batch_size].numpy())
            total_samples += batch_size


    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate features
    for name in all_features:
        all_features[name] = np.concatenate(all_features[name], axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"Extracted features from {len(layers)} layers for {total_samples} samples")

    # Adjust perplexity dynamically based on sample size
    if total_samples < 50:
        perplexity = max(5, total_samples // 3)  # Ensure at least 5
        print(f"Adjusting perplexity to {perplexity} due to small sample size ({total_samples})")
    else:
        perplexity = 30

    # Perform t-SNE dimensionality reduction
    tsne_results = pd.DataFrame()

    for i, (name, feats) in enumerate(tqdm(all_features.items(), desc="Running t-SNE")):
        # Skip layers with too few samples
        if feats.shape[0] < 10:
            print(f"Skipping layer {name} due to insufficient samples ({feats.shape[0]})")
            # Add empty columns to maintain structure
            tsne_results[f'Var{i * 2 + 1}'] = np.nan
            tsne_results[f'Var{i * 2 + 2}'] = np.nan
            continue

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(feats)

        # Store results
        tsne_results[f'Var{i * 2 + 1}'] = reduced[:, 0]
        tsne_results[f'Var{i * 2 + 2}'] = reduced[:, 1]

    # Save results to Excel
    tsne_results.to_excel('Deep_Phenotype.xlsx', index=False)
    print("t-SNE results saved to Deep_Phenotype.xlsx")

    # Visualize last layer's t-SNE result (if available)
    if 'fc' in all_features and all_features['fc'].shape[0] >= 10:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results.iloc[:, -2], tsne_results.iloc[:, -1],
                              c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Class')
        plt.title('t-SNE Visualization of Last Layer Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig('tsne_visualization.png')
        plt.show()
    else:
        print("Skipping visualization due to insufficient samples in the last layer")

    return all_features, tsne_results

# Create a full data loader (includes all data)
# Use test transforms since no data augmentation is needed
full_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Call the feature extraction function (using full dataset)
extract_features_and_tsne(model, full_loader, device)