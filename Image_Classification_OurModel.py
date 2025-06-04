import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, \
    precision_recall_fscore_support
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

# Custom model class
import torch
import torch.nn as nn
import torch.nn.functional as F


# Hybrid Attention Module (Channel + Spatial)
class HybridAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        # Channel attention branch
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention branch
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * channel_att * spatial_att


# Basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# Main network architecture
class OurModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Input layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1: res2 (64 channels)
        self.stage1 = self._make_stage(64, 64, stride=1)
        # Stage 2: res3 (128 channels)
        self.stage2 = self._make_stage(64, 128, stride=2)
        # Stage 3: res4 (256 channels)
        self.stage3 = self._make_stage(128, 256, stride=2)
        # Stage 4: res5 (512 channels)
        self.stage4 = self._make_stage(256, 512, stride=2)

        # Attention modules
        self.attn_stage1 = HybridAttention(64)
        self.attn_stage2 = HybridAttention(128)
        self.attn_stage3 = HybridAttention(256)
        self.attn_stage4 = HybridAttention(512)

        # Global average pooling layers
        self.gapool1 = nn.AdaptiveAvgPool2d(1)
        self.gapool2 = nn.AdaptiveAvgPool2d(1)
        self.gapool3 = nn.AdaptiveAvgPool2d(1)
        self.gapool4 = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(960, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        blocks = [
            ResidualBlock(in_channels, out_channels, stride, downsample),
            ResidualBlock(out_channels, out_channels)
        ]

        return nn.Sequential(*blocks)

    def forward(self, x):
        # Input layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1
        s1 = self.stage1(x)
        s1_att = self.attn_stage1(s1)
        gap1 = self.gapool1(s1_att)

        # Stage 2
        s2 = self.stage2(s1)
        s2_att = self.attn_stage2(s2)
        gap2 = self.gapool2(s2_att)

        # Stage 3
        s3 = self.stage3(s2)
        s3_att = self.attn_stage3(s3)
        gap3 = self.gapool3(s3_att)

        # Stage 4
        s4 = self.stage4(s3)
        s4_att = self.attn_stage4(s4)
        gap4 = self.gapool4(s4_att)

        # Feature fusion
        fused = torch.cat([
            gap1.view(gap1.size(0), -1),
            gap2.view(gap2.size(0), -1),
            gap3.view(gap3.size(0), -1),
            gap4.view(gap4.size(0), -1)
        ], dim=1)

        # Fully connected layer
        out = self.fc(fused)
        return out


# Test code
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    model = OurModel(num_classes=num_classes).to(device)
    print("Modified OurModel Architecture:")
    print(model)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    x = torch.randn(1, 3, 224, 224).to(device)
    out = model(x)
    print(f"Output shape: {out.shape}")  # torch.Size([1, num_classes])

# Use custom model
model = OurModel(num_classes=num_classes).to(device)
print("OurModel Architecture:")
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Layer-specific learning rates
# Higher learning rate for final layer
base_params = [param for name, param in model.named_parameters() if "fc" not in name]
fc_params = model.fc.parameters()

optimizer = optim.SGD([
    {'params': base_params, 'lr': learning_rate},
    {'params': fc_params, 'lr': learning_rate * 10}
], momentum=0.9)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Path to save best model
best_model_path = 'best_model_complex.pth'

# Training function (integrates validation and model saving)
def train_model():
    train_losses, val_losses, accuracies = [], [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
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

        # Update learning rate
        scheduler.step(epoch_acc)

        # Save best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}")

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Accuracy: {epoch_acc:.4f}')

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_metrics_complex.png')
    plt.show()

    return train_losses, val_losses, accuracies


# Test model and calculate metrics
def test_model():
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
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
    plt.savefig('confusion_matrix_complex.png')
    plt.show()

    # Calculate accuracy, precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None,
                                                               labels=range(num_classes))
    accuracy = np.sum(all_labels == all_preds) / len(all_labels)

    print(f"Overall Test Accuracy: {accuracy:.4f}")
    for i, cls in enumerate(class_names):
        print(f"Class {cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1[i]:.4f}")

    # Calculate specificity (per class in multi-class scenario)
    print("Specificity per class:")
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        print(f"Class {class_names[i]} Specificity: {specificity:.4f}")

    # Calculate AUC (one-vs-rest approach for multi-class)
    auc_scores = []
    for i in range(num_classes):
        # Convert true labels to binary format
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')  # Skip AUC calculation if a class has no positive samples
        auc_scores.append(auc)
        print(f"Class {class_names[i]} AUC: {auc:.4f}")

    # Save results
    np.savez('test_results_complex.npz',
             labels=all_labels,
             preds=all_preds,
             probs=all_probs,
             class_names=class_names,
             confusion_matrix=cm,
             accuracy=accuracy,
             precision=precision,
             recall=recall,
             f1=f1,
             auc=auc_scores)

    return accuracy, cm


# Train model
train_metrics = train_model()

# Test model
test_acc, conf_matrix = test_model()


# Feature extraction and t-SNE dimensionality reduction function
def extract_features_and_tsne(model, data_loader, device):

    # Define layers to extract features from - both selection and number of layers can be flexibly adjusted based on model architecture
    layers = {
        'conv1': model.conv1,  # Input convolution layer

        # Convolution layers in Stage1
        'stage1_block1_conv1': model.stage1[0].conv1,
        'stage1_block1_conv2': model.stage1[0].conv2,
        'stage1_block2_conv1': model.stage1[1].conv1,
        'stage1_block2_conv2': model.stage1[1].conv2,
        # Convolution layers in Stage2
        'stage2_block1_conv1': model.stage2[0].conv1,
        'stage2_block1_conv2': model.stage2[0].conv2,
        'stage2_block2_conv1': model.stage2[1].conv1,
        'stage2_block2_conv2': model.stage2[1].conv2,
        # Convolution layers in Stage3
        'stage3_block1_conv1': model.stage3[0].conv1,
        'stage3_block1_conv2': model.stage3[0].conv2,
        'stage3_block2_conv1': model.stage3[1].conv1,
        'stage3_block2_conv2': model.stage3[1].conv2,
        # Convolution layers in Stage4
        'stage4_block1_conv1': model.stage4[0].conv1,
        'stage4_block1_conv2': model.stage4[0].conv2,
        'stage4_block2_conv1': model.stage4[1].conv1,
        'stage4_block2_conv2': model.stage4[1].conv2,
        # Fully connected layer
        'fc': model.fc
    }

    # Create storage for each layer
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

    # Set to evaluation mode
    model.eval()
    total_samples = 0

    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass (triggers hooks)

            # Process current batch features
            batch_size = inputs.size(0)

            for name in layers:
                # Get features and apply global average pooling
                feat = features[name][:batch_size]

                if feat.ndim == 4:  # Convolutional features [B, C, H, W]
                    feat = feat.mean(dim=[2, 3])  # Global average pooling
                elif feat.ndim == 2:  # FC features [B, F]
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

    # Dynamically adjust perplexity based on sample size
    if total_samples < 50:
        perplexity = max(5, total_samples // 3)  # Minimum of 5
        print(f"Adjusted perplexity to {perplexity} due to small sample size ({total_samples})")
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

    # Visualize t-SNE results for the last layer (if available)
    if 'fc' in all_features and all_features['fc'].shape[0] >= 10:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results.iloc[:, -2], tsne_results.iloc[:, -1],
                              c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Class')
        plt.title('t-SNE Visualization of Last Layer Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig('tsne_visualization_complex.png')
        plt.show()
    else:
        print("Insufficient samples for last layer visualization, skipping")

    return all_features, tsne_results

# Create full data loader (all data)
# Use test transform since data augmentation is not needed
full_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Call feature extraction function (using full dataset)
extract_features_and_tsne(model, full_loader, device)