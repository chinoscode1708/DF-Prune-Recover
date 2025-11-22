import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All models requested
MODELS_TO_PREPARE = [
    # ResNet Variants
    'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2',
    # Different Architectures
    'mobilenet_v2', 'vgg19_bn', 'densenet121', 'shufflenet_v2_x1_0'
]

SAVE_DIR = "saved_teachers"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# DATA LOADING (CIFAR-10)
# ==========================================
print("Loading CIFAR-10 data for teacher fine-tuning...")
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# ==========================================
# MODEL FACTORY
# ==========================================
def get_model_and_replace_head(model_name, num_classes=10):
    try:
        model = models.__dict__[model_name](pretrained=True)
    except KeyError:
        raise ValueError(f"Model {model_name} not found.")

    # Auto-replace head
    if hasattr(model, 'fc'): # ResNet, ShuffleNet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential): # VGG, MobileNet
            # VGG/MobileNet structure varies, finding the last Linear layer
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                    break
        elif isinstance(model.classifier, nn.Linear): # DenseNet
             model.classifier = nn.Linear(model.classifier.in_features, num_classes)
             
    return model.to(device)

# ==========================================
# MAIN LOOP
# ==========================================
for model_name in MODELS_TO_PREPARE:
    save_path = os.path.join(SAVE_DIR, f"{model_name}_cifar10.pth")
    
    if os.path.exists(save_path):
        print(f"Skipping {model_name} (Already exists)")
        continue

    print(f"\nProcessing {model_name}...")
    
    # 1. Load & Adapt
    teacher = get_model_and_replace_head(model_name)
    
    # 2. Fine-tune (3 Epochs just to learn CIFAR classes)
    optimizer = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    teacher.train()
    
    print(f"   Fine-tuning {model_name} on CIFAR-10...")
    for epoch in range(3):
        running_loss = 0.0
        # Limit to 200 batches per epoch to speed up setup (sufficient for teacher initialization)
        # Remove '[:200]' if you want full accuracy
        for i, (inputs, labels) in enumerate(tqdm(trainloader, leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(teacher(inputs), labels)
            loss.backward()
            optimizer.step()
            
    # 3. Save
    torch.save(teacher.state_dict(), save_path)
    print(f"   Saved to {save_path}")

print("\nAll teachers prepared!")