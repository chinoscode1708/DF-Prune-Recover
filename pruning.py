import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import copy
import os

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2',
    'mobilenet_v2', 'vgg19_bn', 'densenet121', 'shufflenet_v2_x1_0'
]
PRUNING_AMOUNT = 0.5  # 50% Sparsity

TEACHER_DIR = "saved_teachers"
PRUNED_DIR = "saved_pruned"
os.makedirs(PRUNED_DIR, exist_ok=True)

# ==========================================
# HELPER: LOAD ARCHITECTURE
# ==========================================
def load_architecture(model_name):
    try: model = models.__dict__[model_name](pretrained=False)
    except: return None
    
    # Adapt head to CIFAR-10 (10 classes)
    if hasattr(model, 'fc'): model.fc = nn.Linear(model.fc.in_features, 10)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, 10)
                    break
        elif isinstance(model.classifier, nn.Linear):
             model.classifier = nn.Linear(model.classifier.in_features, 10)
    return model.to(device)

# ==========================================
# MAIN PRUNING LOOP
# ==========================================
print(f"=== Pruning Models ({PRUNING_AMOUNT*100}%) ===")

for model_name in MODELS:
    teacher_path = os.path.join(TEACHER_DIR, f"{model_name}_cifar10.pth")
    save_path = os.path.join(PRUNED_DIR, f"{model_name}_pruned.pth")
    
    if not os.path.exists(teacher_path):
        print(f"Skipping {model_name} (Teacher not found)")
        continue
        
    print(f"Pruning {model_name}...")
    
    # 1. Load Model
    model = load_architecture(model_name)
    model.load_state_dict(torch.load(teacher_path))
    
    # 2. Apply Pruning
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # This adds 'weight_mask' and 'weight_orig' to the model state
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=PRUNING_AMOUNT,
    )
    
    # 3. Save the model WITH the masks (needed for recovery training)
    torch.save(model.state_dict(), save_path)
    print(f"   Saved pruned model to {save_path}")

print("\nStep 1 Complete: All models pruned and stored.")