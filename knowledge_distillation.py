import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torchvision.models as models
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2',
    'mobilenet_v2', 'vgg19_bn', 'densenet121', 'shufflenet_v2_x1_0'
]
BATCH_SIZE = 32
EPOCHS_RECOVERY = 15
LR_RECOVERY = 0.001

TEACHER_DIR = "saved_teachers"
PRUNED_DIR = "saved_pruned"
DATA_DIR = "synthetic_data"
RECOVERED_DIR = "saved_recovered"
os.makedirs(RECOVERED_DIR, exist_ok=True)

# ==========================================
# UTILITIES
# ==========================================
def load_architecture(model_name):
    # (Same architecture loader as Script 1)
    try: model = models.__dict__[model_name](pretrained=False)
    except: return None
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

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
        self.aug = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4)
        )
    def __getitem__(self, index): return self.aug(self.tensors[index])
    def __len__(self): return self.tensors.size(0)

def set_bn_eval(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# ==========================================
# MAIN RECOVERY LOOP
# ==========================================
print("=== Recovering Models via Distillation ===")

for model_name in MODELS:
    t_path = os.path.join(TEACHER_DIR, f"{model_name}_cifar10.pth")
    p_path = os.path.join(PRUNED_DIR, f"{model_name}_pruned.pth")
    d_path = os.path.join(DATA_DIR, f"dreams_{model_name}.pt")
    save_path = os.path.join(RECOVERED_DIR, f"{model_name}_recovered.pth")
    
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(d_path)):
        print(f"Skipping {model_name} (Missing files)")
        continue

    print(f"Recovering {model_name}...")

    # 1. Load Teacher
    teacher = load_architecture(model_name)
    teacher.load_state_dict(torch.load(t_path))
    teacher.eval()

    # 2. Load Pruned Student
    student = load_architecture(model_name)
    
    # CRITICAL: We must re-register pruning hooks before loading state_dict
    # because the saved state_dict contains 'weight_orig' and 'weight_mask'
    parameters_to_prune = []
    for name, module in student.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.5)
    
    student.load_state_dict(torch.load(p_path))

    # 3. Load Data
    synthetic_data = torch.load(d_path)
    loader = torch.utils.data.DataLoader(
        AugmentedDataset(synthetic_data.to(device)), 
        batch_size=BATCH_SIZE, shuffle=True
    )

    # 4. Recovery Training
    optimizer = optim.SGD(student.parameters(), lr=LR_RECOVERY, momentum=0.9)
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    student.train()
    student.apply(set_bn_eval) # Freeze BN stats

    for epoch in range(EPOCHS_RECOVERY):
        student.apply(set_bn_eval)
        total_loss = 0
        for inputs in loader:
            with torch.no_grad():
                t_out = teacher(inputs)
                t_soft = torch.log_softmax(t_out / 3.0, dim=1)
            
            optimizer.zero_grad()
            s_out = student(inputs)
            s_soft = torch.log_softmax(s_out / 3.0, dim=1)
            
            loss = kl_loss(s_soft, t_soft)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"   Recovery Loss: {total_loss:.4f}")

    # 5. Make Pruning Permanent (Remove Masks) for Final Storage
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    
    torch.save(student.state_dict(), save_path)
    print(f"   Saved recovered model to {save_path}")

print("\nStep 2 Complete: All models recovered and stored.")