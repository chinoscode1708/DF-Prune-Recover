import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import csv

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2',
    'mobilenet_v2', 'vgg19_bn', 'densenet121', 'shufflenet_v2_x1_0'
]

# Paths
TEACHER_DIR = "saved_teachers"
PRUNED_DIR = "saved_pruned"
RECOVERED_DIR = "saved_recovered"
CSV_FILE = "final_results.csv"

# ==========================================
# TEST DATA SETUP
# ==========================================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ==========================================
# UTILITIES
# ==========================================
def load_architecture(model_name):
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

def evaluate(model):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ==========================================
# MAIN EXPERIMENT LOOP
# ==========================================
print(f"=== Running Final Experiments ===")
print(f"Saving results to {CSV_FILE}")

with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Teacher Acc', 'Pruned Acc', 'Recovered Acc', 'Improvement'])

    for model_name in MODELS:
        print(f"Evaluating {model_name}...")
        
        # Paths
        t_path = os.path.join(TEACHER_DIR, f"{model_name}_cifar10.pth")
        p_path = os.path.join(PRUNED_DIR, f"{model_name}_pruned.pth")
        r_path = os.path.join(RECOVERED_DIR, f"{model_name}_recovered.pth")
        
        if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(r_path)):
            print(f"   Missing files for {model_name}, skipping.")
            continue

        # 1. Evaluate Teacher
        teacher = load_architecture(model_name)
        teacher.load_state_dict(torch.load(t_path))
        acc_teacher = evaluate(teacher)

        # 2. Evaluate Pruned Model
        # Need to reconstruct pruning hooks to load 'weight_mask' correctly
        pruned = load_architecture(model_name)
        parameters_to_prune = []
        for name, module in pruned.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.5)
        
        pruned.load_state_dict(torch.load(p_path))
        acc_pruned = evaluate(pruned)

        # 3. Evaluate Recovered Model
        # This model has pruning made permanent, so it loads like a normal model
        recovered = load_architecture(model_name)
        recovered.load_state_dict(torch.load(r_path))
        acc_recovered = evaluate(recovered)

        # 4. Record
        imp = acc_recovered - acc_pruned
        print(f"   {model_name}: T={acc_teacher:.1f}% | P={acc_pruned:.1f}% | R={acc_recovered:.1f}% | Imp={imp:+.1f}%")
        writer.writerow([model_name, f"{acc_teacher:.2f}", f"{acc_pruned:.2f}", f"{acc_recovered:.2f}", f"{imp:.2f}"])

print("\nStep 3 Complete: Experiments finished.")