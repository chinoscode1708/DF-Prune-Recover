import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import random
import time
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same list, but now we load them
MODELS_TO_PROCESS = [
    'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2',
    'mobilenet_v2', 'vgg19_bn', 'densenet121', 'shufflenet_v2_x1_0'
]

GEN_IMAGES_COUNT = 1024   # How many images per model?
BATCH_SIZE = 32           # Generation batch size
LR_GEN = 0.05

TEACHER_DIR = "saved_teachers"
DATA_DIR = "synthetic_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# DEEP INVERSION LOGIC
# ==========================================
class DeepInversionFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.mean = None; self.var = None
    def hook_fn(self, module, input, output):
        self.mean = input[0].mean([0, 2, 3])
        self.var = input[0].var([0, 2, 3], unbiased=False)
    def close(self): self.hook.remove()

def synthesize_batch(teacher, batch_size=32, iterations=200):
    teacher.eval()
    hooks = [DeepInversionFeatureHook(m) for m in teacher.modules() if isinstance(m, nn.BatchNorm2d)]
    
    # If model has no BN (shouldn't happen with this list), return noise
    if not hooks: return torch.randn((batch_size, 3, 224, 224))

    inputs = torch.randn((batch_size, 3, 224, 224), requires_grad=True, device=device)
    optimizer = optim.Adam([inputs], lr=LR_GEN)
    
    def jitter(x, lim=32):
        off1 = random.randint(-lim, lim)
        off2 = random.randint(-lim, lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))
    
    for _ in range(iterations):
        optimizer.zero_grad()
        inputs_jit = jitter(inputs)
        outputs = teacher(inputs_jit)
        
        loss_bn = sum([torch.norm(h.mean - m.running_mean, 2) + torch.norm(h.var - m.running_var, 2) for h, m in zip(hooks, [m for m in teacher.modules() if isinstance(m, nn.BatchNorm2d)])])
        loss_ce = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean()
        
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        loss_tv = torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2))

        loss = 1.0*loss_ce + 10.0*loss_bn + 1e-5*loss_tv
        loss.backward()
        optimizer.step()
        inputs.data.clamp_(-3.0, 3.0)
    
    for h in hooks: h.close()
    return inputs.detach().cpu()

# ==========================================
# HELPER: LOAD ARCHITECTURE
# ==========================================
def load_architecture(model_name):
    # Same logic as Script 1 to get correct structure
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

# ==========================================
# MAIN LOOP
# ==========================================
for model_name in MODELS_TO_PROCESS:
    teacher_path = os.path.join(TEACHER_DIR, f"{model_name}_cifar10.pth")
    data_path = os.path.join(DATA_DIR, f"dreams_{model_name}.pt")
    
    if os.path.exists(data_path):
        print(f"Skipping generation for {model_name} (Data exists)")
        continue
    
    if not os.path.exists(teacher_path):
        print(f"Teacher {model_name} not found! Run Script 1 first.")
        continue

    print(f"\nGenerating data for {model_name}...")
    
    # 1. Load Teacher
    teacher = load_architecture(model_name)
    teacher.load_state_dict(torch.load(teacher_path))
    teacher.eval()
    
    # 2. Generate Data
    batches = []
    num_batches = GEN_IMAGES_COUNT // BATCH_SIZE
    
    start_time = time.time()
    for _ in tqdm(range(num_batches), desc=f"Dreaming {model_name}"):
        batch_imgs = synthesize_batch(teacher, batch_size=BATCH_SIZE, iterations=200)
        batches.append(batch_imgs)
        
    synthetic_dataset = torch.cat(batches)
    
    # 3. Save Data
    torch.save(synthetic_dataset, data_path)
    print(f"   Saved {len(synthetic_dataset)} images to {data_path}")
    print(f"   Time taken: {(time.time()-start_time)/60:.1f} mins")

print("\nAll datasets generated!")