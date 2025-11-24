# Zero-Shot Pruning Recovery via DeepInversion

This project implements a pipeline to recover the accuracy of pruned neural networks without accessing the original training data. It uses DeepInversion to generate synthetic data from "Teacher" models and uses Knowledge Distillation to recover "Student" (pruned) models.

## Execution Instructions

Run the scripts in the following numerical order to complete the pipeline.

### 1. Prepare Teacher Models
Fine-tunes pre-trained models (ResNet, MobileNet, VGG, etc.) on CIFAR-10 to serve as baselines.
```bash
python prepare_teacher.py
````

### 2\. Prune Models

Applies 50% global unstructured pruning to the teacher models and saves the sparse networks.

```bash
python pruning.py
```

### 3\. Generate Synthetic Data (DeepInversion)

Generates synthetic "dream" images by inverting the teacher models using BatchNorm statistics.

```bash
python deepinversion.py
```

### 4\. Knowledge Distillation Recovery

Recovers the accuracy of the pruned models by distilling knowledge from the teacher using the generated synthetic data.

```bash
python knowledge_distillation.py
```

### 5\. Final Evaluation

Evaluates Teacher, Pruned, and Recovered models on the CIFAR-10 test set and calculates accuracy recovery.

```bash
python eval.py
```

-----

## Expected Output Structure

After running all scripts, your directory will look like this:

```text
.
├── saved_teachers/                # Created by prepare_teacher.py
│   ├── resnet18_cifar10.pth
│   ├── mobilenet_v2_cifar10.pth
│   └── ... (8 models total)
│
├── saved_pruned/                  # Created by pruning.py
│   ├── resnet18_pruned.pth
│   └── ...
│
├── synthetic_data/                # Created by deepinversion.py
│   ├── dreams_resnet18.pt
│   └── ...
│
├── saved_recovered/               # Created by knowledge_distillation.py
│   ├── resnet18_recovered.pth
│   └── ...
│
└── final_results.csv              # Created by eval.py (Contains accuracy stats)
```
