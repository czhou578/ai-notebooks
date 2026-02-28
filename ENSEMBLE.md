# CNN Ensemble Image Classifier

A deep learning ensemble pipeline that combines **17 fully-implemented image classification models** — all trained on CIFAR-100 — into a single unified classifier. Each model independently predicts over 100 classes and the pipeline aggregates their outputs using configurable voting strategies to produce a final, more robust classification.

---

## Models in the Pipeline

| Model                    | Architecture Family     | Notable Trait                       |
| ------------------------ | ----------------------- | ----------------------------------- |
| AlexNet                  | Plain CNN               | Foundational deep CNN               |
| VGG                      | Plain CNN               | Very deep, uniform 3×3 convs        |
| GoogleNet (Inception v1) | Inception               | Parallel multi-scale filters        |
| InceptionV3              | Inception               | Factorized convolutions, aux losses |
| ResNet-18                | Residual                | Shortcut connections                |
| ResNet-50                | Residual                | Bottleneck blocks                   |
| ResNeXt                  | Residual + Grouped Conv | Grouped convolutions                |
| Wide ResNet              | Residual                | Wider, shallower residual blocks    |
| DenseNet                 | Dense                   | Cross-layer feature reuse           |
| SqueezeNet               | Lightweight             | Fire modules, tiny footprint        |
| MobileNetV3              | Lightweight             | Depthwise separable convs, h-swish  |
| EfficientNet             | Compound Scaled         | MBConv + Squeeze-and-Excite         |
| NFNet                    | Normalizer-Free         | Scaled WS, no BatchNorm             |
| ConvNeXt                 | Modernized CNN          | Transformer-inspired CNN            |
| ConvNeXtV2               | Modernized CNN          | GRN, FCMAE pre-training             |
| Vision Transformer (ViT) | Transformer             | Pure attention, patch embeddings    |
| CoAtNet                  | Hybrid                  | Conv + Self-Attention fusion        |

---

## How the Pipeline Works

```
Input Image (32×32 RGB)
       │
       ▼
  Preprocessing
  (Normalize, Resize)
       │
       ├──► AlexNet       ──► logits[100]
       ├──► VGG           ──► logits[100]
       ├──► ResNet-18     ──► logits[100]
       ├──► ResNet-50     ──► logits[100]
       ├──► ResNeXt       ──► logits[100]
       ├──► Wide ResNet   ──► logits[100]
       ├──► DenseNet      ──► logits[100]
       ├──► GoogleNet     ──► logits[100]
       ├──► InceptionV3   ──► logits[100]
       ├──► SqueezeNet    ──► logits[100]
       ├──► MobileNetV3   ──► logits[100]
       ├──► EfficientNet  ──► logits[100]
       ├──► NFNet         ──► logits[100]
       ├──► ConvNeXt      ──► logits[100]
       ├──► ConvNeXtV2    ──► logits[100]
       ├──► ViT           ──► logits[100]
       └──► CoAtNet       ──► logits[100]
                │
                ▼
     Softmax → prob[100] per model
                │
                ▼
       Aggregation Strategy
       (soft vote / hard vote / weighted)
                │
                ▼
     Top-K Predictions + Confidence Scores
```

---

## Step 1 — Export Trained Weights from Each Notebook

At the end of each training loop in your notebooks, save the model state dict:

```python
# Add this at the end of any training cell
torch.save(model.state_dict(), "weights/resnet50.pt")
```

Organize all saved weights in a `weights/` directory:

```
weights/
  alexnet.pt
  vgg.pt
  googlenet.pt
  inceptionv3.pt
  resnet18.pt
  resnet50.pt
  resnext.pt
  wideresnet.pt
  densenet.pt
  squeezenet.pt
  mobilenetv3.pt
  efficientnet.pt
  nfnet.pt
  convnext.pt
  convnextv2.pt
  vit.pt
  coatnet.pt
```

---

## Step 2 — Project Structure

```
ensemble/
  models/
    alexnet.py        # class AlexNet(nn.Module)
    vgg.py
    resnet.py         # ResNet18, ResNet50
    resnext.py
    wideresnet.py
    densenet.py
    googlenet.py
    inceptionv3.py
    squeezenet.py
    mobilenetv3.py
    efficientnet.py
    nfnet.py
    convnext.py
    convnextv2.py
    vit.py
    coatnet.py
  weights/            # .pt files from Step 1
  pipeline.py         # EnsemblePipeline class
  classify.py         # CLI entry point
  cifar100_labels.py  # 100 class name strings
```

Extract the model class definitions from your notebooks into individual `.py` files under `ensemble/models/`. Each file should contain only the model class and its sub-modules (no training code, datasets, or transforms).

---

## Step 3 — The Ensemble Pipeline

```python
# ensemble/pipeline.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet18, ResNet50
# ... import all others

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],   # CIFAR-100 stats
        std=[0.2675, 0.2565, 0.2761]
    ),
])

MODEL_REGISTRY = {
    "alexnet":     (AlexNet,    "weights/alexnet.pt"),
    "vgg":         (VGG,        "weights/vgg.pt"),
    "resnet18":    (ResNet18,   "weights/resnet18.pt"),
    "resnet50":    (ResNet50,   "weights/resnet50.pt"),
    "resnext":     (ResNeXt,    "weights/resnext.pt"),
    "wideresnet":  (WideResNet, "weights/wideresnet.pt"),
    "densenet":    (DenseNet,   "weights/densenet.pt"),
    "googlenet":   (GoogleNet,  "weights/googlenet.pt"),
    "inceptionv3": (InceptionV3,"weights/inceptionv3.pt"),
    "squeezenet":  (SqueezeNet, "weights/squeezenet.pt"),
    "mobilenetv3": (MobileNetV3,"weights/mobilenetv3.pt"),
    "efficientnet":(EfficientNet,"weights/efficientnet.pt"),
    "nfnet":       (NFNet,      "weights/nfnet.pt"),
    "convnext":    (ConvNeXt,   "weights/convnext.pt"),
    "convnextv2":  (ConvNeXtV2, "weights/convnextv2.pt"),
    "vit":         (ViT,        "weights/vit.pt"),
    "coatnet":     (CoAtNet,    "weights/coatnet.pt"),
}

class EnsemblePipeline:
    def __init__(self, strategy="soft", weights=None, models=None):
        """
        strategy: "soft"     → average softmax probabilities (recommended)
                  "hard"     → majority vote over argmax predictions
                  "weighted" → weighted average (supply `weights` dict)
        weights:  dict mapping model name → scalar weight (for "weighted" strategy)
        models:   list of model names to include (defaults to all)
        """
        self.strategy = strategy
        self.model_weights = weights or {}
        self.models = {}
        target_names = models or list(MODEL_REGISTRY.keys())

        for name in target_names:
            cls, path = MODEL_REGISTRY[name]
            model = cls().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            self.models[name] = model

        print(f"Loaded {len(self.models)} models on {DEVICE}.")

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 5):
        """
        Run the ensemble on a single image.
        Returns a list of (class_name, confidence%) tuples.
        """
        from cifar100_labels import CIFAR100_CLASSES

        img = Image.open(image_path).convert("RGB")
        x = PREPROCESS(img).unsqueeze(0).to(DEVICE)  # [1, 3, 32, 32]

        all_probs = []
        individual = {}

        for name, model in self.models.items():
            logits = model(x)                         # [1, 100]
            probs  = F.softmax(logits, dim=-1)        # [1, 100]
            individual[name] = CIFAR100_CLASSES[probs.argmax().item()]
            all_probs.append(probs)

        stacked = torch.stack(all_probs, dim=0)       # [N, 1, 100]

        if self.strategy == "soft":
            final_probs = stacked.mean(dim=0).squeeze()

        elif self.strategy == "hard":
            votes = stacked.argmax(dim=-1).squeeze()  # [N]
            counts = torch.bincount(votes, minlength=100).float()
            final_probs = counts / counts.sum()

        elif self.strategy == "weighted":
            names = list(self.models.keys())
            w = torch.tensor(
                [self.model_weights.get(n, 1.0) for n in names],
                device=DEVICE
            ).view(-1, 1, 1)
            final_probs = (stacked * w).sum(dim=0).squeeze()
            final_probs = final_probs / final_probs.sum()

        top = final_probs.topk(top_k)
        results = [
            (CIFAR100_CLASSES[idx.item()], round(conf.item() * 100, 2))
            for conf, idx in zip(top.values, top.indices)
        ]

        return {
            "top_predictions": results,
            "per_model":       individual,
            "strategy":        self.strategy,
        }
```

---

## Step 4 — CLI Usage

```python
# ensemble/classify.py
import argparse, json
from pipeline import EnsemblePipeline

parser = argparse.ArgumentParser()
parser.add_argument("image",    help="Path to input image")
parser.add_argument("--strategy", default="soft", choices=["soft","hard","weighted"])
parser.add_argument("--topk",   type=int, default=5)
args = parser.parse_args()

pipeline = EnsemblePipeline(strategy=args.strategy)
result   = pipeline.predict(args.image, top_k=args.topk)

print(f"\nStrategy: {result['strategy']}")
print(f"\nTop-{args.topk} Predictions:")
for i, (cls, conf) in enumerate(result["top_predictions"], 1):
    print(f"  {i}. {cls:<30} {conf:.2f}%")

print(f"\nPer-Model Predictions:")
for model, pred in result["per_model"].items():
    print(f"  {model:<15} → {pred}")
```

**Run it:**

```bash
python classify.py path/to/image.jpg --strategy soft --topk 5
```

**Example output:**

```
Strategy: soft

Top-5 Predictions:
  1. leopard                        34.71%
  2. cheetah                        22.18%
  3. tiger                          14.05%
  4. lion                            9.33%
  5. fox                             5.22%

Per-Model Predictions:
  alexnet         → leopard
  vgg             → leopard
  resnet18        → cheetah
  resnet50        → leopard
  efficientnet    → leopard
  vit             → cheetah
  coatnet         → leopard
  ...
```

---

## Aggregation Strategies

### Soft Voting (Recommended)

Average the softmax probability distributions from all models before taking the argmax. This preserves confidence information — a model that is 99% sure outweighs one that is 51% sure.

$$P_{\text{final}}(c) = \frac{1}{N} \sum_{i=1}^{N} P_i(c)$$

### Hard Voting

Each model casts a single vote for its top-1 prediction. The class with the most votes wins. Simple but discards confidence information.

### Weighted Averaging

Scale each model's probability distribution by a manually assigned or validation-accuracy-derived weight before averaging. Useful for boosting your strongest models.

$$P_{\text{final}}(c) = \frac{\sum_{i=1}^{N} w_i \cdot P_i(c)}{\sum_{i=1}^{N} w_i}$$

To derive weights automatically from validation accuracy:

```python
# Example: use each model's CIFAR-100 val accuracy as its weight
weights = {
    "resnet50":     0.72,
    "efficientnet": 0.74,
    "coatnet":      0.76,
    "vit":          0.71,
    "alexnet":      0.55,
    # ...
}
pipeline = EnsemblePipeline(strategy="weighted", weights=weights)
```

---

## Extensions and Ideas

| Idea                             | Description                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Stacking**                     | Train a lightweight meta-learner (e.g. logistic regression or a small MLP) on the stacked model outputs instead of hand-averaging |
| **Temperature Scaling**          | Post-hoc calibration — divide logits by temperature $T$ before softmax to better-calibrated confidence scores                     |
| **Model Selection UI**           | Wrap in a Gradio or Streamlit app — let users toggle which models are included in the ensemble                                    |
| **Batched Inference**            | Run all models in parallel using `torch.multiprocessing` or GPU streams for faster throughput                                     |
| **ONNX Export**                  | Export each model to ONNX for deployment outside PyTorch                                                                          |
| **TTA (Test-Time Augmentation)** | For each model, run inference on multiple augmented versions of the input and average those before ensembling                     |
| **Uncertainty Estimation**       | Use the variance across model predictions as a calibrated uncertainty signal — high disagreement = low confidence                 |
| **Class Activation Maps**        | Use GradCAM to visualize which image regions each model attends to for its prediction                                             |

---

## Dependencies

```bash
pip install torch torchvision pillow
```

Optional (for UI / visualization):

```bash
pip install gradio grad-cam matplotlib
```

---

## Dataset — CIFAR-100

All models in this pipeline were trained and evaluated on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), which contains:

- **60,000** 32×32 RGB images
- **100 classes** grouped into 20 superclasses
- 500 training images and 100 test images per class

The same normalization used during training must be applied at inference time:

```python
mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]
```
