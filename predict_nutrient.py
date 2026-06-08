import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# ------------------- DEVICE -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- PARAMETERS -------------------
MODEL_PATH = "symptom_model.pth"
CSV_PATH = "symptom_nutrient_weights.csv"
EPS_HEALTHY = 0.4  # Increased threshold slightly to avoid false healthy positives

# ------------------- IMAGE TRANSFORM -------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ------------------- LOAD MODEL -------------------
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 4)
    # Using map_location to ensure it loads on CPU if CUDA is missing
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load resources once
model = load_model()

def load_csv():
    df = pd.read_csv(CSV_PATH)
    classes = df['class_name'].tolist()
    class_vectors = df.iloc[:,1:].values
    return classes, class_vectors

classes, class_vectors = load_csv()

# ------------------- UTILS -------------------
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# ------------------- GRAD-CAM -------------------
def generate_gradcam(model, img_tensor, target_class):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook the last dense block of DenseNet121
    target_layer = model.features.denseblock4.denselayer16
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    model.zero_grad()
    output = model(img_tensor)
    
    # Handle output shape
    score = output[0, target_class]
    score.backward()

    # Generate CAM
    fmap = features[0][0].cpu().detach().numpy()
    grads = gradients[0][0].cpu().detach().numpy()
    weights = np.mean(grads, axis=(1,2))
    
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_forward.remove()
    handle_backward.remove()
    return cam

# ------------------- PREDICTION FUNCTION -------------------
def predict_nutrient(img):
    img_tensor = transform(img)
    
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        symptom_activation = F.softmax(output, dim=1).cpu().numpy()[0]

    # Calculate scores
    scores = [cosine_similarity(symptom_activation, vec) for vec in class_vectors]
    scores = np.array(scores)
    pred_idx = scores.argmax()
    pred_class = classes[pred_idx]

    # Healthy check logic
    if np.max(scores) < EPS_HEALTHY:
        pred_class = "Healthy"

    # Generate Grad-CAM
    # Note: Grad-CAM requires gradients, so we might need to re-run forward pass with grad enabled if needed,
    # but usually hooks work if we just run backward. 
    # To be safe for the visualizer, we run a separate pass with grad enabled.
    with torch.set_grad_enabled(True):
        cam = generate_gradcam(model, img_tensor, pred_idx)

    # Image Processing for Visualization
    cam_img = np.array(img.resize((224,224)))
    
    # 1. Create Heatmap (It returns BGR)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    
    # 2. Convert Heatmap to RGB to match PIL Image
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 3. Blend
    superimposed_img = cv2.addWeighted(cam_img, 0.6, heatmap, 0.4, 0)

    return pred_class, symptom_activation, scores, superimposed_img