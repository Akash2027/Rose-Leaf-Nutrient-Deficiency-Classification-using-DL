# 🌹 Rose Leaf Nutrient Deficiency Classification using Deep Learning

## Overview

This project presents an AI-powered system for identifying nutrient deficiencies in rose plants using leaf images. The system employs a Deep Learning-based image classification model to detect common nutrient deficiencies and provides visual explanations through Grad-CAM heatmaps.

The application is deployed using Streamlit, enabling users to upload leaf images and instantly receive deficiency predictions along with agronomic recommendations.

---

## 🌐 Live Demo

**Streamlit Application:**
https://rose-leaf-nutrient-deficiency-classification-system.streamlit.app/

---

## Features

### Nutrient Deficiency Detection

The system classifies rose leaf images into the following categories:

* Healthy
* Iron Deficiency
* Magnesium Deficiency
* Phosphorus Deficiency

### Deep Learning-Based Prediction

* DenseNet121 architecture
* Transfer Learning approach
* PyTorch implementation

### Explainable AI

* Grad-CAM visualization
* Highlights regions responsible for model predictions
* Improves interpretability and trustworthiness

### Interactive Web Application

* Built with Streamlit
* Image upload functionality
* Real-time prediction generation
* Visual heatmap display

### Expert Recommendation Support

* AI-generated treatment suggestions
* Deficiency-specific nutrient management recommendations

---

## Project Architecture

```text
Input Leaf Image
        │
        ▼
Image Preprocessing
        │
        ▼
DenseNet121 Model
        │
        ▼
Deficiency Classification
        │
        ├─────────────► Predicted Class
        │
        ▼
Grad-CAM Generation
        │
        ▼
Heatmap Visualization
        │
        ▼
Agronomic Recommendation
```

---

## Dataset Classes

| Class      | Description                     |
| ---------- | ------------------------------- |
| Healthy    | No nutrient deficiency detected |
| Iron       | Iron nutrient deficiency        |
| Magnesium  | Magnesium nutrient deficiency   |
| Phosphorus | Phosphorus nutrient deficiency  |

---

## Technology Stack

### Programming Language

* Python

### Deep Learning Framework

* PyTorch
* Torchvision

### Web Framework

* Streamlit

### Data Processing

* NumPy
* Pandas

### Image Processing

* OpenCV
* Pillow

### Explainable AI

* Grad-CAM

### Generative AI

* Google Gemini API

---

## Project Structure

```text
project/
│
├── app.py
├── predict_nutrient.py
├── requirements.txt
├── symptom_model.pth
├── symptom_nutrient_weights.csv
├── Algorithm.docx
├── rose-patent.ipynb
├── README.md
│
├── test_images/
│   ├── healthy.jpg
│   ├── iron.jpg
│   ├── magnesium.jpg
│   └── phoshorus.jpg
│
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Akash2027/Rose-Leaf-Nutrient-Deficiency-Classification-using-DL.git
cd Rose-Leaf-Nutrient-Deficiency-Classification-using-DL
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Gemini API

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

## Run Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Model Details

### Architecture

DenseNet121

### Input Size

```text
224 × 224 × 3
```

### Framework

PyTorch

### Output Classes

```text
4 Classes
```

* Healthy
* Iron
* Magnesium
* Phosphorus

---

## Explainable AI with Grad-CAM

The system generates Grad-CAM heatmaps that:

* Highlight affected leaf regions
* Visualize model attention
* Improve transparency
* Assist researchers in understanding predictions

---

## Sample Workflow

1. Upload a rose leaf image.
2. Image is preprocessed.
3. DenseNet121 predicts nutrient status.
4. Grad-CAM generates attention map.
5. Deficiency category is displayed.
6. Agronomic recommendations are generated.
7. User receives actionable insights.

---

## Research Applications

* Precision Agriculture
* Smart Farming
* Plant Health Monitoring
* Agricultural Decision Support Systems
* Explainable Artificial Intelligence in Agriculture

---

## Future Enhancements

* Multi-deficiency detection
* Mobile application deployment
* Larger agricultural datasets
* Real-time camera integration
* Disease and pest detection support
* Cloud-based inference services

---

## Results

The developed system successfully:

* Detects nutrient deficiencies from rose leaf images
* Provides explainable predictions
* Generates visual heatmaps
* Assists users with nutrient management recommendations

---

## Author

**Akash K**

Software Engineering Student

GitHub:
https://github.com/Akash2027

LinkedIn:
https://www.linkedin.com/in/akash-k-bb9a20274

---

## License

This project is intended for academic, research, and educational purposes.

© 2026 Akash K. All Rights Reserved.
