# Diabetic Retinopathy Detection using Deep Learning 👁️🧠

This project uses deep learning to detect **Diabetic Retinopathy (DR)** from retinal fundus images.  
An **EfficientNet-B3 Ensemble Model** is trained on a large Kaggle dataset with advanced techniques such as **class-balanced sampling, fine-tuning, data augmentation & focal loss** to handle class imbalance.

---

## 🚀 Live Demo  
🔗 Streamlit App: https://diabetes-retinopathy-by-aman-verma.streamlit.app/

---

## 📌 Features

✔️ Upload retinal image & get AI-based DR stage prediction  
✔️ EfficientNet-B3 Transfer Learning  
✔️ Ensemble of two models for higher stability  
✔️ Class imbalance handled using Class Weights + Focal Loss  
✔️ Real-time prediction UI with Streamlit  
✔️ Model deployed publicly for easy access

---

## 🧠 Model Architecture

| Model | Technique | Purpose |
|-------|----------|---------|
| Model A | EfficientNet-B3 + Class Weights | Strong baseline |
| Model B | EfficientNet-B3 + Focal Loss | Better learning on minority classes |
| Ensemble | Soft voting of both models | Stable + Higher accuracy |

📊 Final Test Accuracy: **66%**  
*(Improved performance on Mild, Moderate & Severe DR cases)*  

---

## 📂 Dataset

Dataset used for training:  
🔗 https://www.kaggle.com/datasets/amanvermacoder/diabetes-retinopathy-dataset

- 5 classes of DR  
- ~145K total images  
- Train/Val/Test split included

---

## 📒 Training Notebook

Full training code with evaluation reports:  
🔗 https://www.kaggle.com/code/amanvermacoder/drp-3-0

---

## 🛠️ Technologies Used

| Category | Libraries |
|---------|-----------|
| Deep Learning | TensorFlow, Keras |
| Deployment | Streamlit |
| Data | OpenCV, NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## ⚙️ Local Setup

```bash
# Clone repo
git clone https://github.com/aman-verma-coder/Diabetes-Retinopathy.git
cd Diabetes-Retinopathy

# Create venv
python -m venv dr-env
dr-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
