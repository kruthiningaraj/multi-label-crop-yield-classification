# 🌾 Multi-label Crop & Yield Classification  

This repository implements **multi-label classification approaches** to predict **optimal agricultural crops and their yield**

📄 **[Read My Published Paper Here](research_paper/Research_Paper.pdf)**

---

## 🧠 Key Features:
- Multi-label classification for **crop recommendation & yield prediction**.
- Implements **Problem Transformation methods**: Binary Relevance (BR), Classifier Chains (CC), Label Powerset (LP).
- Includes **Adapted Algorithms**: MLkNN, BRkNN-a.
- Ensemble methods **RAkELd** and **RAkELo** for improved performance.

---

## 📂 Dataset:
We use agricultural datasets containing:
- **Soil Nutrients (NPK)**, **pH**
- **Rainfall**, **Temperature**, **Humidity**
- **Soil Type** & **Crop-Yield labels**

Example: [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## 🚀 Getting Started:
1️⃣ Clone repository:
```bash
git clone https://github.com/your-username/multi-label-crop-yield-classification.git
cd multi-label-crop-yield-classification
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Train and evaluate models:
```bash
python src/train.py
```

---

## 📊 Highlights:
- Implements all approaches detailed in the **published research paper**.
- Compares models using **Accuracy, Hamming Loss, and F1-score**.
- Outputs comparative metrics for easy analysis.

---

## 📜 License:
MIT License © 2025
