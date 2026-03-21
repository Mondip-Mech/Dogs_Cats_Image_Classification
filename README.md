# 🐶🐱 Dogs vs Cats Image Classification (Deep Learning)

An end-to-end Deep Learning project that classifies images of dogs and cats using a Convolutional Neural Network (CNN) with **Transfer Learning (MobileNetV2)** and deploys the model using **Streamlit**.

---

## 🚀 Project Overview

This project uses a pretrained deep learning model to classify images into two categories: **Dog 🐶** and **Cat 🐱**. The model is trained using transfer learning for better accuracy and faster convergence.

An interactive web application is built using Streamlit, allowing users to upload images and get real-time predictions.

---

## 🧠 Key Features

* Deep Learning model using **MobileNetV2**
* Transfer Learning with fine-tuning
* Image preprocessing and augmentation
* Real-time predictions via Streamlit UI
* Clean and interactive web interface

---

## 🧰 Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pillow
* Streamlit

---

## 📂 Project Structure

```
Dogs_Cats_Image_Classification/
│
├── app.py
├── requirements.txt
├── README.md
└── model/ (optional - not uploaded due to size)
```

---


## ▶️ How to Run Locally

```bash
conda create -n streamlit_env python=3.10
conda activate streamlit_env

pip install -r requirements.txt

python -m streamlit run app.py
```

---

## 📊 Model Performance

* Training Accuracy: ~95%
* Validation Accuracy: ~97% (after tuning)

---





