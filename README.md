# 🧠 Detection of Alzheimer’s Disease using Machine Learning

This project focuses on detecting Alzheimer's Disease using Deep Learning techniques applied to brain MRI images. The model classifies MRI scans into different stages of Alzheimer's progression.

## 📌 Project Overview

Alzheimer’s Disease is a progressive neurological disorder that affects memory and cognitive function. Early detection can significantly improve treatment planning and patient care.

This project uses Convolutional Neural Networks (CNN) to analyze MRI images and classify them into predefined categories.

---

## 🚀 Features

- MRI image classification using CNN
- Trained deep learning model (.h5)
- Model evaluation and testing scripts
- Prediction script for new images
- Web application with ML backend
- Jupyter notebooks for experimentation
- Sample input images for testing

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Flask (for web app backend)
- Jupyter Notebook

---

## 📂 Project Structure

Detection-of-Alzheimer-s-Disease-using-Machine-Learning/
│
├── dataset/ # Training dataset (not uploaded if large)
├── notebooks/ # Model development notebooks
├── web_app_with_ml_backend/ # Flask web application
├── demo/ # Demo resources
├── sampleinput/ # Sample MRI images for testing
├── test_images/ # Test dataset
│
├── Alzheimer's_classification.py # Main training script
├── cnn.py # CNN architecture
├── predict.py # Prediction script
├── test.py # Model testing
├── alzheimers.h5 # Trained model
├── snapshot_1.hdf5 # Model checkpoint
│
├── train.csv
├── ALZdataset.csv
├── README.md



---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Detection-of-Alzheimer-s-Disease-using-Machine-Learning.git
cd Detection-of-Alzheimer-s-Disease-using-Machine-Learning

Install dependencies:
pip install -r requirements.txt
🏋️‍♂️ Model Training
To train the model:
python Alzheimer's_classification.py

🔎 Make Predictions
To test on a new MRI image:
python predict.py --image path_to_image
🌐 Run Web Application
cd web_app_with_ml_backend
python app.py
