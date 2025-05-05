# 🌍 Earthquake Magnitude Prediction Web App

A Flask-based web application that predicts earthquake magnitude based on historical seismic data, offering real-time insight using machine learning models. Built with a focus on data-driven disaster preparedness 
and awareness.

---

## 📌 Problem Statement

Earthquakes are among the most destructive natural disasters, often striking without warning. Accurate and timely prediction of earthquake magnitude can greatly help in minimizing loss of life and property. Traditional monitoring systems primarily focus on detection, but lack predictive capabilities. This project addresses that gap by using historical earthquake data to model and forecast potential magnitudes based on various geophysical features.

---

## 📂 Dataset Description

**Source**: [Kaggle - Significant Earthquakes Dataset](https://www.kaggle.com/datasets/usamabuttar/significant-earthquakes/data)  
**Authority**: United States Geological Survey (USGS)

This dataset contains records of significant global earthquakes from 1900 to the present, specifically those with magnitudes 5.0 or greater. It is updated weekly and covers data from all around the globe, including seismically active zones like the Pacific Ring of Fire.

### 📊 Key Features:

- `time`: Timestamp of the earthquake (Unix format)
- `latitude`, `longitude`: Geographic coordinates of the epicenter
- `depth`: Depth of the quake (km)
- `mag`: Magnitude of the earthquake
- `magType`: Type of magnitude scale (e.g., mw, mb)
- `nst`, `gap`, `dmin`, `rms`: Seismic measurement values
- `place`: Human-readable location
- `type`: Nature of event (e.g., earthquake, explosion)
- `status`: Review status from USGS
- `horizontalError`, `depthError`, `magError`: Error margins

---

## 🎯 Project Objectives

- Build a machine learning model to predict earthquake magnitudes.
- Develop a clean, user-friendly web interface using Flask.
- Host the application publicly for demonstration and educational use.

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Python, Flask
- **Machine Learning**: Pandas, NumPy, Scikit-learn
- **Deployment**: PythonAnywhere

---

## 🔍 Features

- Predict earthquake magnitude based on input parameters.
- Clean, responsive interface.
- Live deployment with public access.

---

## 🔗 Live Demo

👉 [Access the Live Web App](https://joynajoy.pythonanywhere.com/)  
📊 [Kaggle Dataset](https://www.kaggle.com/datasets/usamabuttar/significant-earthquakes/data)

---
## 🖼️ Screenshots

### 🔹 Home Page
![Home Page](screenshots/homepage.png)  
![Home Page - Live](https://github.com/user-attachments/assets/1e8d92bf-8820-484b-9cd9-5035a7a44097)

### 🔹 Prediction Output
![Prediction Page](screenshots/prediction.png)


