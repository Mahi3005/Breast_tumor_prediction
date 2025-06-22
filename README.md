

---

```markdown
# Breast Tumor Prediction 🎗️

This project is a web-based machine learning application built using **Flask** to predict whether a breast tumor is **malignant** or **benign** based on medical input features.

## 📁 Project Structure

```

├── app.py                   # Main Flask application
├── breast\_cancer\_model.pkl # Trained ML model
├── scaler.pkl              # Feature scaler
├── requirements.txt        # Python dependencies

````

## 🚀 How to Run

1. **Clone the repository**  
```bash
git clone https://github.com/Mahi3005/Breast_tumor_prediction.git
cd Breast_tumor_prediction
````

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On Mac/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to access the app.

## ⚙️ How It Works

* User inputs medical data (e.g., radius, texture, perimeter).
* The app uses a saved scaler (`scaler.pkl`) to normalize inputs.
* The trained ML model (`breast_cancer_model.pkl`) predicts the tumor type.
* The result is displayed as **Malignant** or **Benign**.

## 🧪 Requirements

* Python 3.x
* Flask
* scikit-learn
* numpy
* pandas

Install using:

```bash
pip install -r requirements.txt
```

## ✅ Features

* Simple UI to input tumor characteristics
* Preprocessing with saved scaler
* Accurate ML predictions using a trained model
* Educational and practical use case for medical diagnosis

## 📌 To Improve

* Add model training code
* Input validation and error handling
* Deploy on Heroku or Render
* Add prediction confidence scores

## 📄 License

This project is open-source and free to use for learning purposes.

```

---

Let me know if you want a badge version, training script, or deployment guide added!
```
