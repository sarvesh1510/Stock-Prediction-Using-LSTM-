#  Stock Market Prediction Web App using LSTM

This project is a **full-stack stock prediction web application** built using **Flask** and **Deep Learning (LSTM)**. It predicts the **next day's stock price** based on the last 60 days of historical closing prices using Yahoo Finance data.

---

##  About the Project

- Uses a **Long Short-Term Memory (LSTM)** neural network model for time series forecasting.
- Retrieves live and historical data using **yFinance API**.
- Fetches real-time **stock market news** via the **NewsAPI**.
- Displays actual vs predicted stock price on dynamic graphs.
- Includes a **user login system**, profile settings, and recent search history.

---

##  Features

- Predict next-day stock prices using LSTM
- Deep Learning model trained on 60-day sequences
- Live charts of actual vs predicted prices
- Latest stock market news
- Secure user authentication (register/login/logout)
- User settings for stock alerts and thresholds
- Recent activity tracking
- Persistent model saving & reloading

---

## Tech Stack

### Backend:
- Python
- Flask
- Flask-SQLAlchemy
- TensorFlow / Keras
- yFinance
- NewsAPI (via `requests`)
- SQLite (default DB)

### Frontend:
- HTML, CSS, JS (Jinja2 Templates)
- Bootstrap (or your styling of choice)
- Matplotlib (for graph generation)

---

## Libraries Used

```bash
flask
flask_sqlalchemy
werkzeug
tensorflow
pandas
numpy
matplotlib
sklearn
yfinance
requests
```

---

## LSTM Model Architecture

- Input: 60 previous days of closing prices
- Layers:
  - LSTM (64 units, return_sequences=True)
  - LSTM (64 units)
  - Dense (25 neurons, ReLU)
  - Dense (1 neuron for final price)
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Trained for 28 epochs with batch size of 32
- Model is saved as `.h5` and reused to avoid retraining

---

## Setup Instructions

1. **Create and Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

4. **Set Up Configurations**

Create a `config.py` file with at least:

```python
class Config:
    SECRET_KEY = 'your_secret_key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'
```

5. **Run the App**

```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
.
├── app.py
├── config.py
├── models/
├── static/
│   └── images/
├── templates/
│   ├── dashboard.html
│   ├── predict.html
│   └── ...
├── requirements.txt
└── README.md
```

---

## 📡 APIs Used

- [Yahoo Finance API (via yfinance)](https://pypi.org/project/yfinance/)
- [NewsAPI](https://newsapi.org/) — for real-time stock news

---

## Developer

**Sarvesh Bhardwaj**  
---


## To-Do (Optional Future Features)

-  Add technical indicators (RSI, MACD)
-  Implement other deep learning models (GRU, Transformer)
-  Dashboard analytics with Plotly
-  Deploy on Heroku or Render
