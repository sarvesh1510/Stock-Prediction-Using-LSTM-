from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import os
import glob
from flask import Flask, render_template, request
import requests
from datetime import datetime

# Initialize Flask app and database
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# Ensure necessary directories exist
MODEL_DIR = "models"
import os

# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure images are saved inside the correct static folder
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")

# Create the directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------
# DATABASE MODEL
# ---------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# -------------------------------
# NEWS FETCHING FUNCTION
# -------------------------------
import requests

from datetime import datetime

import requests
from datetime import datetime, timedelta

def get_market_news():
    try:
        api_key = '_______________________'  # Your API Key
        # Define the date range (e.g., past 1 day)
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q=stock%20market&language=en&from={yesterday.strftime('%Y-%m-%d')}"
            f"&to={today.strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={api_key}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data['totalResults'] == 0:
            print("No stock news found.")
            return []

        # Convert 'publishedAt' string to datetime object
        for article in data.get('articles', []):
            published_at = article.get('publishedAt', '')
            if published_at:
                article['publishedAt'] = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')  # Adjust format if needed

        # Return top 5 stock-related articles
        return data.get('articles', [])[:6]

    except requests.exceptions.RequestException as e:
        print(f" Error fetching stock market news: {e}")
        return []

# Example usage
stock_news = get_market_news()
if stock_news:
    print("Latest stock market news fetched successfully.")
    for article in stock_news:
        print(f"Title: {article['title']}, Source: {article['source']['name']}")
else:
    print("No stock news articles found.")

    
from datetime import datetime

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    if not value:
        return ""
    try:
        # If the value is a string in ISO format, parse it into a datetime object
        if isinstance(value, str):
            value = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')  # Adjust format if needed
        # Format the datetime object as required
        return value.strftime(format)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------
# HOME ROUTE (Dashboard)
# ---------------------------------
@app.route('/')
def home():
    try:
        # Fetch today's and yesterday's closing prices
        nifty_data = yf.Ticker("^NSEI").history(period="2d")["Close"]
        sensex_data = yf.Ticker("^BSESN").history(period="2d")["Close"]

        if len(nifty_data) < 2 or len(sensex_data) < 2:
            raise ValueError("Not enough data available")

        # Extract latest and previous close values
        nifty_today, nifty_yesterday = nifty_data.iloc[-1], nifty_data.iloc[-2]
        sensex_today, sensex_yesterday = sensex_data.iloc[-1], sensex_data.iloc[-2]

        # Calculate percentage change
        nifty_change = round(((nifty_today - nifty_yesterday) / nifty_yesterday) * 100, 2)
        sensex_change = round(((sensex_today - sensex_yesterday) / sensex_yesterday) * 100, 2)

        # Round values to 2 decimal places
        nifty_price = round(nifty_today, 2)
        sensex_price = round(sensex_today, 2)

    except Exception as e:
        print(f" Error fetching NIFTY 50 and SENSEX: {e}")
        nifty_price, sensex_price, nifty_change, sensex_change = "Data Unavailable", "Data Unavailable", 0, 0

    # Fetch market news
    news_articles = get_market_news()
    print(f"Fetched {len(news_articles)} articles")  # Log the number of articles fetched

    # Check if user is logged in
    user_logged_in = 'user_id' in session
    user_name = None

    if user_logged_in:
        try:
            user = db.session.get(User, session['user_id'])  # Updated to use db.session.get()
            if user:  # Check if the user exists in the database
                user_name = user.username
            else:
                user_name = 'Guest'  # Handle case where user is not found
        except Exception as e:
            print(f" Error fetching user: {e}")
            user_name = 'Guest'

    return render_template('dashboard.html', 
                           nifty_price=nifty_price, sensex_price=sensex_price,
                           nifty_change=nifty_change, sensex_change=sensex_change,
                           news_articles=news_articles,
                           user_logged_in=user_logged_in,  # Pass whether the user is logged in
                           user_name=user_name)  # Pass the user's name


# ---------------------------------
# REGISTER ROUTE
# ---------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# ---------------------------------
# LOGIN ROUTE
# ---------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Set user_id in session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

# ---------------------------------
# LOGOUT ROUTE
# ---------------------------------
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

from datetime import datetime
# ---------------------------------
# FUNCTION TO FETCH STOCK DATA
# ---------------------------------
# Function to format large numbers (millions or billions)

def format_number(num):
    if num == 'Unavailable' or num == 'N/A':
        return num
    try:
        num = float(num)
        if num >= 1e12:
            return f'{num / 1e12:.2f}T'  # Trillions
        elif num >= 1e9:
            return f'{num / 1e9:.2f}B'  # Billions
        elif num >= 1e6:
            return f'{num / 1e6:.2f}M'  # Millions
        else:
            return f'{num:.2f}'  # Numbers less than a million
    except (ValueError, TypeError):
        return num  # Return the original value if there's an issue

    
def get_stock_data(symbol):
    try:
        data = yf.download(symbol, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        if data.empty:
            return None
        data = data[['Close']].dropna().reset_index()

        # Ensure all columns have the same length
        if len(data['Close']) != len(data['Date']):
            print(f" Data length mismatch for {symbol}")
            return None

        return data
    except Exception as e:
        print(f" Error fetching stock data for {symbol}: {e}")
        return None

# ---------------------------------
# FUNCTION TO PREPROCESS DATA
# ---------------------------------
def preprocess_stock_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    print(f" Before Scaling:\n", data[['Close']].tail())
    print(f" After Scaling:\n", scaled_data[-5:])

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_sequences(dataset, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(dataset)):
            X.append(dataset[i - seq_length:i, 0])
            y.append(dataset[i, 0])
        return np.array(X).reshape(-1, seq_length, 1), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    return X_train, y_train, X_test, y_test, scaler

# ---------------------------------
# PROFILE ROUTE
# ---------------------------------
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to view your profile.', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        # Get the new password from the form
        new_password = request.form.get('password')

        if new_password:
            # Hash the new password using pbkdf2:sha256
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')

            # Update the user's password in the database
            user.password = hashed_password
            db.session.commit()  # Assuming you're using SQLAlchemy to handle your database

            flash('Password updated successfully!', 'success')
            return redirect(url_for('profile'))

        flash('Please enter a new password.', 'warning')

    return render_template('profile.html',
                           user_logged_in=True,
                           user_name=user.username)

# ---------------------------------
# SETTINGS ROUTE
# ---------------------------------
from flask import render_template, request, session

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access settings.', 'warning')
        return redirect(url_for('login'))

    # Fetch the logged-in user from the database
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('login'))

    # Handle form submission
    if request.method == 'POST':
        stock_alerts = request.form.get('stock_alerts') == 'on'  # Check if checkbox is checked
        price_threshold = float(request.form.get('price_threshold', 100.0))  # Default to 100.0 if not provided

        # Update session with the new settings
        session['user_stock_alerts'] = stock_alerts
        session['user_price_threshold'] = price_threshold

        flash('Settings updated successfully!', 'success')
        return redirect(url_for('settings'))

    # Pass user information and settings to the template
    return render_template('settings.html',
                           user_logged_in=True,
                           user_name=user.username,
                           user_stock_alerts=session.get('user_stock_alerts', False),
                           user_price_threshold=session.get('user_price_threshold', 100.0))


# ---------------------------------
# RECENT ACTIVITY ROUTE
# ---------------------------------
@app.route('/recent-activity')
def recent_activity():
    if 'user_id' not in session:
        flash('Please log in to view recent activity.', 'warning')
        return redirect(url_for('login'))
    
    user_name = User.query.get(session['user_id']).username
    recent_stocks = session.get('recent_activity', [])

    return render_template('recent_activity.html',
                           user_logged_in=True,
                           user_name=user_name,
                           recent_stocks=recent_stocks)

# ---------------------------------
# PREDICT ROUTE
# ---------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to access predictions.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        symbols = request.form.get('symbols')

        if not symbols:
            flash('Please enter at least one stock symbol!', 'danger')
            return redirect(url_for('home'))

        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        predicted_prices = {}
        images = {}
        current_prices = {}
        price_movements = {}  # Stores movement status (increase/decrease)
        market_caps = {}  # Stores market cap data
        net_incomes = {}  # Stores net income data
        revenues = {}  # Stores revenue data

        # Store recent activity in session (Limit to last 5 searches)
        if 'recent_activity' not in session:
            session['recent_activity'] = []

        for symbol in symbol_list:
            stock_data = get_stock_data(symbol)

            if stock_data is None:
                flash(f'Stock symbol {symbol} not found!', 'danger')
                continue

            # Fetch latest stock price
            try:
                stock_info = yf.Ticker(symbol).history(period="1d")
                current_price = round(stock_info['Close'].iloc[-1], 2)  # Get latest closing price
                current_prices[symbol] = current_price
            except:
                current_prices[symbol] = "Unavailable"

            # Fetch additional stock information (Market Cap, Net Income, Revenue)
            try:
                stock_info = yf.Ticker(symbol).info
                market_caps[symbol] = stock_info.get('marketCap', 'N/A')
                net_incomes[symbol] = stock_info.get('netIncomeToCommon', 'N/A')
                revenues[symbol] = stock_info.get('totalRevenue', 'N/A')
            except:
                market_caps[symbol] = "Unavailable"
                net_incomes[symbol] = "Unavailable"
                revenues[symbol] = "Unavailable"

            # Data Preprocessing & Prediction
            X_train, y_train, X_test, y_test, scaler = preprocess_stock_data(stock_data)
            model = load_or_train_model(symbol, X_train, y_train)
            predicted_price = round(make_prediction(model, X_test, scaler), 2)
            predicted_prices[symbol] = predicted_price

            # Determine Price Movement (Up/Down)
            if isinstance(current_prices[symbol], (int, float)):
                price_movements[symbol] = "up" if predicted_price > current_prices[symbol] else "down"
            else:
                price_movements[symbol] = "unknown"

            images[symbol] = plot_predictions(stock_data, predicted_price, symbol)

            # Store in session
            session['recent_activity'].append({'symbol': symbol, 'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

        session['recent_activity'] = session['recent_activity'][-5:]  # Keep only last 5 records
        session.modified = True  # Ensure changes persist

        return render_template(
            'predict.html',
            predicted_prices=predicted_prices,
            images=images,
            current_prices=current_prices,
            price_movements=price_movements,  # Send price movement data
            market_caps=market_caps,  # Send market cap data
            net_incomes=net_incomes,  # Send net income data
            revenues=revenues,  # Send revenue data
            format_number=format_number  # Pass the format_number function
        )

    return render_template('predict.html')


# ---------------------------------
# FUNCTION TO LOAD OR TRAIN LSTM MODEL
# ---------------------------------
def load_or_train_model(symbol, X_train, y_train):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm.h5")

    if os.path.exists(model_path):
        print(f" Loading existing model for {symbol}")
        return load_model(model_path)

    print(f" Training new LSTM model for {symbol}...")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=28, batch_size=32, verbose=1)
    model.save(model_path)

    return model

# ---------------------------------
# FUNCTION TO MAKE PREDICTIONS
# ---------------------------------
def make_prediction(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    print(f" Predicted Scaled Prices (Before Clipping):\n", predicted_prices[-5:])
    predicted_prices = np.clip(predicted_prices, 0, 1)
    print(f" Predicted Scaled Prices (After Clipping):\n", predicted_prices[-5:])
    predicted_prices = scaler.inverse_transform(predicted_prices)
    print(f" Final Predicted Prices:\n", predicted_prices[-5:])
    return predicted_prices[-1][0]

# ---------------------------------
# FUNCTION TO PLOT PREDICTIONS
# ---------------------------------
def plot_predictions(stock_data, predicted_price, symbol):
    image_folder = IMAGE_DIR
    os.makedirs(image_folder, exist_ok=True)

    # Delete old images for this stock before saving a new one
    old_images = glob.glob(os.path.join(image_folder, f"{symbol}_prediction*.png"))
    for old_image in old_images:
        os.remove(old_image)
        print(f" Deleted old graph: {old_image}")

    # Generate a new unique filename for the graph
    image_filename = f"{symbol}_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    file_path = os.path.join(image_folder, image_filename)

    print(f" Generating new graph for {symbol} at: {file_path}")

    # Ensure 'Date' is in datetime format for plotting
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Plot the actual stock price history
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], label=f'{symbol} Actual Stock Price', color='blue')
    # Mark the prediction point and predicted price
    plt.axvline(x=stock_data['Date'].iloc[-1], color='red', linestyle='--', label='Prediction Point')
    plt.scatter(stock_data['Date'].iloc[-1], predicted_price, color='green', s=100, label='Predicted Price')
    plt.legend()
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    if os.path.exists(file_path):
        print(f" New graph saved: {file_path}")
    else:
        print(f" Failed to save graph for {symbol}!")

    return image_filename

from flask import jsonify
@app.route('/market-data')
def market_data():
    try:
        # Add ADANIENT.NS (Adani Enterprises) to ensure at least one positive stock
        nifty50_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "KOTAKBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "ASIANPAINT.NS",
            "ADANIENT.NS" ,"GRASIM.NS" # Added Adani Enterprises
        ]

        # Fetch stock prices for today and yesterday
        stock_data = yf.download(nifty50_tickers, period="2d")['Close']

        if stock_data.shape[0] < 2:
            return jsonify({"error": "Not enough data available"})

        # Get latest closing price and previous day's closing price
        latest_close = stock_data.iloc[-1]  # Last available closing price
        prev_close = stock_data.iloc[-2]    # Day before last closing price

        # Ensure there are no NaN values in data
        # Instead of modifying in place, create a new DataFrame to avoid SettingWithCopyWarning
        latest_close = latest_close.dropna()
        prev_close = prev_close.dropna()  # Create a new DataFrame instead of modifying in place

        # Calculate percentage change correctly
        percentage_change = ((latest_close - prev_close) / prev_close) * 100

        # Convert to dictionary and filter valid results
        stock_changes = percentage_change.dropna().to_dict()

        # Sort stocks correctly
        top_gainers = sorted(stock_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        top_losers = sorted(stock_changes.items(), key=lambda x: x[1])[:5]

        #  If no gainers exist, show the least negative stocks
        if all(change < 0 for _, change in top_gainers):
            top_gainers = sorted(stock_changes.items(), key=lambda x: x[1], reverse=True)[:5]

        # Fix the sign formatting to remove "±" issues
        gainers_list = [
            {"symbol": g[0], "change": f"{'+' if g[1] > 0 else ''}{round(g[1], 2)}"} for g in top_gainers
        ]
        losers_list = [
            {"symbol": l[0], "change": f"{round(l[1], 2)}"} for l in top_losers
        ]

        return jsonify({"gainers": gainers_list, "losers": losers_list})

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------
# RUN APP
# ---------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensure the database tables exist
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)

# -------------------------------
# FETCH TOP GAINERS & LOSERS
# -------------------------------

app = Flask(__name__)