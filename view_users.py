from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
import pandas as pd

# Path to your database file
DATABASE_URI = "sqlite:///C:/Users/bhard/Major project/Stock prediction using LSTM/database/stock_data.db"

# Create a connection using SQLAlchemy
engine = create_engine(DATABASE_URI)

# Read the User table into a Pandas DataFrame
query = "SELECT * FROM User"
df = pd.read_sql(query, engine)

# Display the DataFrame
print(df)
df.to_csv("user_table.csv", index=False)  # Saves as CSV
df.to_excel("user_table.xlsx", index=False)  # Saves as Excel