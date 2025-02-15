import pandas as pd
import os

# Load datasets
demand_data = pd.read_csv("../data/raw/train.csv")  # Change filename as needed
supply_data = pd.read_csv("../data/raw/supply_chain.csv")  # Change filename

# Clean demand data
demand_data = demand_data.dropna()  # Remove missing values
demand_data['date'] = pd.to_datetime(demand_data['date'])  # Convert date column

# Clean supply data
supply_data = supply_data.drop_duplicates()  # Remove duplicates
supply_data = supply_data.fillna(method='ffill')  # Fill missing values

# Save processed data
os.makedirs("../data/processed", exist_ok=True)
demand_data.to_csv("../data/processed/demand_cleaned.csv", index=False)
supply_data.to_csv("../data/processed/supply_cleaned.csv", index=False)

print("✅ Data Preprocessing Completed!")
