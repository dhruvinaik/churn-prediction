# generate_auto_taxi_churn.py
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_auto_taxi_churn(n=5000, seed=42, target_churn_rate=0.25):
    rng = np.random.default_rng(seed)

    cities = ["Mumbai","Delhi","Pune","Bangalore","Hyderabad","Chennai","Kolkata","Ahmedabad"]
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    times = ["Morning","Afternoon","Evening","Night"]
    weathers = ["Clear","Cloudy","Rainy","Foggy"]
    payments = ["Cash","Card","UPI"]
    zones = ["Commercial","Residential","Mixed"]
    genders = ["Male","Female","Other"]
    trip_types = ["Auto","Taxi"]

    rows = []
    for i in range(n):
        customer_id = f"CUST{i:05d}"
        day = rng.choice(days)
        time_of_day = rng.choice(times)
        location = rng.choice(cities)
        weather = rng.choice(weathers, p=[0.6,0.2,0.15,0.05])
        payment_method = rng.choice(payments, p=[0.45,0.35,0.20])
        zone_type = rng.choice(zones, p=[0.4,0.45,0.15])
        gender = rng.choice(genders, p=[0.48,0.48,0.04])
        trip_type = rng.choice(trip_types, p=[0.7,0.3])

        # Usage patterns
        avg_trips_per_week = max(0, rng.poisson(5 if trip_type=="Auto" else 3) + rng.normal(0,1))
        avg_estimated_price = max(30, rng.normal(80 if trip_type=="Taxi" else 50, 15))
        avg_spent_per_week = avg_trips_per_week * avg_estimated_price + rng.normal(0,50)
        last_trip_days_ago = int(abs(rng.exponential(scale=10)))  # 0 = active, high = inactive
        tenure_days = int(abs(rng.exponential(scale=250))) + 10   # how long they’ve been customers

        rows.append({
            "customer_id": customer_id,
            "day_of_week": day,
            "time_of_day": time_of_day,
            "location": location,
            "weather": weather,
            "payment_method": payment_method,
            "zone_type": zone_type,
            "gender": gender,
            "trip_type": trip_type,
            "avg_trips_per_week": round(avg_trips_per_week, 2),
            "avg_estimated_price": round(avg_estimated_price, 2),
            "avg_spent_per_week": round(avg_spent_per_week, 2),
            "last_trip_days_ago": last_trip_days_ago,
            "tenure_days": tenure_days
        })

    df = pd.DataFrame(rows)

    # Calculate churn probability
    score = (
        -1.0
        - 0.4 * df["avg_trips_per_week"]
        - 0.001 * df["avg_spent_per_week"]
        + 0.07 * df["last_trip_days_ago"]
        - 0.002 * df["tenure_days"]
        + rng.normal(0,0.5,len(df))
    )
    p_raw = sigmoid(score)

    # Adjust to target churn rate
    mean_raw = p_raw.mean()
    shift = np.log(target_churn_rate/(1-target_churn_rate)) - np.log(mean_raw/(1-mean_raw))
    churn_prob = sigmoid(score + shift)

    df["churn_prob"] = churn_prob
    df["churn"] = (rng.random(len(df)) < churn_prob).astype(int)

    print(f"Generated {len(df)} rows. Churn rate: {df['churn'].mean():.2f}")

    df.to_csv("auto_taxi_churn.csv", index=False)
    print("Saved as auto_taxi_churn.csv")

if __name__ == "__main__":
    generate_auto_taxi_churn(n=5000)
