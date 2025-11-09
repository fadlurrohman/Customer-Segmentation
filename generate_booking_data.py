import numpy as np
import pandas as pd

def generate_booking_data(n_customers: int = 2000, max_bookings_per_cust: int = 20, random_state: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    countries = np.array(['TH', 'TW', 'KR', 'JP', 'SG'])

    rows = []
    for cid in customer_ids:
        num_bookings = rng.integers(1, max_bookings_per_cust + 1)
        booking_dates = pd.to_datetime(rng.choice(pd.date_range('2024-01-01', '2025-03-31'), size=num_bookings))
        booking_dates = sorted(booking_dates)
        for d in booking_dates:
            nights = rng.integers(1, 8)
            price_per_night = rng.uniform(20, 200)
            revenue = nights * price_per_night
            rows.append({
                "customer_id": cid,
                "checkin_date": d,
                "nights": int(nights),
                "country": rng.choice(countries),
                "revenue_usd": float(round(revenue, 2)),
            })

    df = pd.DataFrame(rows)
    df.to_csv("booking_data.csv", index=False)
    print(f"Generated booking_data.csv with {len(df)} rows for {n_customers} customers")
    return df

if __name__ == "__main__":
    generate_booking_data()
