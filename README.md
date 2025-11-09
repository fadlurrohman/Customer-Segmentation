# Project 3 – Customer Segmentation using RFM & Clustering

This project simulates hotel booking transactions and segments customers using:

- RFM (Recency, Frequency, Monetary) scoring.
- K-means clustering to group customers (e.g., VIP, frequent bookers, deal hunters).

## Files

- `generate_booking_data.py` – create synthetic hotel booking data.
- `rfm_segmentation.py` – build RFM features, run clustering, and output segment labels.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_booking_data.py
python rfm_segmentation.py
```
