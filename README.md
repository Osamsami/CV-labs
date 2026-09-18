<div align="center">

# NYC Airbnb Market Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-unlicensed-lightgrey)

**Market segmentation of NYC Airbnb listings using K-Means clustering to uncover
distinct pricing and stay patterns across boroughs and room types.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyc-airbnb-clustering-analysis.streamlit.app/)

</div>

<br>

![Average Price per Cluster](visuals/price_per_cluster.png)

<br>

## Overview

This project explores the 2019 NYC Airbnb Open Data listings dataset and applies
unsupervised learning (K-Means) to segment listings into interpretable market
groups. The pipeline covers data cleaning, feature engineering, model selection
(Elbow Method + Silhouette Score), clustering, and profiling of the resulting
segments by price, room type, and borough.

<br>

## Dataset

- **Source**: NYC Airbnb Open Data (2019 snapshot), ~49,000 listings.
- **Cleaned dataset**: `data/airbnb_cleaned.csv` — listings with missing values
  handled and a `log_price` feature added.
- **Clustered dataset**: `data/airbnb_with_clusters.csv` — the cleaned dataset
  with an added `cluster` column (0–3), produced by running the notebook
  end-to-end. This file also powers the Streamlit demo.
- **Key columns**: `price`, `room_type`, `neighbourhood_group` (borough),
  `neighbourhood`, `latitude`, `longitude`, `minimum_nights`,
  `number_of_reviews`, `reviews_per_month`, `availability_365`.

<br>

## Model

- **Algorithm**: K-Means (scikit-learn), features standardised with `StandardScaler`.
- **Features used**: log-transformed price, minimum nights, number of reviews,
  reviews per month, availability (365-day), host listing count, neighbourhood
  group, and room type (categoricals label-encoded).
- **Model selection**: Elbow Method (inertia) and Silhouette Score across K=2–10.
- **Chosen K**: **4**, with a **silhouette score of ≈0.262**.

<div align="center">

| Elbow Method | Silhouette Scores |
|:---:|:---:|
| ![Elbow plot](visuals/elbow_plot.png) | ![Silhouette plot](visuals/silhouette_plot.png) |

</div>

<br>

## Results — Cluster Interpretation

Four distinct market segments emerged from the clustering:

| Cluster | Label | Avg Price | Dominant Room Type | Main Borough | Listings |
|---|---|---|---|---|---|
| 0 | Mid-Range Mixed | $121 | Entire/Private (51/48%) | Brooklyn (41%) | 6,465 |
| 1 | High-Value Entire Homes | $203 | Entire home (98%) | Manhattan (54%) | 21,839 |
| 2 | Budget Private Rooms | $76 | Private room (94%) | Brooklyn (47%) | 19,777 |
| 3 | Long-Stay Luxury Apts | $273 | Entire home (98%) | Manhattan (99%) | 564 |

- **Cluster 0 — Mid-Range Mixed Listings**: A balanced split between entire
  homes and private rooms, spread mainly across Brooklyn and Manhattan. This
  segment shows the highest review activity (~106 reviews on average),
  suggesting established, frequently-booked listings.
- **Cluster 1 — High-Value Entire Homes**: Almost entirely entire home/apartment
  listings concentrated in Manhattan, with longer minimum stays (~9 nights) —
  premium urban accommodation.
- **Cluster 2 — Budget Private Rooms**: The most affordable segment, dominated
  by private rooms in Brooklyn and the outer boroughs — ideal for
  budget-conscious short stays.
- **Cluster 3 — Long-Stay Luxury Apartments**: The smallest and most expensive
  segment, almost exclusively entire homes in Manhattan with a very high
  minimum stay (~30 nights) and the highest availability (~282 days/year) —
  likely monthly, corporate, or executive rentals.

<div align="center">

| Room Type per Cluster | Borough per Cluster |
|:---:|:---:|
| ![Room type per cluster](visuals/roomtype_per_cluster.png) | ![Neighbourhood per cluster](visuals/neighbourhood_per_cluster.png) |

</div>

See [`reports/analysis.md`](reports/analysis.md) for the full write-up of
methodology and results.

<br>

## Live Demo

🔗 **Try it live: [nyc-airbnb-clustering-analysis.streamlit.app](https://nyc-airbnb-clustering-analysis.streamlit.app/)**

The interactive Streamlit app (`app.py`) lets you filter listings by borough,
room type, and price range, and explore cluster distributions, summary
statistics, and a colour-coded scatter map of listings.

<br>

## Project Structure

```
data/       – cleaned & clustered dataset CSVs
notebooks/  – clustering notebook (data prep, modeling, profiling)
visuals/    – generated plots and charts
reports/    – analysis write-up (reports/analysis.md)
app.py      – Streamlit interactive cluster explorer
scripts/    – smoke test used by CI
```

<br>

## Installation

```bash
git clone https://github.com/Osamsami/nyc-airbnb-clustering-analysis.git
cd nyc-airbnb-clustering-analysis
python -m venv venv && source venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```

<br>

## Usage

**Run the notebook** (reproduces cleaning, modeling, and visuals):

```bash
jupyter notebook notebooks/notebooks_03_clustering.ipynb
```

**Run the Streamlit demo app**:

```bash
streamlit run app.py
```

**Run with Docker**:

```bash
docker build -t nyc-airbnb-clustering .
docker run -p 8501:8501 nyc-airbnb-clustering
```

<br>

## Technologies Used

- Python, pandas, NumPy
- scikit-learn (StandardScaler, KMeans, silhouette_score)
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook

<br>

<div align="center">

*A city's rent prices tell a story — but its rental listings tell it in far
more detail. Cluster by cluster, the data reveals not just what a stay
costs, but who it was built for.*

</div>
