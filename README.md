# WildGuard 🛡️
### Evidence-Based Wildlife-Highway Conflict Screening & Roadkill Method Validation

WildGuard is an open-source Python + Streamlit decision support platform designed to identify, rank, and validate ecological conflict zones where linear transport infrastructure intersects sensitive wildlife habitats.

Built for data-poor conservation landscapes, WildGuard implements a rigorous two-pronged framework:
1. **Algorithmic Validation:** Proves the predictive validity of the 1 km roadkill hotspot outlier detection method on verified, high-volume roadkill records from the **Taiwan Roadkill Observation Network (TaiRON)** using temporal holdout validation.
2. **Predictive Exposure Screening:** Applies multi-criteria habitat, sightings density, and road vulnerability scoring directly to the **Kanha–Pench Ecological Corridor in Madhya Pradesh (MP), India**, objectively prioritizing highway stretches for field surveys and mitigation structures.

---

## 📌 Project Architecture

```
wildguard/
├── app.py                      # Main Streamlit dashboard application
├── pages/                      # Multi-page dashboard modules
│   ├── 1_Overview.py           # Dataset KPI metrics, fallback rationale & study geography
│   ├── 2_Wildlife_Explorer.py  # MP mammal sightings explorer (points vs wide grid cells)
│   ├── 3_Roadkill_Validation.py# Temporal holdout validation on TaiRON roadkill records
│   ├── 4_Highway_Screening.py  # Ranked stretches near Kanha-Pench ("Predicted exposure")
│   └── 5_Data_Quality.py       # Cleaning funnel, uncertainty breakdown & sampling biases
├── src/
│   ├── config.py               # Central configuration, study bounding boxes & parameters
│   ├── pipeline.py             # Master orchestrator connecting fetch, clean, analysis & DB
│   ├── store.py                # SQLite (data/wildguard.db) & Parquet persistence layer
│   ├── ui_common.py            # Shared CSS, filters, loaders & map footnotes
│   ├── fetch/                  # Open data fetchers (GBIF API, Overpass API, geoBoundaries, IMD)
│   │   ├── gbif.py             # GBIF occurrence search & candidate dataset evaluator
│   │   ├── osm.py              # OpenStreetMap Overpass highway fetcher
│   │   ├── boundary.py         # geoBoundaries state and district polygon fetcher
│   │   └── weather.py          # IMD gridded precipitation and temperature climatology
│   ├── clean/
│   │   └── cleaning.py         # QA pipeline, uncertainty tagging & deliberate blur isolation
│   └── analysis/
│       ├── hotspots.py         # 1 km segmentation & Modified Z-Score outlier detection
│       ├── validation.py       # Temporal holdout, IoU tolerance, baselines & bootstrap CIs
│       ├── sightings.py        # Grid aggregation & tourist safari route bias analysis
│       └── screening.py        # Kanha-Pench composite exposure ranking model
├── data/
│   ├── raw/                    # Raw JSON and GeoJSON pulls with queries and timestamps
│   ├── processed/              # Parquet tables, GeoJSON layers and audit reports
│   └── wildguard.db            # SQLite relational database store
├── tests/
│   ├── test_clean.py           # Unit tests for cleaning and uncertainty tagging
│   ├── test_metrics.py         # Unit tests for modified Z-score, IoU, and validation metrics
│   └── test_screening.py       # Unit tests for composite exposure scoring
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 🔬 Core Methodologies

### 1. Country Candidate Selection & Fallback Logic
* **Target Country (India):** Evaluated the *India Roadkill Monitoring Project* (`f334ae5e-0991-44bb-b64b-d602f4c8c289`). The dataset holds only **491** records nationwide with **0** records in Madhya Pradesh, failing the minimum statistical power threshold of **1,000 georeferenced records with $\le 1\text{ km}$ coordinate uncertainty**.
* **Validated Fallback (Taiwan):** Selected the *Taiwan Roadkill Observation Network (TaiRON)* (`db09684b-0fd1-431e-b5fa-4c1532fbdb14`), providing **46,416** georeferenced roadkill occurrences to rigorously evaluate the outlier hotspot model.
* **Geographic Isolation:** Wildlife sightings and highway exposure screening remain strictly focused on **Madhya Pradesh, India**.

### 2. Precautionary Coordinate Uncertainty & Deliberate Blurring
To protect red-listed endangered mammals from poaching, iNaturalist automatically injects a deliberate **~30 km coordinate blur** on key species (*Tiger, Leopard, Gaur, Sambar, Sloth Bear, Dhole, Barasingha*).
* **Precise Records ($\le 1,000\text{ m}$):** Plotted as discrete points.
* **Blurred Records ($\ge 20,000\text{ m}$ or Endangered):** Never plotted as false precision points. Aggregated into wide **0.25° ($\approx 27\text{ km}$)** grid cells.
* **Prohibited Analytics:** Point-based spatial clustering (such as DBSCAN) is strictly prohibited on sightings data to prevent artificial clustering on rounded coordinates.

### 3. Robust Outlier Hotspot Rule
Highway ways are segmented into uniform 1 km stretches. Collision density outliers are identified using the Modified Z-score:
$$M_i = \frac{0.6745 \times (x_i - \text{median})}{\text{MAD}} > 3.5$$
Where $\text{MAD} = \text{median}(|x_i - \text{median}|)$. If $\text{MAD} = 0$ (common in sparse road networks where majority of segments have 0 collisions), the threshold automatically falls back to the **90th percentile of non-zero segments**.

### 4. Temporal Holdout Validation & Baselines
* Records are partitioned by time ($\text{Train} \le 2015$, $\text{Test} \ge 2016$).
* Spatial overlap is evaluated using **Intersection-over-Union (IoU)** with a 1 km tolerance buffer.
* Outlier model is benchmarked against two baseline models:
  1. **Random Ranking Baseline**
  2. **Density-Only Proximity Baseline**
* 95% Confidence Intervals are calculated via 1,000 bootstrap resamples.

### 5. Kanha–Pench Highway Exposure Screening
1 km stretches along the Kanha–Pench corridor (including NH-44) are scored on a 0–100 Composite Exposure Index:
* **Protected Area Proximity (40 pts):** Distance decay from Kanha and Pench Tiger Reserves.
* **Wildlife Sightings Density (40 pts):** Density of verified mammal observations within 13 km.
* **Road Class & Speed Vulnerability (20 pts):** Motorway/Trunk (NH-44) > Primary (SH) > Secondary.
* **Mandatory Label:** All screening results carry the explicit label **"Predicted exposure, not observed roadkill"** to prevent conflating potential traffic exposure with measured mortality.

---

## 🚀 Quickstart & Installation

### Requirements
- Python 3.11
- pip / virtualenv

### 1. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
pytest
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```
On first launch, WildGuard automatically queries open APIs, processes and cleans all datasets, compiles SQLite tables and Parquet files, and initializes the interactive dashboard.

---

## 📊 Data Quality & Transparent Limits

| Limitation | Impact on Analysis | WildGuard Safeguard |
| :--- | :--- | :--- |
| **Deliberate Taxon Blurring** | Red-listed species shifted by ~30 km | Restricted to wide 0.25° grid cells; never plotted as points |
| **Safari Track Bias** | 78%+ of sightings within 30 km of 5 tiger reserves | Acknowledged in UI; not interpreted as animal abundance |
| **Monsoon Park Closures** | Sightings collapse July–Sept; peak in April | Integrated with IMD climatology; not interpreted as migration |
| **India Roadkill Deficit** | Only 491 open roadkill records nationwide | Evaluated & documented; validated on TaiRON fallback dataset |

