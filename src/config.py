"""
Configuration settings for WildGuard.
All coordinates, thresholds, fallback logic parameters, and paths are centrally defined here.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env configuration
env_file = BASE_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SQLITE_DB_PATH = DATA_DIR / "wildguard.db"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cache Configuration
CACHE_MAX_AGE_DAYS = int(os.getenv("CACHE_MAX_AGE_DAYS", "30"))

# GBIF API Credentials (Optional bulk download authentication)
GBIF_USER = os.getenv("GBIF_USER", "")
GBIF_PWD = os.getenv("GBIF_PWD", "")
GBIF_EMAIL = os.getenv("GBIF_EMAIL", "")

# Madhya Pradesh (India) Study Area
MP_CONFIG = {
    "name": "Madhya Pradesh, India",
    "country_code": "IN",
    "state_name": "Madhya Pradesh",
    # Bounding rectangle around MP: lon 74 to 83, lat 21 to 27
    "bbox": {
        "min_lon": 74.0,
        "max_lon": 83.0,
        "min_lat": 21.0,
        "max_lat": 27.0,
    },
    "taxon_key": 359,  # Mammalia
    "basis_of_record": "HUMAN_OBSERVATION",
    "min_year": 2000,
}

# Kanha-Pench Highway Screening Corridor (Central India)
KANHA_PENCH_CONFIG = {
    "name": "Kanha-Pench Ecological Corridor, MP",
    "bbox": {
        "min_lon": 79.1,
        "max_lon": 81.1,
        "min_lat": 21.4,
        "max_lat": 22.8,
    },
    "highways": ["motorway", "trunk", "primary"],
    "protected_areas": [
        {"name": "Kanha National Park", "lat": 22.33, "lon": 80.61, "buffer_km": 25},
        {"name": "Pench National Park", "lat": 21.76, "lon": 79.33, "buffer_km": 25},
    ],
}

# Threatened mammal species with deliberate ~30 km blurring on iNaturalist
# Must NEVER be plotted as points and NEVER mixed into precise point counts.
THREATENED_BLURRED_SPECIES = {
    "Panthera tigris": "Tiger",
    "Panthera pardus": "Leopard",
    "Bos gaurus": "Gaur",
    "Rusa unicolor": "Sambar",
    "Melursus ursinus": "Sloth Bear",
    "Cuon alpinus": "Dhole (Asiatic Wild Dog)",
    "Rucervus duvaucelii": "Barasingha (Swamp Deer)",
}

# Roadkill Dataset Candidates and Fallback Thresholds
ROADKILL_THRESHOLD = {
    "min_records": 1000,
    "max_uncertainty_meters": 1000,  # <= 1 km coordinate uncertainty
}

# Primary Candidate: India Roadkill Monitoring Project
INDIA_ROADKILL_CONFIG = {
    "country": "India",
    "country_code": "IN",
    "dataset_key": "f334ae5e-0991-44bb-b64b-d602f4c8c289",
    "title": "India Roadkill Monitoring Project",
    "bbox": {
        "min_lon": 68.0,
        "max_lon": 97.5,
        "min_lat": 6.5,
        "max_lat": 37.0,
    },
}

# Fallback Candidate 1: Taiwan Roadkill Observation Network (TaiRON)
TAIWAN_ROADKILL_CONFIG = {
    "country": "Taiwan",
    "country_code": "TW",
    "dataset_key": "db09684b-0fd1-431e-b5fa-4c1532fbdb14",
    "title": "The Taiwan Roadkill Observation Network Data Set (TaiRON)",
    "bbox": {
        "min_lon": 120.0,
        "max_lon": 122.1,
        "min_lat": 21.8,
        "max_lat": 25.4,
    },
    "validation_area_bbox": {
        # Representative highway network focal corridor (Taichung/Nantou Expressway & Provincial Hwy corridor)
        "min_lon": 120.5,
        "max_lon": 120.9,
        "min_lat": 23.8,
        "max_lat": 24.4,
    },
    "train_end_year": 2015,
    "test_start_year": 2016,
}

# Hotspot Analysis Parameters
HOTSPOT_CONFIG = {
    "segment_length_km": 1.0,
    "buffer_distance_m": 200.0,
    "modified_z_threshold": 3.5,
    "percentile_fallback": 90.0,  # Used if MAD == 0
}

# Validation Parameters
VALIDATION_CONFIG = {
    "spatial_tolerance_km": 1.0,
    "bootstrap_iterations": 1000,
    "bootstrap_ci": 0.95,
    "precision_at_k": [0.05, 0.10, 0.20],
}
