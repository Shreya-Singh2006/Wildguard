"""
Test script to run GBIF dataset evaluation and fetch initial batch of data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fetch.gbif import determine_roadkill_dataset, fetch_mp_mammal_sightings, fetch_roadkill_data
from src.fetch.boundary import fetch_mp_state_boundary
from src.clean.cleaning import clean_sightings_data, clean_roadkill_data

print("Evaluating roadkill dataset...")
selection = determine_roadkill_dataset()
print("Selected dataset:", selection["selected_dataset"]["title"])
print("Reason:", selection["selected_dataset"]["selection_reason"])

print("\nFetching MP boundary...")
mp_bnd = fetch_mp_state_boundary()
print(f"MP boundary loaded: {len(mp_bnd)} features")

print("\nFetching MP mammal sightings...")
sightings_raw = fetch_mp_mammal_sightings()
print(f"Raw MP sightings fetched: {len(sightings_raw.get('records', []))}")

print("\nCleaning MP sightings...")
clean_sightings, audit = clean_sightings_data(sightings_raw.get("records", []), mp_bnd)
print(f"Clean MP sightings: {len(clean_sightings)} records")
print("Uncertainty breakdown:", audit.get("uncertainty_breakdown"))
print("Threatened blurred records:", audit.get("threatened_blurred_count"))

print("\nFetching roadkill data...")
selected_key = selection["selected_dataset"]["dataset_key"]
roadkill_raw = fetch_roadkill_data(selected_key, max_records=6000)
print(f"Raw roadkill fetched: {len(roadkill_raw.get('records', []))}")

print("\nCleaning roadkill data...")
clean_rk, rk_audit = clean_roadkill_data(roadkill_raw.get("records", []), selection["selected_dataset"]["country"])
print(f"Clean roadkill records: {len(clean_rk)}")
