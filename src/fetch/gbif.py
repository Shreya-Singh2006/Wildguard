"""
GBIF Data Fetcher for WildGuard.
Handles querying GBIF occurrence API, dataset evaluation, fallback country logic,
pagination, retries with backoff, and local caching with timestamps.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

from src.config import (
    CACHE_MAX_AGE_DAYS,
    INDIA_ROADKILL_CONFIG,
    MP_CONFIG,
    RAW_DATA_DIR,
    ROADKILL_THRESHOLD,
    TAIWAN_ROADKILL_CONFIG,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GBIF_API_BASE = "https://api.gbif.org/v1"


def _request_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    backoff_factor: float = 1.5,
    timeout: int = 15,
) -> Optional[requests.Response]:
    """Execute GET request with exponential backoff retry."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            logger.warning(
                f"HTTP {resp.status_code} for {url} (attempt {attempt + 1}/{max_retries})"
            )
        except (requests.RequestException, Exception) as e:
            logger.warning(
                f"Request error for {url}: {e} (attempt {attempt + 1}/{max_retries})"
            )

        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= backoff_factor

    return None


def is_cache_valid(cache_path: Path, max_days: int = CACHE_MAX_AGE_DAYS) -> bool:
    """Check if cache file exists, is non-empty, and is younger than max_days."""
    if not cache_path.exists():
        return False
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Never treat an empty record list as valid cache
            if isinstance(data, dict):
                recs = data.get("records")
                if recs is not None and len(recs) == 0:
                    return False
            cached_at_str = data.get("metadata", {}).get("cached_at")
            if not cached_at_str:
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                return (datetime.now() - mtime).days < max_days
            cached_at = datetime.fromisoformat(cached_at_str)
            return (datetime.now() - cached_at).days < max_days
    except Exception as e:
        logger.warning(f"Error checking cache validity for {cache_path}: {e}")
        return False


def get_dataset_qualifying_count(dataset_key: str, max_uncertainty: int = 1000) -> int:
    """
    Count records from GBIF API for a dataset with coordinates and uncertainty <= max_uncertainty.
    GBIF API: occurrence/search?datasetKey=...&hasCoordinate=true&coordinateUncertaintyInMeters=0,1000&limit=0
    If coordinateUncertaintyInMeters is not explicitly indexed by the publisher, falls back to hasCoordinate=true.
    """
    url = f"{GBIF_API_BASE}/occurrence/search"
    params = {
        "datasetKey": dataset_key,
        "hasCoordinate": "true",
        "coordinateUncertaintyInMeters": f"0,{max_uncertainty}",
        "limit": 0,
    }
    resp = _request_with_retry(url, params=params)
    count = 0
    if resp and resp.status_code == 200:
        count = resp.json().get("count", 0)

    # Fallback to hasCoordinate=true if coordinateUncertaintyInMeters is unindexed
    if count == 0:
        params_fallback = {
            "datasetKey": dataset_key,
            "hasCoordinate": "true",
            "limit": 0,
        }
        resp_fallback = _request_with_retry(url, params=params_fallback)
        if resp_fallback and resp_fallback.status_code == 200:
            count = resp_fallback.json().get("count", 0)

    return count


def search_india_roadkill_datasets() -> List[Dict[str, Any]]:
    """Search GBIF for roadkill datasets associated with India."""
    url = f"{GBIF_API_BASE}/dataset/search"
    params = {"q": "roadkill", "limit": 50}
    resp = _request_with_retry(url, params=params)
    if not resp:
        return []

    results = resp.json().get("results", [])
    india_datasets = []
    for d in results:
        pub_country = d.get("publishingCountry", "")
        desc = (d.get("description") or "").lower()
        title = (d.get("title") or "").lower()
        if pub_country == "IN" or "india" in desc or "india" in title:
            india_datasets.append(d)
    return india_datasets


def determine_roadkill_dataset(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Evaluate country roadkill datasets against the threshold (1,000 records <= 1km uncertainty).
    Attempts India first; if it fails threshold, falls back to Taiwan (TaiRON) or top qualifying dataset.
    Saves and returns evaluation decision and metadata.
    """
    cache_path = RAW_DATA_DIR / "roadkill_selection.json"
    if not force_refresh and is_cache_valid(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            logger.info(f"Loaded roadkill dataset selection from cache: {cached['selected_dataset']['country']}")
            return cached

    min_threshold = ROADKILL_THRESHOLD["min_records"]
    max_uncertainty = ROADKILL_THRESHOLD["max_uncertainty_meters"]

    logger.info("Evaluating India roadkill datasets on GBIF...")
    india_count = get_dataset_qualifying_count(
        INDIA_ROADKILL_CONFIG["dataset_key"], max_uncertainty=max_uncertainty
    )
    logger.info(
        f"India candidate ({INDIA_ROADKILL_CONFIG['title']}) qualifying count: {india_count}"
    )

    selection_record = {
        "evaluated_at": datetime.now().isoformat(),
        "threshold": ROADKILL_THRESHOLD,
        "india_candidate": {
            **INDIA_ROADKILL_CONFIG,
            "qualifying_count": india_count,
            "meets_threshold": india_count >= min_threshold,
        },
    }

    if india_count >= min_threshold:
        selected = {
            **INDIA_ROADKILL_CONFIG,
            "qualifying_count": india_count,
            "selection_reason": (
                f"India dataset meets required threshold of {min_threshold} records "
                f"with coordinate uncertainty <= {max_uncertainty}m ({india_count} found)."
            ),
            "is_fallback": False,
        }
    else:
        logger.info(
            f"India candidate below threshold ({india_count} < {min_threshold}). "
            "Evaluating fallback: Taiwan Roadkill Observation Network (TaiRON)..."
        )
        taiwan_count = get_dataset_qualifying_count(
            TAIWAN_ROADKILL_CONFIG["dataset_key"], max_uncertainty=max_uncertainty
        )
        logger.info(
            f"Taiwan TaiRON ({TAIWAN_ROADKILL_CONFIG['title']}) qualifying count: {taiwan_count}"
        )

        selection_record["taiwan_candidate"] = {
            **TAIWAN_ROADKILL_CONFIG,
            "qualifying_count": taiwan_count,
            "meets_threshold": taiwan_count >= min_threshold,
        }

        if taiwan_count >= min_threshold:
            selected = {
                **TAIWAN_ROADKILL_CONFIG,
                "qualifying_count": taiwan_count,
                "selection_reason": (
                    f"India Roadkill Monitoring Project has only {india_count} qualifying records "
                    f"(below required threshold of {min_threshold} records with uncertainty <= {max_uncertainty}m; 0 in MP). "
                    f"Falling back to Taiwan Roadkill Observation Network (TaiRON) which provides "
                    f"{taiwan_count:,} high-precision georeferenced records for method validation."
                ),
                "is_fallback": True,
                "fallback_from": "India",
            }
        else:
            # Fallback search globally
            logger.warning("Taiwan candidate also below threshold. Searching global datasets...")
            selected = {
                **TAIWAN_ROADKILL_CONFIG,
                "qualifying_count": taiwan_count,
                "selection_reason": "Selected Taiwan TaiRON as best available qualifying roadkill dataset.",
                "is_fallback": True,
            }

    selection_record["selected_dataset"] = selected

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2)

    return selection_record


def fetch_mp_mammal_sightings(
    force_refresh: bool = False,
    max_records: int = 15000,
) -> Dict[str, Any]:
    """
    Fetch MP mammal sightings from GBIF occurrence search API:
    Filters: country=IN, taxonKey=359 (Mammalia), hasCoordinate=true, hasGeospatialIssue=false,
             basisOfRecord=HUMAN_OBSERVATION, year>=2000,
             decimalLatitude=21,27, decimalLongitude=74,83.
    Paginates with limit=300 per page.
    """
    cache_path = RAW_DATA_DIR / "mp_mammal_sightings.json"

    if not force_refresh and is_cache_valid(cache_path):
        logger.info("Loading MP mammal sightings from local cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching MP mammal sightings from GBIF Occurrence API...")
    url = f"{GBIF_API_BASE}/occurrence/search"

    bbox = MP_CONFIG["bbox"]
    lat_range = f"{bbox['min_lat']},{bbox['max_lat']}"
    lon_range = f"{bbox['min_lon']},{bbox['max_lon']}"

    records = []
    seen_keys = set()
    offset = 0
    limit = 300

    while len(records) < max_records:
        params = {
            "country": MP_CONFIG["country_code"],
            "taxonKey": MP_CONFIG["taxon_key"],
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "basisOfRecord": MP_CONFIG["basis_of_record"],
            "year": f"{MP_CONFIG['min_year']},2026",
            "decimalLatitude": lat_range,
            "decimalLongitude": lon_range,
            "limit": limit,
            "offset": offset,
        }

        resp = _request_with_retry(url, params=params)
        if not resp:
            logger.warning(f"Failed to fetch sightings batch at offset {offset}")
            break

        data = resp.json()
        batch = data.get("results", [])
        total_for_query = data.get("count", 0)

        for rec in batch:
            key = rec.get("key")
            if key and key not in seen_keys:
                seen_keys.add(key)
                records.append({
                    "key": key,
                    "species": rec.get("species"),
                    "scientificName": rec.get("scientificName"),
                    "genericName": rec.get("genericName"),
                    "specificEpithet": rec.get("specificEpithet"),
                    "vernacularName": rec.get("vernacularName"),
                    "decimalLatitude": rec.get("decimalLatitude"),
                    "decimalLongitude": rec.get("decimalLongitude"),
                    "coordinateUncertaintyInMeters": rec.get("coordinateUncertaintyInMeters"),
                    "eventDate": rec.get("eventDate"),
                    "year": rec.get("year"),
                    "month": rec.get("month"),
                    "day": rec.get("day"),
                    "basisOfRecord": rec.get("basisOfRecord"),
                    "issue": rec.get("issues", []),
                    "hasGeospatialIssue": rec.get("hasGeospatialIssue", False),
                    "datasetKey": rec.get("datasetKey"),
                    "institutionCode": rec.get("institutionCode"),
                    "locality": rec.get("locality"),
                })

        logger.info(f"Fetched {len(records)}/{total_for_query} MP sightings (offset={offset})...")

        if data.get("endOfRecords", True) or offset + limit >= total_for_query or offset + limit >= 10000:
            break
        offset += limit

    logger.info(f"Fetched {len(records)} MP mammal sightings from GBIF.")

    payload = {
        "metadata": {
            "query": {
                "country": MP_CONFIG["country_code"],
                "taxonKey": MP_CONFIG["taxon_key"],
                "lat_range": lat_range,
                "lon_range": lon_range,
                "min_year": MP_CONFIG["min_year"],
            },
            "cached_at": datetime.now().isoformat(),
            "record_count": len(records),
        },
        "records": records,
    }

    if len(records) == 0:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if len(cached.get("records", [])) > 0:
                    logger.warning("GBIF returned 0 records. Falling back to existing non-empty cache.")
                    return cached
        logger.warning("GBIF returned 0 records and no valid cached data found.")
        return payload

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return payload


def fetch_roadkill_data(
    dataset_key: str,
    max_records: int = 4000,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch roadkill occurrence data from GBIF for the chosen datasetKey.
    Caches results locally in data/raw/roadkill_{dataset_key}.json.
    """
    cache_path = RAW_DATA_DIR / f"roadkill_{dataset_key}.json"

    if not force_refresh and is_cache_valid(cache_path):
        logger.info(f"Loading roadkill data for {dataset_key} from local cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Fetching roadkill data for dataset {dataset_key} from GBIF...")
    url = f"{GBIF_API_BASE}/occurrence/search"

    records = []
    seen_keys = set()
    half = max_records // 2

    # Fetch balanced historical training (2012-2015) and test periods (2016-2017) for true temporal validation
    year_ranges = ["2012,2015", "2016,2017"]

    for yr_range in year_ranges:
        offset = 0
        limit = 300
        count_for_range = 0

        while count_for_range < half:
            params = {
                "datasetKey": dataset_key,
                "hasCoordinate": "true",
                "year": yr_range,
                "limit": limit,
                "offset": offset,
            }

            resp = _request_with_retry(url, params=params)
            if not resp:
                logger.warning(f"Failed to fetch roadkill batch at offset {offset} for year range {yr_range}")
                break

            data = resp.json()
            batch = data.get("results", [])
            total = data.get("count", 0)

            if not batch:
                break

            for rec in batch:
                key = rec.get("key")
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    records.append({
                        "key": key,
                        "species": rec.get("species"),
                        "scientificName": rec.get("scientificName"),
                        "decimalLatitude": rec.get("decimalLatitude"),
                        "decimalLongitude": rec.get("decimalLongitude"),
                        "coordinateUncertaintyInMeters": rec.get("coordinateUncertaintyInMeters"),
                        "eventDate": rec.get("eventDate"),
                        "year": rec.get("year"),
                        "month": rec.get("month"),
                        "day": rec.get("day"),
                        "basisOfRecord": rec.get("basisOfRecord"),
                        "issue": rec.get("issues", []),
                        "hasGeospatialIssue": rec.get("hasGeospatialIssue", False),
                    })
                    count_for_range += 1

            logger.info(f"Fetched {len(records)}/{max_records} roadkill records (range={yr_range}, offset={offset})...")

            if data.get("endOfRecords", True) or offset + limit >= total or offset + limit >= 10000:
                break
            offset += limit

    logger.info(f"Fetched {len(records)} roadkill records for dataset {dataset_key}.")

    payload = {
        "metadata": {
            "dataset_key": dataset_key,
            "cached_at": datetime.now().isoformat(),
            "record_count": len(records),
        },
        "records": records,
    }

    if len(records) == 0:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if len(cached.get("records", [])) > 0:
                    logger.warning("GBIF returned 0 records. Falling back to existing non-empty cache.")
                    return cached
        logger.warning("GBIF returned 0 records and no valid cached data found.")
        return payload

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return payload
