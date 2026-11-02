"""
Fetch real open data from GBIF, geoBoundaries, and IMD.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_real_data")

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    logger.info("Starting WildGuard complete data pipeline...")
    result = run_full_pipeline(force_refresh=False)
    logger.info(f"Pipeline finished with result: {result}")
