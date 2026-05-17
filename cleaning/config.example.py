from pathlib import Path

# Copy this file to config.py and fill in your actual paths.
# config.py is gitignored — never committed.

RAW_JSON_DIR   = Path("/path/to/Yelp-JSON")    # folder with yelp_academic_dataset_*.json
RAW_PHOTOS_DIR = Path("/path/to/Yelp-Photos")  # folder with photos.json
DB_PATH        = Path("/path/to/YELP.duckdb")  # output database file (created if not exists)
