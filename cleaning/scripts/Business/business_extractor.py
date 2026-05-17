import sys
import pandas as pd
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import BusinessJsonCleaner, BusinessDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class BusinessExtractor:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_business.json")
    output_path: Path = field(init=False, default=DB_PATH)
    chunk_size: int = field(init=False, default=5000)
    cleaner: BusinessJsonCleaner = field(init=False, factory=BusinessJsonCleaner)
    creator: BusinessDatabaseCreator = field(init=False, factory=BusinessDatabaseCreator)
    valu_list: list = field(init=False, factory=list)



    def __attrs_post_init__(self):
        self.valu_list = [
            "business_id",
            "name",
            "address",
            "city",
            "state",
            "postal_code",
            "latitude",
            "longitude",
            "stars",
            "review_count",
            "is_open"
            ]

    def extract_non_dict_values(self, chunck_json):
        return [
        tuple(record.get(f) for f in self.valu_list)
        for record in chunck_json
    ]

    def db_connection(self):
        self.creator.db_connection(output_path=self.output_path)

    def create_database(self):
        self.creator.create_business_table()

    def close_database(self):
        self.creator.db_close()

    def insert_data(self, data):
        self.creator.inert_business_data(data)

    def main(self):
        try:
            self.db_connection()
            # FK child tables must be dropped before business
            for t in ("business_attribute", "business_hours", "business_category", "business"):
                self.creator.db_drop(table_name=t)
            self.create_database()
            for chunk in self.cleaner.clean_json(self.input_path):
                extracted_data = self.extract_non_dict_values(chunk)
                self.creator.inert_business_data(extracted_data)
        except Exception as e:
            logger.exception(f"[ERROR] {e}")
        finally:
            self.close_database()


if __name__ == "__main__":
    extractor = BusinessExtractor()
    extractor.main()