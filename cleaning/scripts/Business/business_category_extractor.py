import sys
import pandas as pd
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import BusinessJsonCleaner, BusinessDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class BusinessCategoryExtractor:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_business.json")
    output_path: Path = field(init=False, default=DB_PATH)
    chunk_size: int = field(init=False, default=5000)
    cleaner: BusinessJsonCleaner = field(init=False, factory=BusinessJsonCleaner)
    creator: BusinessDatabaseCreator = field(init=False, factory=BusinessDatabaseCreator)


    def extract_category_dict_values(self, chunk_json):

        rows = []

        for record in chunk_json:

            business_id = record.get("business_id")
            categories = record.get("categories")

            if not categories:
                continue

            for category in categories.split(","):
                category = category.strip()

                if category:
                    rows.append((business_id, category))

        return rows

    def db_connection(self):
        self.creator.db_connection(output_path=self.output_path)

    def create_category_table(self):
        self.creator.create_category_table()

    def create_business_category_table(self):
        self.creator.create_business_category_table()

    def close_database(self):
        self.creator.db_close()

    def insert_category_data(self, data):
        self.creator.insert_categories(data)

    def get_category_map(self, categories: list[str] = None):
        return self.creator.get_category_map(categories)

    def insert_business_category_data(self, data, category_map):
        self.creator.insert_business_categories(data, category_map)

    def main(self):

        try:

            self.db_connection()
            self.creator.db_drop("business_category")
            self.creator.db_drop("category")
            self.create_category_table()
            self.create_business_category_table()
            category_map={}

            for chunk in self.cleaner.clean_json(self.input_path):
                extracted = self.extract_category_dict_values(chunk)

                categories = list({cat for _, cat in extracted} - category_map.keys())  # 只取真正新的
                if categories:
                    self.insert_category_data(categories)
                    new_map = self.creator.get_category_map(categories)  # 只查新增的
                    category_map.update(new_map)  # 增量更新本地 map

                raw_ids = list({bid for bid, _ in extracted})
                bid_map = self.creator.get_business_id_map(raw_ids)
                extracted = [(bid_map[bid], cat) for bid, cat in extracted if bid in bid_map]

                self.insert_business_category_data(extracted, category_map)

        except Exception as e:
            logger.exception(f"[ERROR] {e}")
        finally:
            self.close_database()

if __name__ == "__main__":
    extractor = BusinessCategoryExtractor()
    extractor.main()