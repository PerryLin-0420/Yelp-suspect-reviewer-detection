import sys
from datetime import datetime
from typing import Literal
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import BusinessJsonCleaner, BusinessDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class BusinessHoursExtractor:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_business.json")
    output_path: Path = field(init=False, default=DB_PATH)
    chunk_size: int = field(init=False, default=5000)
    cleaner: BusinessJsonCleaner = field(init=False, factory=BusinessJsonCleaner)
    creator: BusinessDatabaseCreator = field(init=False, factory=BusinessDatabaseCreator)


    def extract_hour_dict_values(self, chunk_json):

        def day_time(day_dict: dict|str, day: str, timing: Literal["start", "end"]):
            try:
                if isinstance(day_dict, str) and day_dict == "null":
                    return None
                elif isinstance(day_dict, dict):
                    open_time = day_dict.get(day)
                    if not open_time or open_time in ("null", "0:0-0:0"):
                        return None
                    elif open_time and timing == "start":
                        start_time = datetime.strptime(open_time.split("-")[0], "%H:%M")
                        return start_time
                    elif open_time and timing == "end":
                        end_time = datetime.strptime(open_time.split("-")[1], "%H:%M")
                        return end_time
            except Exception as e:
                logger.error(f"[ERROR] Failed to parse time for {day} {timing}: {e}")
                return None

        rows = []

        for record in chunk_json:

            business_id = record.get("business_id")
            hours = record.get("hours") #it is a {}

            if not hours:
                continue

            rows.append((
            business_id,
            day_time(hours, "Monday",    "start"), day_time(hours, "Monday",    "end"),
            day_time(hours, "Tuesday",   "start"), day_time(hours, "Tuesday",   "end"),
            day_time(hours, "Wednesday", "start"), day_time(hours, "Wednesday", "end"),
            day_time(hours, "Thursday",  "start"), day_time(hours, "Thursday",  "end"),
            day_time(hours, "Friday",    "start"), day_time(hours, "Friday",    "end"),
            day_time(hours, "Saturday",  "start"), day_time(hours, "Saturday",  "end"),
            day_time(hours, "Sunday",    "start"), day_time(hours, "Sunday",    "end"),
        ))

        return rows

    def db_connection(self):
        self.creator.db_connection(output_path=self.output_path)

    def create_hour_table(self):
        self.creator.create_business_hours_table()

    def close_database(self):
        self.creator.db_close()

    def insert_hour_data(self, data):
        self.creator.insert_business_hours(data)



    def main(self):

        try:
            self.db_connection()
            self.creator.db_drop("business_hours")
            self.create_hour_table()

            for chunk in self.cleaner.clean_json(self.input_path):
                extracted = self.extract_hour_dict_values(chunk)
                if extracted:
                    raw_ids = list({row[0] for row in extracted})
                    bid_map = self.creator.get_business_id_map(raw_ids)
                    extracted = [(bid_map[row[0]],) + row[1:] for row in extracted if row[0] in bid_map]
                if extracted:
                    self.insert_hour_data(extracted)
        except Exception as e:
            logger.exception(f"[ERROR] {e}")
        finally:
            self.close_database()

if __name__ == "__main__":
    extractor = BusinessHoursExtractor()
    extractor.main()