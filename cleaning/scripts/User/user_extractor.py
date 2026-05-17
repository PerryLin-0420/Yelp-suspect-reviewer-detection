import sys
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import UserJsonCleaner, UserDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class UserExtractor:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_user.json")
    output_path: Path = field(init=False, default=DB_PATH)
    chunk_size: int = field(init=False, default=50_000)
    cleaner: UserJsonCleaner = field(init=False, default=None)
    creator: UserDatabaseCreator = field(init=False, factory=UserDatabaseCreator)
    value_list: list = field(init=False, factory=list)

    def __attrs_post_init__(self):
        self.cleaner = UserJsonCleaner(chunk_size=self.chunk_size)
        self.value_list = [
            "user_id",
            "name",
            "yelping_since",
            "review_count",
            "fans",
            "average_stars",
            "useful",
            "funny",
            "cool",
            "compliment_hot",
            "compliment_more",
            "compliment_profile",
            "compliment_cute",
            "compliment_list",
            "compliment_note",
            "compliment_plain",
            "compliment_cool",
            "compliment_funny",
            "compliment_writer",
            "compliment_photos",
        ]

    def extract_user_values(self, chunk_json):
        return [
            tuple(record.get(f) for f in self.value_list)
            for record in chunk_json
        ]

    def main(self):
        try:
            self.creator.db_connection(output_path=self.output_path)
            for t in ("user_friends", "user_elite", "user"):
                self.creator.db_drop(table_name=t)
            self.creator.create_user_table()
            self.creator.db_begin()
            for chunk in self.cleaner.clean_json(self.input_path):
                extracted_data = self.extract_user_values(chunk)
                self.creator.insert_user_data(extracted_data)
            self.creator.db_commit()
        except Exception as e:
            self.creator.db_rollback()
            logger.exception(f"[ERROR] {e}")
        finally:
            self.creator.db_close()


if __name__ == "__main__":
    extractor = UserExtractor()
    extractor.main()
