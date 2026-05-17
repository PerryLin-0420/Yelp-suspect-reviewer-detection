import sys
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import UserDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class UserEliteExtractor:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_user.json")
    output_path: Path = field(init=False, default=DB_PATH)
    creator: UserDatabaseCreator = field(init=False, factory=UserDatabaseCreator)

    def main(self):
        try:
            self.creator.db_connection(output_path=self.output_path)
            self.creator.db_drop("user_elite")
            self.creator.create_user_elite_table()
            self.creator.insert_user_elite_bulk(self.input_path)
        except Exception as e:
            logger.exception(f"[ERROR] {e}")
        finally:
            self.creator.db_close()


if __name__ == "__main__":
    extractor = UserEliteExtractor()
    extractor.main()
