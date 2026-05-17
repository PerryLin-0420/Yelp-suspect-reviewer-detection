import sys
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import UserDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class UserFriendsExtractor:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_user.json")
    output_path: Path = field(init=False, default=DB_PATH)
    creator: UserDatabaseCreator = field(init=False, factory=UserDatabaseCreator)

    def main(self):
        try:
            self.creator.db_connection(output_path=self.output_path)
            self.creator.db_drop("user_friends")
            self.creator.create_user_friends_table()

            logger.info("[Info] Bulk-inserting friend pairs via SQL...")
            self.creator.db_begin()
            self.creator.insert_user_friends_bulk(self.input_path)
            self.creator.db_commit()
        except Exception as e:
            self.creator.db_rollback()
            logger.exception(f"[ERROR] {e}")
        finally:
            self.creator.db_close()


if __name__ == "__main__":
    extractor = UserFriendsExtractor()
    extractor.main()
