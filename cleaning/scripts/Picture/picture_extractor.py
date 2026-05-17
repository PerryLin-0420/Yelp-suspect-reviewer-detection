import sys
from loguru import logger
from pathlib import Path
from attrs import define, field
from Function import PictureDatabaseCreator

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_PHOTOS_DIR, DB_PATH

@define
class PictureExtractor:
    input_path: Path = field(init=False, default=RAW_PHOTOS_DIR / "photos.json")
    output_path: Path = field(init=False, default=DB_PATH)
    creator: PictureDatabaseCreator = field(init=False, factory=PictureDatabaseCreator)

    def main(self):
        try:
            self.creator.db_connection(output_path=self.output_path)
            self.creator.db_drop(table_name="photo")
            self.creator.db_drop(table_name="label")
            self.creator.create_label_table()
            self.creator.create_photo_table()
            self.creator.insert_label_data_bulk(self.input_path)
            self.creator.insert_photo_data_bulk(self.input_path)
        except Exception as e:
            logger.exception(f"[ERROR] {e}")
        finally:
            self.creator.db_close()


if __name__ == "__main__":
    extractor = PictureExtractor()
    extractor.main()
