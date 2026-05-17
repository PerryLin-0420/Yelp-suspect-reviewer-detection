import sys
import json
from attrs import define, field
from pathlib import Path
from loguru import logger
from Function import BusinessDatabaseCreator, BusinessJsonCleaner

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR, DB_PATH

@define
class BusinessAttributesExtractor:
    input_path: Path = field(init=False)
    output_path: Path = field(init=False)
    cleaner: any = field(init=False) 
    creator: BusinessDatabaseCreator = field(init=False, factory=BusinessDatabaseCreator)
    attribute_map: dict = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self.input_path = RAW_JSON_DIR / "yelp_academic_dataset_business.json"
        self.output_path = DB_PATH
        self.cleaner = BusinessJsonCleaner(chunk_size=2000)

    def parse_boolean(self, val):
        s_val = str(val).lower()
        if s_val in ['true', "u'true'", 'yes']: return True
        if s_val in ['false', "u'false'", 'no']: return False
        return None

    def extract_attribute_values(self, chunk_json):
        rows = []
        for record in chunk_json:
            bid = record.get("business_id")
            attrs = record.get("attributes") or {}
            for key, val in attrs.items():
                aid = self.attribute_map.get(key)
                if aid is None: continue

                # 處理字典型字串
                if isinstance(val, str) and val.startswith('{'):
                    try: val = json.loads(val.replace("'", '"'))
                    except: pass

                if isinstance(val, dict):
                    for sk, sv in val.items():
                        vb = self.parse_boolean(sv)
                        rows.append((bid, aid, None if vb is not None else str(sv), vb, None, None, sk))
                else:
                    vb = self.parse_boolean(val)
                    vn = float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
                    vt = str(val) if vb is None and vn is None else None
                    rows.append((bid, aid, vt, vb, vn, None, "root"))
        return rows

    def main(self):
        self.creator.db_connection(self.output_path)
        
        self.creator.db_drop("business_attribute")
        self.creator.db_drop("attribute_definition")
        self.creator.create_attribute_definition_table()
        self.creator.create_business_attribute_table()

        logger.info("Scanning attributes...")
        names = set()
        for chunk in self.cleaner.clean_json(self.input_path):
            for r in chunk:
                names.update((r.get("attributes") or {}).keys())
        
        self.creator.insert_attribute_definition([(n, "MIXED") for n in names])
        
        db_rows = self.creator.conn.execute("SELECT name, attribute_id FROM attribute_definition").fetchall()
        self.attribute_map = {n: aid for n, aid in db_rows}

        logger.info("Loading business ID map...")
        all_bid_rows = self.creator.conn.execute(
            "SELECT business_raw_id, business_id FROM business"
        ).fetchall()
        bid_map = {raw_id: biz_id for raw_id, biz_id in all_bid_rows}
        logger.info(f"Loaded {len(bid_map)} business IDs into memory")

        logger.info("Inserting data...")
        self.creator.conn.begin()
        try:
            for chunk in self.cleaner.clean_json(self.input_path):
                data_rows = self.extract_attribute_values(chunk)
                if data_rows:
                    data_rows = [(bid_map[row[0]],) + row[1:] for row in data_rows if row[0] in bid_map]
                if data_rows:
                    self.creator.insert_business_attributes(data_rows)
            self.creator.conn.commit()
        except Exception:
            self.creator.conn.rollback()
            raise
        
        logger.info("Done.")

if __name__ == "__main__":
    extractor = BusinessAttributesExtractor()
    try:
        extractor.main()
    finally:
        extractor.creator.db_close()