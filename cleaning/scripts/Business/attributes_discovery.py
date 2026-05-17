import sys
from attrs import define, field
from pathlib import Path
import pandas as pd
import duckdb
from Function import BusinessJsonCleaner

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_JSON_DIR

@define
class AttributesDiscovery:
    input_path: Path = field(init=False, default=RAW_JSON_DIR / "yelp_academic_dataset_business.json")
    cleaner: BusinessJsonCleaner = field(init=False, factory=BusinessJsonCleaner)
    records: list = field(init=False, factory=list)

    def detect_type(self, value):

        if value is None:
            return "NULL"

        if isinstance(value, dict):
            return "DICT"

        if isinstance(value, list):
            return "LIST"

        if isinstance(value, str):

            v = value.strip()

            if v in ("True", "False"):
                return "BOOL_STR"

            if v.isdigit():
                return "NUM_STR"

            if v.startswith("{") and v.endswith("}"):
                return "DICT_STR"

            if v.startswith("[") and v.endswith("]"):
                return "LIST_STR"

            return v

        return "OTHER"

    def extract_attribute_dict_values(self, chunk_json):

        for record in chunk_json:

            attributes = record.get("attributes")

            if not attributes:
                continue

            for key, value in attributes.items():

                attr_type = self.detect_type(value)

                self.records.append({
                    "name": key,
                    "type": attr_type
                })

    def main(self):

        try:

            for chunk in self.cleaner.clean_json(self.input_path):

                self.extract_attribute_dict_values(chunk)

            df = pd.DataFrame(self.records)

            df.to_parquet(
                Path(__file__).parents[2] / "parquets" / "attribute_map.parquet",
                index=False
            )

        except Exception as e:
            print(f"Error processing dataset: {e}")

    def aggregate_attribute_types(self):
        parquet_path = Path(__file__).parents[2] / "parquets" / "attribute_map.parquet"
        query = f'''
        SELECT 
        name,
        type,
        COUNT(*) as frequency
        FROM read_parquet('{parquet_path}')
        GROUP BY name, type
        '''
        try:
            result = duckdb.query(query).to_df()
            result.to_csv(
                Path(__file__).parent / "attribute_type_aggregation.csv",
                index=False
            )

        except Exception as e:
            print(f"Error aggregating attribute types: {e}")

if __name__ == "__main__":

    discovery = AttributesDiscovery()
    discovery.main()
    discovery.aggregate_attribute_types()