import orjson
from loguru import logger
from pathlib import Path
from attrs import define, field

@define
class BusinessJsonCleaner:
    chunk_size: int = field(init=True, default=5000)

    def clean_json(self, path: Path):
        chunk = []
        chunk_number = 0

        with open(path, "r", encoding="utf-8-sig") as f:
            for current_line, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError as e:
                    logger.exception(f"[ERROR] {current_line} failed: {e}")
                    continue

                chunk.append(record)

                if len(chunk) >= self.chunk_size:
                    chunk_number += 1
                    total = chunk_number * self.chunk_size

                    logger.info(f"[Success] Processed {total} records")

                    yield chunk
                    chunk = []

        if chunk:
            chunk_number += 1
            total = (chunk_number - 1) * self.chunk_size + len(chunk)

            logger.info(f"[Success] Processed {total} records")

            yield chunk