import dbm
import json


class Cache:
    def __init__(self, db: "dbm._Database") -> None:
        self.db = db

    @staticmethod
    def try_load(cache_dir: str):
        try:
            db = dbm.open(cache_dir, "r")
            return Cache(db)
        except Exception:
            return None

    def get(self, identifier: str) -> dict | None:
        raw = self.db.get(identifier)
        if raw is None:
            return None

        return json.loads(raw)
