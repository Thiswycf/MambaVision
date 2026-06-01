import json
import os
import time
from copy import deepcopy
from types import SimpleNamespace


def normalize_cache_row(row):
    return {
        "genotype": row["genotype"],
        "top1": float(row["top1"]),
        "top5": None if row.get("top5") is None else float(row["top5"]),
        "loss": None if row.get("loss") is None else float(row["loss"]),
        "flops": float(row["flops"]) if row.get("flops") is not None else None,
        "params": float(row["params"]) if row.get("params") is not None else None,
        "throughput": float(row["throughput"]) if row.get("throughput") is not None else None,
    }


class EvaluationCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.lock_path = f"{cache_path}.lock"
        self._lock_fd = None
        self.cache = None

    def _acquire_lock(self, timeout=300):
        start_time = time.time()
        while True:
            try:
                self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return
            except FileExistsError:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise RuntimeError(f"Failed to acquire cache lock after {timeout} seconds")
                time.sleep(0.1)

    def _release_lock(self):
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except:
                pass
            try:
                os.unlink(self.lock_path)
            except:
                pass
            self._lock_fd = None

    def load(self):
        if not os.path.exists(self.cache_path):
            self.cache = {}
            return self.cache
        
        self._acquire_lock()
        try:
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        finally:
            self._release_lock()
        return self.cache

    def save(self, cache_data):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        self._acquire_lock()
        try:
            tmp_path = f"{self.cache_path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)
            os.replace(tmp_path, self.cache_path)
            self.cache = deepcopy(cache_data)
        finally:
            self._release_lock()

    def update(self, genotype, data):
        if self.cache is None:
            self.load()
        
        existing = self.cache[genotype]
        for key, value in data.items():
            if value is not None:
                existing[key] = value
        
        self.save(self.cache)
        return self.cache[genotype]

    def batch_update(self, rows):
        if self.cache is None:
            self.load()
        
        for row in rows:
            genotype = row["genotype"]
            existing = self.cache[genotype] if genotype in self.cache else {}
            for key, value in row.items():
                if value is not None:
                    existing[key] = value
            self.cache[genotype] = existing
        
        self.save(self.cache)
        return self.cache

    def get(self, genotype):
        if self.cache is None:
            self.load()
        return self.cache.get(genotype)

    def has(self, genotype):
        if self.cache is None:
            self.load()
        return genotype in self.cache

    def is_complete(self, genotype, cost_objective):
        row = self.get(genotype)
        if not row:
            return False
        cost_key = cost_objective if cost_objective == "throughput" else cost_objective
        if cost_objective == "throughput":
            return row.get("throughput") is not None
        else:
            return row.get(cost_key) is not None

    def __len__(self):
        if self.cache is None:
            self.load()
        return len(self.cache)

    def __contains__(self, genotype):
        return self.has(genotype)

    def __getitem__(self, genotype):
        return self.get(genotype)
