import csv
from pathlib import Path
from datetime import datetime


class CSVLogger:
    """Append-only CSV logger, safe to call every round across long runs.

    Creates the file (and header) on first use if it doesn't exist yet.
    Subsequent calls just append a row - safe even if the process crashes
    and gets restarted, or if multiple folds write to the same summary file.
    """

    def __init__(self, filepath, fieldnames):
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.filepath.exists()
        if write_header:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        # fill missing keys with empty string, ignore any extra keys
        clean_row = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_row)


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")