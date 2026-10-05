"""Camera image ingestion: backfill + daily runs, with status tracking and retries.

Local version of the Databricks pipeline I built. On Databricks:
  - image files went to a Unity Catalog Volume  -> here: a local folder
  - status / metadata went to Delta tables       -> here: SQLite
  - the daily run was a scheduled Databricks Job -> here: run_daily()
The logic (sampling, status tracking, retries, safe re-runs) is the same.
"""
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SLOTS = [6, 8, 10, 12, 14, 16, 18]  # one image every 2 hours, 6 AM to 6 PM
MAX_ATTEMPTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS image_status (
    project_id TEXT, camera_id TEXT, slot_ts TEXT, image_id TEXT,
    status TEXT, attempts INTEGER DEFAULT 0, file_path TEXT,
    error TEXT, updated_at TEXT,
    PRIMARY KEY (project_id, camera_id, slot_ts)
);
CREATE TABLE IF NOT EXISTS job_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, job_type TEXT,
    started_at TEXT, finished_at TEXT, ok INTEGER, failed INTEGER
);
"""


def pick_image(images, slot_time):
    """From the 10-minute feed, pick the image closest to the slot time."""
    if not images:
        return None
    return min(images, key=lambda im: abs((im["captured_at"] - slot_time).total_seconds()))


class Ingestor:
    def __init__(self, api, project_id, cameras, root, db_path):
        """api needs list_images(camera_id, day) and download(camera_id, image_id) -> bytes"""
        self.api, self.project_id, self.cameras = api, project_id, cameras
        self.root = Path(root)
        self.db = sqlite3.connect(db_path)
        self.db.executescript(SCHEMA)

    def _status(self, camera, slot_ts):
        row = self.db.execute(
            "SELECT status, attempts FROM image_status "
            "WHERE project_id=? AND camera_id=? AND slot_ts=?",
            (self.project_id, camera, slot_ts)).fetchone()
        return row or (None, 0)

    def _save(self, camera, slot_ts, **fields):
        fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        upd = ", ".join(f"{c}=excluded.{c}" for c in fields)
        self.db.execute(
            f"INSERT INTO image_status (project_id, camera_id, slot_ts, {cols}) "
            f"VALUES (?, ?, ?, {marks}) "
            f"ON CONFLICT(project_id, camera_id, slot_ts) DO UPDATE SET {upd}",
            (self.project_id, camera, slot_ts, *fields.values()))
        self.db.commit()

    def ingest_day(self, day):
        ok = failed = 0
        for cam in self.cameras:
            images = self.api.list_images(cam, day)
            for hour in SLOTS:
                slot = datetime(day.year, day.month, day.day, hour)
                slot_ts = slot.isoformat()
                status, attempts = self._status(cam, slot_ts)
                # this check is what makes re-runs safe: finished images are skipped,
                # and an image that keeps failing stops after MAX_ATTEMPTS
                if status == "done" or attempts >= MAX_ATTEMPTS:
                    continue
                img = pick_image(images, slot)
                if img is None:
                    self._save(cam, slot_ts, status="missing", attempts=attempts + 1,
                               error="no image returned by API")
                    failed += 1
                    continue
                try:
                    data = self.api.download(cam, img["image_id"])
                    path = (self.root / self.project_id / cam / day.isoformat()
                            / f"{hour:02d}00_{img['image_id']}.jpg")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    self._save(cam, slot_ts, image_id=img["image_id"], status="done",
                               attempts=attempts + 1, file_path=str(path), error=None)
                    ok += 1
                except Exception as e:  # timeouts etc. get logged and retried on the next run
                    self._save(cam, slot_ts, image_id=img["image_id"], status="failed",
                               attempts=attempts + 1, error=str(e)[:200])
                    failed += 1
        return ok, failed

    def _run(self, job_type, days):
        started = datetime.now().isoformat(timespec="seconds")
        ok = failed = 0
        for d in days:
            a, b = self.ingest_day(d)
            ok, failed = ok + a, failed + b
        self.db.execute(
            "INSERT INTO job_runs (job_type, started_at, finished_at, ok, failed) VALUES (?,?,?,?,?)",
            (job_type, started, datetime.now().isoformat(timespec="seconds"), ok, failed))
        self.db.commit()
        return ok, failed

    def backfill(self, start, end):
        days = [start + timedelta(n) for n in range((end - start).days + 1)]
        return self._run("backfill", days)

    def run_daily(self, today=None):
        # runs for yesterday, so all 7 slots exist by the time the job runs
        today = today or date.today()
        return self._run("daily", [today - timedelta(1)])

    def pending_retries(self):
        return self.db.execute(
            "SELECT camera_id, slot_ts, attempts, error FROM image_status "
            "WHERE status != 'done' AND attempts < ?", (MAX_ATTEMPTS,)).fetchall()


class FakeCameraAPI:
    """Stand-in for the vendor API: one image every 10 minutes, all day."""

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)

    def list_images(self, camera_id, day):
        start = datetime(day.year, day.month, day.day)
        return [{"image_id": f"{camera_id}-{day:%Y%m%d}-{i:03d}",
                 "captured_at": start + timedelta(minutes=10 * i)} for i in range(144)]

    def download(self, camera_id, image_id):
        if image_id in self.fail_ids:
            raise TimeoutError("download timed out")
        return b"fake-jpeg-bytes"


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    ing = Ingestor(FakeCameraAPI(), "PRJ01", ["CAM1", "CAM2", "CAM3"], tmp / "volume", tmp / "status.db")
    print("backfill 7 days (ok, failed):", ing.backfill(date(2026, 3, 1), date(2026, 3, 7)))
    print("daily run (ok, failed):     ", ing.run_daily(date(2026, 3, 9)))
    print("same backfill again:        ", ing.backfill(date(2026, 3, 1), date(2026, 3, 7)), "<- nothing re-downloaded")
    print("10-min feed = 144 images/day per camera, we keep 7 -> %.1f%% less data" % (100 * (1 - 7 / 144)))
