from datetime import date

from ingestion import FakeCameraAPI, Ingestor, MAX_ATTEMPTS


def make(tmp_path, api=None):
    return Ingestor(api or FakeCameraAPI(), "P1", ["C1", "C2", "C3"], tmp_path / "vol", tmp_path / "s.db")


def test_seven_images_per_camera_per_day(tmp_path):
    ok, failed = make(tmp_path).backfill(date(2026, 1, 1), date(2026, 1, 1))
    assert (ok, failed) == (21, 0)
    assert len(list((tmp_path / "vol" / "P1" / "C1" / "2026-01-01").iterdir())) == 7


def test_rerun_skips_done_images(tmp_path):
    ing = make(tmp_path)
    ing.backfill(date(2026, 1, 1), date(2026, 1, 2))
    assert ing.backfill(date(2026, 1, 1), date(2026, 1, 2)) == (0, 0)


def test_failed_download_is_retried_then_capped(tmp_path):
    bad = "C1-20260101-036"  # the 6 AM image
    ing = make(tmp_path, FakeCameraAPI(fail_ids={bad}))
    assert ing.backfill(date(2026, 1, 1), date(2026, 1, 1)) == (20, 1)
    assert len(ing.pending_retries()) == 1
    for _ in range(MAX_ATTEMPTS):
        ing.backfill(date(2026, 1, 1), date(2026, 1, 1))
    assert ing.pending_retries() == []  # gave up after MAX_ATTEMPTS, still visible in the table


def test_daily_job_processes_yesterday(tmp_path):
    ing = make(tmp_path)
    assert ing.run_daily(date(2026, 2, 10)) == (21, 0)
    assert (tmp_path / "vol" / "P1" / "C2" / "2026-02-09").exists()
