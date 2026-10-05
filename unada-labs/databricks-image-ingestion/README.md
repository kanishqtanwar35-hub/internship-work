# Camera image ingestion on Databricks

## The problem
We had 3 cameras on a project, each producing an image roughly every 10 minutes through an external API. One API gives you the list of images for a camera and a date (with thumbnails), and a second API gives you the high-resolution version for a given image ID.

We didn't need an image every 10 minutes for the analysis. That's 144 images per camera per day. One image every 2 hours between 6 AM and 6 PM was enough, which is 7 per camera per day, about 95% less data to download and store.

## What I built
**Storage.** High-res images go into Unity Catalog Volumes, organised as `Project / Camera / Date / images`, so anyone can find a specific day's images without a lookup.

**Metadata tables.** Alongside the files we kept tables for project and camera details, image metadata and source IDs, download status, job runs, attempt counts, timestamps and errors. This turned out to be the most important part of the whole thing.

**Backfill.** A notebook that pulls about 6 months of history with the same 2-hour sampling. I kept it as a notebook on purpose because it was a big one-time load and I wanted to run and watch it myself.

**Daily job.** A scheduled Databricks Job that runs every day for the previous day, across all 3 cameras and all 7 time slots. Once it was set up, nobody had to touch it.

## Why the status tables matter
Without them, every run is a fresh start. If the API times out halfway through, you either download everything again or you lose images without knowing.

With them, every run first checks what's already done. Finished images are skipped, failed ones are retried, and anything that keeps failing stops after a few attempts but stays visible in the table so someone can look at it. You can run the same job twice and nothing gets duplicated.

## What's in this folder
`ingestion.py` is the same logic running locally: a folder instead of a Unity Catalog Volume, SQLite instead of Delta tables, and a fake camera API that produces an image every 10 minutes.

```bash
python ingestion.py
```

It runs a 7-day backfill, a daily run, and then the same backfill again to show that nothing gets downloaded twice.
