from pathlib import Path

from archive.collect import capture
from archive.storage import SqliteArchive
from plans.registry import PlanRegistry
from recordings.batches import read_recording
from recordings.source import RecordingSource
from review.query import query


SAMPLES = {
    "retest.jsonl": (2, ("x-first", "x-second"), [(0, 0), (0, 1)]),
    "retest_adaptive.jsonl": (3, ("z", "x", "y"), [(0, 0), (0, 1), (1, 0)]),
}


def main():
    sample_dir = Path(__file__).resolve().parent / "samples"
    with SqliteArchive.open(":memory:") as archive:
        for filename, (width, expected_ids, expected_values) in SAMPLES.items():
            plan = PlanRegistry().register("station-12", width, {"x": 1, "y": 2})
            with RecordingSource(sample_dir / filename) as source:
                capture_id = capture(archive, plan, read_recording(source))
            frames = archive.get(capture_id).evidence.frames
            frame_ids = tuple(frame.frame_id for frame in frames)
            values = query(archive, capture_id, 1, 2)
            if (frame_ids, values) != (expected_ids, expected_values):
                raise AssertionError(f"{filename}: frames={frame_ids!r}, values={values!r}")
            print(f"{filename}: frames={frame_ids!r}, values={values!r}")


if __name__ == "__main__":
    main()
