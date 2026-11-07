import struct

from loratrain.serve import LogTail, scan_samples, status_dict


def _png(name_bytes: bytes = b"\x89PNG\r\n\x1a\n"):
    return name_bytes


def test_scan_samples_orders_newest_first_and_parses_steps(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()

    first = folder / "mylora_00000200_00_42_0.png"
    second = folder / "mylora_00000400_00_42_1.png"
    first.write_bytes(_png())
    second.write_bytes(_png())
    (folder / "notes.txt").write_text("ignore me")

    # make the step-400 file newer
    import os

    os.utime(first, (1_000_000, 1_000_000))
    os.utime(second, (2_000_000, 2_000_000))

    samples = scan_samples(folder)

    assert [item["name"] for item in samples] == [second.name, first.name]
    assert samples[0]["step"] == 400
    assert samples[0]["url"] == "/samples/" + second.name


def test_scan_samples_missing_folder_is_empty(tmp_path):
    assert scan_samples(tmp_path / "nope") == []


def test_log_tail_reads_progress_written_in_place(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("steps:  10%| | 10/100 [00:01<00:09, 10.00it/s, loss=0.5]\r", encoding="utf-8")

    tail = LogTail(log)
    tail.poll()

    assert tail.status()["step"] == 10
    assert tail.status()["loss"] == 0.5

    with open(log, "a", encoding="utf-8") as handle:
        handle.write("steps:  20%| | 20/100 [00:02<00:08, 10.00it/s, loss=0.4]\r")

    tail.poll()

    status = tail.status()
    assert status["step"] == 20
    assert status["percent"] == 20.0
    assert len(status["points"]) == 2


def test_log_tail_survives_truncation(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("steps:  90%| | 900/1000 [00:09<00:01, 100.00it/s, loss=0.2]\r", encoding="utf-8")

    tail = LogTail(log)
    tail.poll()
    assert tail.status()["step"] == 900

    log.write_text("steps:  5%| | 5/1000 [00:01<03:00, 5.00it/s, loss=0.9]\r", encoding="utf-8")
    tail.poll()

    assert tail.status()["step"] == 5


def test_status_dict_without_data():
    from loratrain.progress import Monitor

    status = status_dict(Monitor())

    assert status["step"] is None
    assert status["points"] == []
    assert status["warnings"] == []
