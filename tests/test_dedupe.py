from loratrain.dedupe import find_duplicates


def _write(path, data):
    path.write_bytes(data)
    return path


def test_identical_files_are_one_exact_group(tmp_path):
    a = _write(tmp_path / "a.png", b"same-bytes")
    b = _write(tmp_path / "b.png", b"same-bytes")
    c = _write(tmp_path / "c.png", b"different")

    report = find_duplicates([a, b, c])

    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.kind == "exact"
    assert {path.name for path in group.members} == {"a.png", "b.png"}


def test_duplicate_paths_keeps_the_representative(tmp_path):
    a = _write(tmp_path / "a.png", b"same")
    b = _write(tmp_path / "b.png", b"same")

    report = find_duplicates([a, b])

    assert report.duplicate_paths() == {b}


def test_single_file_produces_no_groups(tmp_path):
    a = _write(tmp_path / "a.png", b"x")

    assert find_duplicates([a]).groups == []
