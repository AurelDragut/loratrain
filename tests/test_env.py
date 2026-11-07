from loratrain.env import SdScripts, which_accelerate


def _scripts(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sdxl_train_network.py").write_text("")
    return folder


def test_resolve_accepts_the_scripts_folder(tmp_path):
    folder = _scripts(tmp_path / "sd-scripts")

    assert SdScripts._resolve(folder) == folder


def test_resolve_accepts_a_kohya_root_with_sd_scripts_inside(tmp_path):
    inner = _scripts(tmp_path / "kohya_ss" / "sd-scripts")

    assert SdScripts._resolve(tmp_path / "kohya_ss") == inner


def test_resolve_rejects_a_folder_without_scripts(tmp_path):
    (tmp_path / "models").mkdir()

    assert SdScripts._resolve(tmp_path) is None


def test_discover_uses_the_hint_first(tmp_path):
    _scripts(tmp_path / "kohya_ss" / "sd-scripts")

    found = SdScripts.discover(str(tmp_path / "kohya_ss"))

    assert found is not None
    assert found.root == (tmp_path / "kohya_ss" / "sd-scripts").resolve()


def test_which_accelerate_prefers_the_given_environment(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python"
    interpreter.write_text("")
    accelerate = bin_dir / "accelerate"
    accelerate.write_text("#!/bin/sh\n")
    accelerate.chmod(0o755)

    monkeypatch.setattr("loratrain.env.shutil.which", lambda name: None)

    assert which_accelerate(str(interpreter)) == str(accelerate)


def test_which_accelerate_falls_back_to_path(tmp_path, monkeypatch):
    on_path = tmp_path / "elsewhere" / "accelerate"
    on_path.parent.mkdir(parents=True)
    on_path.write_text("#!/bin/sh\n")
    on_path.chmod(0o755)

    monkeypatch.setattr("loratrain.env.shutil.which", lambda name: str(on_path))

    assert which_accelerate("/nonexistent/bin/python") == str(on_path)


def test_which_accelerate_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("loratrain.env.shutil.which", lambda name: None)

    assert which_accelerate(str(tmp_path / "python")) is None


def test_which_accelerate_ignores_a_non_executable_file(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "accelerate").write_text("not a program")
    (bin_dir / "accelerate").chmod(0o644)

    monkeypatch.setattr("loratrain.env.shutil.which", lambda name: None)

    assert which_accelerate(str(bin_dir / "python")) is None
