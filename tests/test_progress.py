import math

from loratrain.progress import Monitor, Sample, format_duration, parse_line

BAR = "steps:  16%|█▌        | 320/2000 [02:30<13:10, 2.13it/s, loss=0.128]"


def test_parse_standard_bar():
    sample = parse_line(BAR)

    assert sample is not None
    assert sample.step == 320
    assert sample.total == 2000
    assert sample.loss == 0.128
    assert sample.it_per_s == 2.13
    assert sample.elapsed == 150
    assert sample.remaining == 790


def test_parse_takes_the_last_paint_of_a_carriage_return_chunk():
    chunk = "steps: 5%| | 100/2000 [00:10<03:00, 5.00it/s, loss=0.5]\r" + BAR

    sample = parse_line(chunk)

    assert sample is not None
    assert sample.step == 320


def test_parse_s_per_it():
    sample = parse_line("steps: 50%| | 10/20 [00:05<00:05, 0.50s/it, loss=0.2]")

    assert sample is not None
    assert math.isclose(sample.it_per_s, 2.0)


def test_parse_nan_loss():
    sample = parse_line("steps: 50%| | 100/200 [00:10<00:10, 1.00it/s, loss=nan]")

    assert sample is not None
    assert sample.loss is not None and math.isnan(sample.loss)


def test_epoch_line_does_not_look_like_a_step():
    sample = parse_line("epoch 3/10")

    assert sample is not None
    assert sample.epoch == 3
    assert sample.step is None


def test_unrelated_line_is_ignored():
    assert parse_line("noscript: loading cached latents for 40 images") is None


def _series(start=1.0, step=-0.01, count=40):
    monitor = Monitor()
    for index in range(count):
        monitor.add(Sample(step=index, total=count, loss=start + step * index))
    return monitor


def test_trend_is_negative_when_loss_falls():
    trend = _series().trend(40)

    assert trend is not None
    assert trend[1] < 0


def test_flat_loss_warns():
    monitor = Monitor()
    for index in range(40):
        monitor.add(Sample(step=index, total=40, loss=0.5))

    assert any("flat" in warning for warning in monitor.warnings())


def test_rising_loss_warns():
    monitor = _series(start=0.4, step=0.01)

    assert any("rising" in warning for warning in monitor.warnings())


def test_divergence_is_reported():
    monitor = Monitor()
    monitor.add(Sample(step=1, loss=float("nan")))

    assert any("diverged" in warning for warning in monitor.warnings())


def test_status_line_mentions_step_and_speed():
    monitor = Monitor()
    monitor.add(parse_line(BAR))

    line = monitor.status_line()

    assert "step 320/2000" in line
    assert "it/s" in line


def test_format_duration():
    assert format_duration(45) == "45s"
    assert format_duration(790) == "13m10s"
    assert format_duration(3720) == "1h02m"
