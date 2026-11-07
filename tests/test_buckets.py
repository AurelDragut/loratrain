from loratrain.buckets import bucket_for, upscale_factor


def test_square_image_keeps_the_target_resolution():
    assert bucket_for(1024, 1024) == (1024, 1024)


def test_portrait_is_clamped_to_the_bucket_range():
    bucket_w, bucket_h = bucket_for(832, 1216)
    assert bucket_w >= 1024
    assert bucket_h <= 1536
    assert bucket_h > bucket_w


def test_bucket_sides_are_multiples_of_the_step():
    bucket_w, bucket_h = bucket_for(1000, 1500)
    assert bucket_w % 64 == 0
    assert bucket_h % 64 == 0


def test_upscale_factor_is_above_one_when_the_bucket_is_larger():
    assert upscale_factor(512, 512, (1024, 1024)) > 1.0


def test_upscale_factor_is_about_one_when_it_fits():
    assert abs(upscale_factor(1024, 1024, (1024, 1024)) - 1.0) < 1e-9
