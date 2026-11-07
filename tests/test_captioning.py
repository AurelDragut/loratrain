from loratrain.captioning import assemble


def test_sentence_style_starts_with_the_trigger():
    assert assemble("tok", "a woman standing", ["indoor"], "sentence", {}) == "tok, a woman standing"


def test_tags_style_uses_the_tags():
    assert assemble("tok", "ignored", ["blonde hair", "outdoor"], "tags", {}) == (
        "tok, blonde hair, outdoor"
    )


def test_tags_style_falls_back_to_the_sentence():
    assert assemble("tok", "a woman standing", [], "tags", {}) == "tok, a woman standing"


def test_deviations_are_appended_explicitly():
    text = assemble("tok", "portrait", [], "sentence", {"hair_color": "blonde"})

    assert text == "tok, portrait, hair_color: blonde"


def test_no_trigger_and_no_text_is_empty():
    assert assemble("", "", [], "sentence", {}) == ""
