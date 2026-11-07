from loratrain.attributes import deviations, vote


def test_vote_returns_the_majority_value():
    items = {f"i{index}": {"hair_color": "brown"} for index in range(9)}
    items["outlier"] = {"hair_color": "blonde"}

    assert vote(items, min_share=0.8, min_items=3) == {"hair_color": "brown"}


def test_vote_ignores_attributes_that_are_rare():
    items = {f"i{index}": {"hair_color": "brown"} for index in range(10)}
    items["i0"]["hat"] = "red"

    result = vote(items, min_share=0.8, min_items=3)

    assert "hair_color" in result
    assert "hat" not in result


def test_vote_is_case_insensitive():
    items = {f"i{index}": {"hair_color": "Brown"} for index in range(4)}

    assert vote(items, min_share=0.8, min_items=3) == {"hair_color": "brown"}


def test_deviations_reports_only_the_outlier():
    dominant = {"hair_color": "brown"}

    assert deviations({"hair_color": "Blonde"}, dominant) == {"hair_color": "Blonde"}
    assert deviations({"hair_color": "brown"}, dominant) == {}
