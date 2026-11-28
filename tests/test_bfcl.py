import json

import pytest
from conftest import ROOT

from headroom.bfcl.checker import check_call, standardize
from headroom.bfcl.convert import PYTHON_NOTE, to_tool
from headroom.bfcl.run import BfclRecord, load_bfcl, load_items, report, timing_ok
from headroom.schemas.chat import ToolCall
from headroom.schemas.common import unavailable

FN = {
    "name": "weather.get_forecast",
    "description": "Forecast.",
    "parameters": {
        "type": "dict",
        "properties": {
            "city": {"type": "string", "description": "City."},
            "days": {"type": "integer", "description": "Days."},
            "threshold": {"type": "float", "description": "Rain threshold."},
            "metric": {"type": "boolean", "description": "Metric units."},
            "hours": {"type": "array", "items": {"type": "integer"}, "description": "Hours."},
            "coords": {"type": "tuple", "items": {"type": "float"}, "description": "Lat, lon."},
            "options": {"type": "dict", "properties": {"lang": {"type": "string"}}, "description": "Options."},
        },
        "required": ["city", "days"],
    },
}
TRUTH = [{"weather.get_forecast": {
    "city": ["New York, NY", "NYC"], "days": [3], "threshold": [0.5, ""], "metric": [True, ""],
    "hours": [[9, 12], ""], "coords": [""], "options": [{"lang": ["en", ""]}, ""],
}}]


def call(**args) -> tuple[ToolCall, ...]:
    return (ToolCall(name="weather_get_forecast", arguments=args),)


def verdict(calls):
    return check_call(calls, [FN], TRUTH)


def test_a_correct_call_passes_with_bfcl_string_standardisation():
    assert verdict(call(city="new york ny", days=3)).valid
    assert verdict(call(city="NYC", days=3, threshold=0.5, metric=True, hours=[9, 12], options={"lang": "EN"})).valid


@pytest.mark.parametrize("calls, error_type", [
    ((), "no_function_call"),
    (call(city="NYC", days=3) * 2, "wrong_count"),
    ((ToolCall(name="weather.get_forecast", arguments={"city": "NYC", "days": 3}),), "wrong_function"),
    (call(city="NYC"), "missing_required"),
    (call(city="NYC", days=3, units="c"), "unexpected_param"),
    (call(city="NYC", days="3"), "type_error"),
    (call(city="NYC", days=True), "type_error"),
    (call(city="Boston", days=3), "value_error"),
    (call(city="NYC", days=3, hours=[12, 9]), "value_error"),
    (call(city="NYC", days=3, options={"lang": "fr"}), "value_error"),
    ((ToolCall(name="weather_get_forecast", arguments={}, parse_error="not JSON"),), "unparseable_arguments"),
])
def test_each_bfcl_rule_fails_for_its_reason(calls, error_type):
    v = verdict(calls)
    assert not v.valid and v.error_type == error_type, v


def test_an_integer_is_accepted_for_a_float_but_not_the_reverse():
    assert verdict(call(city="NYC", days=3, threshold=0.5)).valid
    truth = [{"weather.get_forecast": {**TRUTH[0]["weather.get_forecast"], "threshold": [1.0]}}]
    assert check_call(call(city="NYC", days=3, threshold=1), [FN], truth).valid
    assert check_call(call(city="NYC", days=3.0, threshold=1), [FN], truth).error_type == "type_error"


def test_a_list_of_dicts_is_checked_dict_by_dict():
    fn = {"name": "f", "description": "", "parameters": {"type": "dict", "required": ["data"], "properties": {
        "data": {"type": "array", "items": {"type": "dict"}}}}}
    truth = [{"f": {"data": [[{"name": ["Li Lei"], "age": [18]}, {"name": ["Li Li"], "age": [21]}]]}}]
    ok = ({"age": 18, "name": "li lei"}, {"name": "Li Li", "age": 21})
    assert check_call((ToolCall(name="f", arguments={"data": list(ok)}),), [fn], truth).valid
    swapped = [ok[1], ok[0]]
    assert check_call((ToolCall(name="f", arguments={"data": swapped}),), [fn], truth).error_type == "value_error"


def test_leaving_out_a_parameter_needs_a_default():
    truth = [{"weather.get_forecast": {**TRUTH[0]["weather.get_forecast"], "metric": [True]}}]
    assert check_call(call(city="NYC", days=3), [FN], truth).error_type == "missing_optional"


def test_standardize_matches_bfcl():
    assert standardize("New York, NY") == standardize("new_york-ny") == "newyorkny"
    assert standardize("it's") == 'it"s'


def test_tool_conversion_matches_bfcl_fc_preprocessing():
    tool = to_tool(FN)
    assert tool.name == "weather_get_forecast" and tool.description == "Forecast." + PYTHON_NOTE
    p = tool.parameters
    assert p["type"] == "object" and p["properties"]["coords"]["type"] == "array"
    assert p["properties"]["coords"]["items"]["type"] == "number"
    assert p["properties"]["threshold"]["description"].endswith("This is a float type value.")
    assert p["properties"]["options"]["type"] == "object"
    assert FN["parameters"]["type"] == "dict"  # the source is not modified


def test_the_pinned_configs_load_and_pair():
    a, b = load_bfcl(ROOT / "experiments" / "bfcl_v3.toml"), load_bfcl(ROOT / "experiments" / "bfcl_v3_llamacpp.toml")
    assert a.categories == b.categories and a.data_revision == b.data_revision and a.model_id == b.model_id
    assert a.bfcl_id != b.bfcl_id and a.base_experiment != b.base_experiment


def test_items_are_refused_when_a_file_does_not_match_its_pin(tmp_path):
    cfg = load_bfcl(ROOT / "experiments" / "bfcl_v3.toml").model_copy(update={"data_dir": "d"})
    for c in cfg.categories:
        for rel in (c.questions, c.answers):
            (tmp_path / "d" / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "d" / rel).write_text(json.dumps({"id": "x"}) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        load_items(cfg, tmp_path)


def test_report_counts_accuracy_per_category():
    def rec(i, cat, valid, et=None):
        return BfclRecord(
            bfcl_id="b", session_id="s", item_id=i, category=cat, model_id="m", started_at="2026-11-28T00:00:00Z",
            latency_s=1.0, load_s=unavailable("x"), cold_start=None, prompt_tokens=unavailable("x"),
            completion_tokens=unavailable("x"), done_reason="stop", content="", tool_calls=(), valid=valid,
            error_type=et, error_detail=None,
        )
    text = report([rec("1", "simple", True), rec("2", "simple", False, "value_error"), rec("3", "multiple", True)])
    assert "simple             2      1     50.0%" in text and "value_error 1" in text
    assert "all                3      2     66.7%" in text


def test_items_the_machine_slept_through_are_left_out_of_timing_only():
    base = dict(bfcl_id="b", session_id="s", category="simple", model_id="m", started_at="2026-11-28T00:00:00Z",
                load_s=unavailable("x"), cold_start=None, prompt_tokens=unavailable("x"),
                completion_tokens=unavailable("x"), done_reason="stop", content="", tool_calls=(),
                valid=True, error_type=None, error_detail=None)
    awake = BfclRecord(item_id="1", latency_s=2.0, suspended_s=0.0, **base)
    slept = BfclRecord(item_id="2", latency_s=3000.0, suspended_s=2995.0, **base)
    old_hung = BfclRecord(item_id="3", latency_s=3000.0, **base)  # logged before sleep was measured
    assert timing_ok(awake) and not timing_ok(slept) and not timing_ok(old_hung)
    text = report([awake, slept, old_hung])
    assert "simple             3      3    100.0%      2.00" in text and "2 item(s) left out" in text
