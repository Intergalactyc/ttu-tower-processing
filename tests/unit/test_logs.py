import json
import logging

from ttu_tower.logs import CONTEXT_FIELDS, JsonLinesFormatter, get_logger


def test_json_lines_formatter_produces_valid_json_with_expected_keys():
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    line = JsonLinesFormatter().format(record)
    obj = json.loads(line)
    assert obj["msg"] == "hello world"
    assert obj["levelname"] == "INFO"
    assert obj["name"] == "test"


def test_context_adapter_injects_context_fields(caplog):
    logger = get_logger("test.context", stage="primary", boom=4)
    with caplog.at_level(logging.INFO, logger="test.context"):
        logger.info("unit done")
    record = caplog.records[0]
    assert record.stage == "primary"
    assert record.boom == 4
    assert record.unit is None
    for field in CONTEXT_FIELDS:
        assert hasattr(record, field)
