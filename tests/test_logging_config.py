from loguru import logger

import src.logging_config as lc


def test_configure_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "_CONFIGURED", False)
    logger.remove()
    try:
        for i in range(5):
            lc.configure_logging(tmp_path / f"log{i}.log")
        # stderr + exactly one file sink, no matter how many calls
        assert len(logger._core.handlers) == 2
    finally:
        logger.remove()
