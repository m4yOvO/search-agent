"""Structured JSON logging for API, graph routing, tools, and memory events."""

from __future__ import annotations

import logging.config


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "format": (
                        "%(asctime)s %(levelname)s %(name)s %(message)s %(event)s "
                        "%(request_id)s %(conversation_id)s %(match_type)s "
                        "%(cache_status)s %(result_id)s %(write_operation)s"
                    ),
                    "rename_fields": {
                        "asctime": "timestamp",
                        "levelname": "level",
                        "name": "logger",
                    },
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["default"], "level": level.upper()},
            "loggers": {
                "uvicorn.access": {
                    "handlers": ["default"],
                    "level": level.upper(),
                    "propagate": False,
                }
            },
        }
    )
