# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: meng
#     language: python
#     name: python3
# ---

# %% [markdown]
# # My Logging

# %%
"""
A custom logging module that formats log messages in Markdown style.
"""
import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any
from sparam_surrogate.paths import LOGGING_DIR, LOG_CFG_PATH


# %%
class MarkdownLogger(logging.Logger):
    """
    Markdown Logger, formats messages as Markdown with special methods.
    """

    def heading1(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Markdown heading level 1."""
        self._log(logging.INFO, msg, args, extra={"md_type": "h1"}, **kwargs)

    def heading2(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Markdown heading level 2."""
        self._log(logging.INFO, msg, args, extra={"md_type": "h2"}, **kwargs)

    def heading3(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Markdown heading level 3."""
        self._log(logging.INFO, msg, args, extra={"md_type": "h3"}, **kwargs)

    def bullet(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Markdown bullet point."""
        self._log(logging.INFO, msg, args, extra={"md_type": "bullet"}, **kwargs)

    def code_block(self, code: str, language: str = "") -> None:
        """Markdown code block."""
        msg = f"```{language}\n{code}\n```"
        self._log(logging.INFO, msg, (), extra={"md_type": "raw"})


class MarkdownFormatter(logging.Formatter):
    """Markdown Formatter, converts log records to Markdown-styled."""

    level_icons = {
        logging.DEBUG: "🔍DEBUG",
        logging.INFO: "ℹ️ INFO",
        logging.WARNING: "⚠️ WARN",
        logging.ERROR: "❌ERROR",
        logging.CRITICAL: "🚨CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record based on its md_type for Markdown output.

        - h1, h2, h3: Markdown headings
        - bullet: Markdown bullet point
        - raw: Output message as-is without timestamp or icon
        - default: Prepend log level icon and timestamp to the message
        """
        message = record.getMessage()
        md_type = getattr(record, "md_type", "log")

        if md_type == "h1":
            return f"\n# {message}\n"

        if md_type == "h2":
            return f"\n## {message}\n"

        if md_type == "h3":
            return f"\n### {message}\n"

        if md_type == "bullet":
            return f"- {message}"

        if md_type == "raw":
            return message

        icon = self.level_icons.get(record.levelno, "")
        timestamp = datetime.fromtimestamp(record.created)
        timestamp = timestamp.strftime("%m-%d %H:%M:%S.%f")[:-3]

        return f"{icon} {timestamp} {record.name} — {message}  "


def set_logging_cfg(
    log_dir: Path = LOGGING_DIR, log_cfg_path: Path = LOG_CFG_PATH
) -> None:
    """
    Setup logging configuration from a JSON file

    - class MarkdownLogger is uesed as the logger class
    - log_dir: Directory where log files will be saved, default is LOGGING_DIR
    - log_cfg_path: Path to the logging configuration JSON file
    """

    # Only set up logging if it hasn't been configured yet
    if logging.root.handlers:
        return

    # Ensure the log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set the custom logger class before configuring logging
    logging.setLoggerClass(MarkdownLogger)

    # Load the logging configuration from the specified JSON file
    with open(log_cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Update the log file path in the configuration to include the log directory
    log_file = config["handlers"]["markdown_file"]["filename"]
    config["handlers"]["markdown_file"]["filename"] = str(log_dir / log_file)

    logging.config.dictConfig(config)


def get_md_logger(name: str = "sparam_surrogate") -> MarkdownLogger:
    """
    Get a MarkdownLogger instance by name.

    - name: Logger name, default is "sparam_surrogate"
    """
    return logging.getLogger(name)  # type: ignore


# %%
if __name__ == "__main__":
    set_logging_cfg()
    logger: MarkdownLogger = logging.getLogger("sparam_surrogate")  # type: ignore

    logger.heading1("Experiment 1: Random Forest Baseline")

    logger.heading2("Dataset")
    logger.info("Loaded parameter.csv")
    logger.info("Loaded 1200 Touchstone files")
    logger.warning("15 samples were skipped because the S-parameter file was missing")

    logger.heading2("Training")
    logger.debug("RandomForestRegressor parameters: n_estimators=200, max_depth=None")
    logger.info("Training completed")
    logger.error("Validation failed for fold 3")

    logger.heading2("Result")
    logger.bullet("Target: IL21 at 10 GHz")
    logger.bullet("Validation RMSE: 0.032")
    logger.code_block("model.fit(X_train, y_train)", language="python")
