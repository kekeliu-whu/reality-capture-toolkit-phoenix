import base64
import builtins
import datetime
import logging
import os
import sys

from cryptography.hazmat.primitives import hashes, padding, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqzudEYdMdilhTiONl10Z
+uliDOczfjspRZFgSt4ZCJe6+AhnE/GBLXcaB1Rk/sQtKmOPoxIrbZIVhRqtT1um
02orUTGUdbdO9mxALWA72e+wzI5y7VPRxPc6bgeJ57voMyATFxyH+DTjJ8pXTLKf
9I3UNgyG6pXO6N/jJdIQsi8zR6pM7RVAZ+UHMho23FI3A27imqdRYMH4P/aAmnsK
yMdWpe8l0Idb5SpKhADqgoR+fKqOOg/jDRLKXv8z8VbLuRbi7uC/CuZi3GPTOodo
yHgzaKPwrDm08mSnvkids3Dlq+pAoCh0abbG/RGUg+j5QTm+apvtNaN8nxYnunDg
lwIDAQAB
-----END PUBLIC KEY-----"""

LOG_MODE_ENV = "MIGRATION_PY_LOG_MODE"
_PRINT_PATCHED = False
_ORIGINAL_PRINT = builtins.print


def _normalize_message(text: str) -> str:
    text = text.strip()
    for prefix in ("[OK] ", "[WARN] ", "[ERROR] ", "[INFO] "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _infer_level(text: str) -> int:
    normalized = text.lstrip()
    if normalized.startswith("[ERROR]") or normalized.startswith("[错误]"):
        return logging.ERROR
    if normalized.startswith("[WARN]") or normalized.startswith("[警告]"):
        return logging.WARNING
    if normalized.startswith("[INFO]"):
        return logging.INFO
    return logging.INFO


class SpdlogLikeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        timestamp = datetime.datetime.fromtimestamp(record.created)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{timestamp.microsecond // 1000:03d}"

    def format(self, record):
        message = _normalize_message(record.getMessage())
        level = record.levelname.lower()
        return f"[{self.formatTime(record)}] [{level}] {message}"


class ColorSpdlogFormatter(SpdlogLikeFormatter):
    LEVEL_COLORS = {
        logging.DEBUG: "\033[37m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        message = _normalize_message(record.getMessage())
        level = record.levelname.lower()
        color = self.LEVEL_COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        return f"[{self.formatTime(record)}] [{color}{level}{reset}] {message}"


class EncryptedStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
        self._header_written = False
        self._public_key = serialization.load_pem_public_key(PUBLIC_KEY_PEM)
        self._aes_key = os.urandom(32)
        self._encrypted_key_header = base64.b64encode(
            self._public_key.encrypt(
                self._aes_key,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA1()),
                    algorithm=hashes.SHA1(),
                    label=None,
                ),
            )
        ).decode("ascii")

    def emit(self, record):
        try:
            if not self._header_written:
                self.stream.write(self._encrypted_key_header + "\n")
                self._header_written = True

            plain_text = self.format(record).encode("utf-8")
            iv = os.urandom(16)
            padder = padding.PKCS7(128).padder()
            padded = padder.update(plain_text) + padder.finalize()
            cipher = Cipher(algorithms.AES(self._aes_key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(padded) + encryptor.finalize()
            encoded = base64.b64encode(iv + encrypted).decode("ascii")
            self.stream.write(encoded + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def _resolve_log_mode() -> str:
    mode = os.getenv(LOG_MODE_ENV, "encrypted").strip().lower()
    if mode in {"console", "plain", "debug"}:
        return "console"
    if mode in {"encrypted", "encrypt", "release"}:
        return "encrypted"
    return "encrypted"


def install_print_hook(logger: logging.Logger):
    global _PRINT_PATCHED
    if _PRINT_PATCHED:
        return

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file not in (None, sys.stdout, sys.stderr):
            _ORIGINAL_PRINT(*args, sep=sep, end=end, file=file, flush=flush)
            return

        text = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            text += end

        lines = text.splitlines() or [""]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            logger.log(_infer_level(stripped), stripped)

        if flush:
            for handler in logger.handlers:
                handler.flush()

    builtins.print = _print
    _PRINT_PATCHED = True


def init_spdlog_like_logger(name: str = "migration.scripts") -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_spdlog_compat_initialized", False):
        return logger

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if _resolve_log_mode() == "console":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorSpdlogFormatter())
    else:
        handler = EncryptedStreamHandler(sys.stdout)
        handler.setFormatter(SpdlogLikeFormatter())

    logger.addHandler(handler)
    logger._spdlog_compat_initialized = True
    install_print_hook(logger)
    return logger