from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


PACKAGE_NAME = "f1-tire-energy-strategy"
DEFAULT_VERSION = "0.1.0"


def package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return DEFAULT_VERSION


@dataclass(frozen=True)
class BuildInfo:
    version: str
    build_sha: str
    build_date: str


APP_VERSION = package_version()


def build_info() -> BuildInfo:
    return BuildInfo(
        version=APP_VERSION,
        build_sha=os.getenv("F1_BUILD_SHA", "unknown"),
        build_date=os.getenv("F1_BUILD_DATE", "unknown"),
    )
