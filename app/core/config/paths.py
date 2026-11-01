"""Filesystem paths and the persisted app-state location."""

from pathlib import Path

from pydantic import Field, model_validator

from .base import ConfigGroup


class PathsSettings(ConfigGroup):
    data_dir: Path = Path("data")
    docs_dir: Path = Field(default=Path("data/docs"))
    index_dir: Path = Field(default=Path("data/index"))
    state_path: Path = Path("data/app_state.json")

    @model_validator(mode="after")
    def _derive_subdirectories(self) -> "PathsSettings":
        """Follow a custom ``data_dir`` for docs/index unless they were set too.

        Only the untouched defaults are derived, so an explicit
        ``DOCINTEL_PATHS__DOCS_DIR`` still wins.
        """
        if self.data_dir != Path("data"):
            if self.docs_dir == Path("data/docs"):
                self.docs_dir = self.data_dir / "docs"
            if self.index_dir == Path("data/index"):
                self.index_dir = self.data_dir / "index"
        return self
