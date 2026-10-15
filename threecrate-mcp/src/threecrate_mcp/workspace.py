"""Where the server may read and write, and how output paths are chosen.

Every path an agent passes in is resolved against one root folder and must stay
inside it. Outputs never overwrite an input, and never overwrite an existing
file unless the caller asks for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT_ENV = "THREECRATE_MCP_ROOT"
MAX_POINTS_ENV = "THREECRATE_MCP_MAX_POINTS"
DEFAULT_MAX_POINTS = 50_000_000

POINT_CLOUD_WRITE_FORMATS = {".ply", ".pcd", ".xyz", ".csv"}
MESH_FORMATS = {".ply", ".obj"}


class WorkspaceError(ValueError):
    """A path or size the server refuses to work with. The message is shown to the agent."""


@dataclass(frozen=True)
class Workspace:
    root: Path
    max_points: int = DEFAULT_MAX_POINTS

    @classmethod
    def from_env(cls) -> "Workspace":
        root = Path(os.environ.get(ROOT_ENV) or Path.cwd()).resolve()
        if not root.is_dir():
            raise WorkspaceError(f"workspace root {root} is not a folder")
        max_points = int(os.environ.get(MAX_POINTS_ENV, DEFAULT_MAX_POINTS))
        return cls(root=root, max_points=max_points)

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise WorkspaceError(
                f"{path} is outside the workspace {self.root}. "
                f"Set {ROOT_ENV} to allow another folder."
            )
        return resolved

    def input_path(self, path: str) -> Path:
        """An existing file inside the workspace."""
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise WorkspaceError(f"{path} does not exist or is not a file")
        return resolved

    def output_dir(self, path: str) -> Path:
        """A folder inside the workspace, created if needed."""
        resolved = self._resolve(path)
        if resolved.exists() and not resolved.is_dir():
            raise WorkspaceError(f"{path} exists and is not a folder")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def output_path(
        self,
        path: Optional[str],
        *,
        inputs: list[Path],
        default_suffix: str,
        allowed: set[str],
        default_extension: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """Where to write a result.

        Without `path`, the result goes next to the first input, named
        `<input stem>.<default_suffix><extension>`. The extension must be in
        `allowed`. An input is never overwritten; an existing file only when
        `overwrite` is set.
        """
        if path is None:
            first = inputs[0]
            extension = default_extension or first.suffix.lower()
            if extension not in allowed:
                # e.g. a LAS input: write the result as PLY, which every writer supports
                extension = ".ply"
            resolved = first.with_name(f"{first.stem}.{default_suffix}{extension}")
        else:
            resolved = self._resolve(path)

        if resolved.suffix.lower() not in allowed:
            raise WorkspaceError(
                f"cannot write {resolved.name}: use one of {', '.join(sorted(allowed))}"
            )
        if any(resolved == item for item in inputs):
            raise WorkspaceError(f"refusing to overwrite the input file {resolved.name}")
        if resolved.exists() and not overwrite:
            raise WorkspaceError(
                f"{self.relative(resolved)} already exists; pass overwrite=true to replace it"
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def check_size(self, count: int, path: Path) -> None:
        if count > self.max_points:
            raise WorkspaceError(
                f"{path.name} has {count:,} points, above the limit of {self.max_points:,}. "
                f"Downsample it first or raise {MAX_POINTS_ENV}."
            )

    def relative(self, path: Path) -> str:
        """`path` relative to the workspace root, for messages and results."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)
