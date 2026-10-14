from pathlib import Path

import pytest

from threecrate_mcp.workspace import Workspace, WorkspaceError

ALLOWED = {".ply", ".pcd"}


@pytest.fixture
def root(tmp_path: Path) -> Workspace:
    (tmp_path / "in.ply").write_bytes(b"x")
    (tmp_path / "in.las").write_bytes(b"x")
    return Workspace(root=tmp_path.resolve())


def test_relative_input_resolves_inside_root(root: Workspace):
    assert root.input_path("in.ply") == root.root / "in.ply"


def test_paths_outside_root_are_refused(root: Workspace):
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        root.input_path("../in.ply")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        root.output_path("../out.ply", inputs=[root.root / "in.ply"], default_suffix="x", allowed=ALLOWED)


def test_missing_input_is_refused(root: Workspace):
    with pytest.raises(WorkspaceError, match="does not exist"):
        root.input_path("nope.ply")


def test_default_output_goes_next_to_input(root: Workspace):
    out = root.output_path(None, inputs=[root.root / "in.ply"], default_suffix="clean", allowed=ALLOWED)
    assert out == root.root / "in.clean.ply"


def test_default_output_falls_back_to_ply_for_read_only_formats(root: Workspace):
    out = root.output_path(None, inputs=[root.root / "in.las"], default_suffix="clean", allowed=ALLOWED)
    assert out.name == "in.clean.ply"


def test_never_overwrites_an_input(root: Workspace):
    with pytest.raises(WorkspaceError, match="overwrite the input"):
        root.output_path("in.ply", inputs=[root.root / "in.ply"], default_suffix="x", allowed=ALLOWED, overwrite=True)


def test_existing_output_needs_overwrite(root: Workspace):
    (root.root / "out.ply").write_bytes(b"x")
    with pytest.raises(WorkspaceError, match="already exists"):
        root.output_path("out.ply", inputs=[root.root / "in.ply"], default_suffix="x", allowed=ALLOWED)
    assert root.output_path("out.ply", inputs=[root.root / "in.ply"], default_suffix="x",
                            allowed=ALLOWED, overwrite=True).name == "out.ply"


def test_unsupported_output_format_is_refused(root: Workspace):
    with pytest.raises(WorkspaceError, match="use one of"):
        root.output_path("out.las", inputs=[root.root / "in.ply"], default_suffix="x", allowed=ALLOWED)


def test_point_limit(root: Workspace):
    small = Workspace(root=root.root, max_points=10)
    with pytest.raises(WorkspaceError, match="above the limit"):
        small.check_size(11, root.root / "in.ply")
