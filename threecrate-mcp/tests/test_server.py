"""The MCP layer: tools are registered with usable schemas and calls reach ops."""

import asyncio
import json

import pytest

from mcp.server.mcpserver.exceptions import ToolError

from threecrate_mcp import server

EXPECTED_TOOLS = {
    "inspect_point_cloud", "inspect_mesh", "convert", "downsample", "remove_outliers",
    "estimate_normals", "register", "transform", "segment_plane", "extract_clusters",
    "reconstruct", "simplify_mesh", "smooth_mesh", "render", "repair_for_print",
}


def test_all_tools_are_registered_with_descriptions():
    tools = asyncio.run(server.mcp.list_tools())
    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name
        assert tool.input_schema["type"] == "object"


def test_method_choices_are_enums_in_the_schema():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    methods = tools["register"].input_schema["properties"]["method"]
    assert set(methods["enum"]) == {"icp", "point_to_plane", "gicp", "global"}


def test_call_tool_end_to_end(ws, monkeypatch):
    monkeypatch.setenv("THREECRATE_MCP_ROOT", str(ws.root))
    server._ws.cache_clear()
    try:
        result = asyncio.run(server.mcp.call_tool("inspect_point_cloud", {"path": "scan.ply"}))
    finally:
        server._ws.cache_clear()
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["points"] == 40 * 40 + 20


def test_tool_errors_keep_their_message(ws, monkeypatch):
    """The agent must see why a call failed, not a bare "Error executing tool"."""
    monkeypatch.setenv("THREECRATE_MCP_ROOT", str(ws.root))
    server._ws.cache_clear()
    try:
        with pytest.raises(ToolError, match="outside the workspace"):
            asyncio.run(server.mcp.call_tool("inspect_point_cloud", {"path": "../elsewhere.ply"}))
        with pytest.raises(ToolError, match="already exists"):
            for _ in range(2):
                asyncio.run(server.mcp.call_tool("downsample", {"path": "scan.ply", "voxel_size": 0.1}))
    finally:
        server._ws.cache_clear()


def test_bad_mesh_is_an_error_not_a_crash(ws, monkeypatch):
    """A face index past the vertex count used to panic inside Rust and kill the server."""
    (ws.root / "bad.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 99\n"
    )
    monkeypatch.setenv("THREECRATE_MCP_ROOT", str(ws.root))
    server._ws.cache_clear()
    try:
        with pytest.raises(ToolError, match="vertex index"):
            asyncio.run(server.mcp.call_tool("smooth_mesh", {"path": "bad.ply"}))
    finally:
        server._ws.cache_clear()


def test_non_exception_failures_reach_the_agent(monkeypatch):
    """A Rust panic arrives as a BaseException (pyo3 PanicException); it must not escape."""

    class FakePanic(BaseException):
        pass

    def boom(*args, **kwargs):
        raise FakePanic("rust panicked")

    monkeypatch.setattr(server.ops, "inspect_mesh", boom)
    with pytest.raises(ToolError, match="rust panicked"):
        asyncio.run(server.mcp.call_tool("inspect_mesh", {"path": "x.ply"}))


def test_render_returns_the_picture_to_the_agent(ws, monkeypatch):
    monkeypatch.setenv("THREECRATE_MCP_ROOT", str(ws.root))
    server._ws.cache_clear()
    try:
        result = asyncio.run(server.mcp.call_tool(
            "render", {"paths": ["sphere.ply"], "views": ["front", "top"], "size": 128}
        ))
    finally:
        server._ws.cache_clear()
    assert not result.is_error
    kinds = [block.type for block in result.content]
    assert kinds == ["text", "image"]
    assert json.loads(result.content[0].text)["rows"][0]["kind"] == "point_cloud"
    assert result.content[1].mime_type == "image/png"
