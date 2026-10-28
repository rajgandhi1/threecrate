# Demo: make a scan printable

Two short clips. Record them in any agent that has the threecrate skill or the
threecrate-mcp server installed (see [INSTALL.md](../INSTALL.md)). Needs threecrate 0.12.0 or newer.

## Clip 1: phone scan to printable model

Start in a folder that holds `table_scan.ply` from this directory, then type:

> table_scan.ply is a phone scan of a small vase on my table. Turn it into something I can 3D print.

What you should see:

1. The agent inspects the scan: about 30,000 points, in metres.
2. It renders a preview. The table and loose dots are visible around the vase.
3. It runs `repair_for_print` with the floor removed: 58 stray points and the table (about 9,000 points) go, then Poisson builds a closed surface.
4. It renders the result and saves `table_scan.printable.stl`, about 9.3 x 9.3 x 12.3 cm and watertight.

## Clip 2: a real museum scan

The Abraham Lincoln life mask by Leonard Volk, 1860, from the Smithsonian National Portrait Gallery (CC0).
Download the 11.7 MB OBJ zip and unzip it:

```bash
curl -LO "https://3d-api.si.edu/content/document/3d_package:2b4a081a-9ea1-4b0c-b1c3-6f5389da3244/resources/npg_71_24-100k-obj.zip"
```

Then type:

> Is npg_71_24_mesh-100k.obj ready to 3D print? It is Y-up. If not, fix it.

What you should see:

1. The health check says no: 4 small holes and a loose piece.
2. The repair removes the piece and closes the holes in under a second.
3. The result is watertight, about 203 x 235 x 161 mm, and the render looks the same as the original, because only the problems changed.
