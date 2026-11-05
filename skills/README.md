# threecrate plugin

Lets Claude work on 3D scans and meshes in plain words: "make my scan printable", "clean up this scan", "align these two scans", "how tall is the pole?".

The plugin is one skill, [`threecrate`](threecrate/SKILL.md), with a guide for each task:

- [Clean a scan](threecrate/guides/clean-scan.md): stray points, downsampling, the floor, separate objects, normals
- [Align scans](threecrate/guides/align-scans.md): feature matching, then GICP or ICP
- [Make it printable](threecrate/guides/make-printable.md): holes, a missing base, flipped faces, loose bits, millimetres for the slicer
- [Measure](threecrate/guides/measure.md): size, area, volume, heights above the ground

Claude checks each result by rendering a before and after picture and looking at it before it reports.

## What it runs

The skill drives the open-source [threecrate](https://github.com/rajgandhi1/threecrate) Python library from short Python scripts:

- It needs Python 3.8 or newer on your PATH.
- The first time it runs, it creates a virtual environment (`.venv`) in your working folder and installs `threecrate` and `numpy` from PyPI into it. It never installs into your system Python.
- `scripts/inspect_cloud.py` reads a point cloud and prints its size and point spacing.
- Everything runs on your machine. The plugin sends no data anywhere; the only network access is the PyPI install.
- It writes results to new files next to your input and leaves the input untouched.

## Formats

Point clouds: PLY, PCD, XYZ, CSV, LAS/LAZ, E57. Meshes: PLY, OBJ, STL.

## License

MIT OR Apache-2.0, the same as the threecrate library.
