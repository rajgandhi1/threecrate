# Sample scans

Small generated scans with known answers, for demos, tests and agent evals.
They are made by `generate.py` (no third-party data) and share the repo's licence.

```bash
python samples/generate.py
```

| File | What it is | Known answers |
| --- | --- | --- |
| `table_scan.ply` | Phone-style scan of a vase on a table, in metres, z up. 30,060 points. | Vase about 9.4 cm wide and 12 cm tall. The table is 24 x 24 cm at z = 0 (9,000 points). 60 stray points. The vase base was never scanned. |
| `street_lidar.pcd` | LiDAR-style street patch, in metres, z up. 32,980 points. | Ground 20 x 20 m with a 2% slope along x. A wall at x = 10, 3 m tall. A pole at (-3, 2), 4 m tall. A car-sized box 4.2 x 1.8 x 1.4 m. 1 cm noise. |
| `broken_part.stl` | Rounded part in millimetres, broken like a bad export. | Body about 40 x 30 x 20 mm. 2 holes (top and one end), about 5% of faces flipped, and a small loose fragment floating above it, so the full bounding box is 32 mm tall. After `repair_for_print` it is watertight, about 17,500 mm³. |
