---
name: Python binding request
about: Request a Rust function to be exposed to the Python API
title: '[Python] Expose `function_name` to Python bindings'
labels: python-bindings, good first issue
assignees: ''

---

**Which Rust function(s) should be exposed?**
e.g. `threecrate_algorithms::icp_point_to_plane` from `threecrate-algorithms/src/registration.rs`

**Proposed Python API**
```python
import threecrate as tc

# How you'd like to call it:
result = tc.icp_point_to_plane(source, target, max_iterations=50)
```

**Why is this useful from Python?**
Brief description of the use case.

**Notes for the implementor**
Any tricky types, numpy conversion, or lifetime issues to be aware of. Leave blank if unsure.
