---
name: Bug report
about: Something is broken or producing wrong results
title: '[Bug] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what is wrong.

**To Reproduce**
Minimal code snippet that reproduces the issue:

```rust
// Rust
use threecrate::prelude::*;

fn main() {
    // ...
}
```

```python
# Python
import threecrate as tc
# ...
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened (include any error messages or panics in full).

**Environment**
- OS: [e.g. macOS 14, Ubuntu 22.04, Windows 11]
- Rust version: [output of `rustc --version`]
- threecrate version: [e.g. 0.7.1, or git SHA]
- Python version (if using Python bindings): [e.g. 3.11]
- GPU (if using `threecrate-gpu`): [e.g. Apple M3, NVIDIA RTX 4090]

**Additional context**
Any other relevant information (input file format, point cloud size, etc.).
