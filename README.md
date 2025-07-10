# glpk-rust

A Rust wrapper for the GNU Linear Programming Kit (GLPK) for solving Integer Linear Programming (ILP) problems.

## Features

- **Self-contained**: No need to install GLPK separately! The build system automatically downloads, compiles, and statically links GLPK 5.0.
- **Cross-platform**: Works on macOS, Linux, and Windows
- **Zero runtime dependencies**: Everything is statically linked
- **Safe Rust API**: High-level, memory-safe interface to GLPK

## License

This library is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

**Important**: Since this library statically links with GLPK (which is GPL-3.0 licensed), any software that uses this library must also be licensed under GPL-3.0 or a compatible license. This means:

- ✅ Your project can use this library if it's open source under GPL-3.0
- ✅ You can use it in other GPL-compatible open source projects
- ❌ You cannot use this library in proprietary/closed-source software
- ❌ You cannot use this library in projects with incompatible licenses (MIT, Apache, etc.)

If you need to use GLPK in a non-GPL project, you'll need to either:
1. Purchase a commercial license from the GLPK authors, or
2. Use a different solver with a more permissive license

## Prerequisites

Only standard build tools are required:
- **macOS**: Xcode command line tools (`xcode-select --install`)
- **Linux**: `build-essential` package (Ubuntu/Debian) or `gcc make` (RHEL/CentOS)
- **Windows**: Visual Studio Build Tools or MinGW

The build system will automatically:
1. Download GLPK 5.0 source code
2. Configure and compile it with the correct flags
3. Statically link it into your Rust binary

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
glpk-rust = "0.1.0"
```

## Example

```rust
use glpk_rust::*;
use std::collections::HashMap;

// Create variables
let variables = vec![
    Variable { id: "x1", bound: (0, 5) },
    Variable { id: "x2", bound: (0, 3) },
];

// Define constraints
let mut polytope = SparseLEIntegerPolyhedron {
    A: IntegerSparseMatrix {
        rows: vec![0, 1],
        cols: vec![0, 1], 
        vals: vec![1, 1],
        shape: Shape { nrows: 2, ncols: 2 },
    },
    b: vec![(0, 10), (0, 8)],
    variables,
    double_bound: true,
};

// Define objective
let mut objective = HashMap::new();
objective.insert("x1", 1.0);
objective.insert("x2", 1.0);

// Solve
let solutions = solve_ilps(&mut polytope, vec![objective], false, false);
println!("{:?}", solutions);
```