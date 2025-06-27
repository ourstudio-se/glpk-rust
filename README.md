# glpk-rust

A Rust wrapper for the GNU Linear Programming Kit (GLPK) for solving Integer Linear Programming (ILP) problems.

## Features

- **Self-contained**: No need to install GLPK separately! The build system automatically downloads, compiles, and statically links GLPK 5.0.
- **Cross-platform**: Works on macOS, Linux, and Windows
- **Zero runtime dependencies**: Everything is statically linked
- **Safe Rust API**: High-level, memory-safe interface to GLPK

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
    Variable { id: "x1".to_string(), bound: (0, 5) },
    Variable { id: "x2".to_string(), bound: (0, 3) },
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
objective.insert("x1".to_string(), 1.0);
objective.insert("x2".to_string(), 1.0);

// Solve
let solutions = solve_ilps(&mut polytope, vec![objective], false, false);
println!("{:?}", solutions);
```