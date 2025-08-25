#!/bin/bash
echo "Testing GLPK build..."
echo "Current directory: $(pwd)"
echo "Checking for GLPK libraries..."

if [ -f "/opt/homebrew/lib/libglpk.dylib" ]; then
    echo "✅ Found GLPK at /opt/homebrew/lib/libglpk.dylib"
elif [ -f "/usr/local/lib/libglpk.dylib" ]; then
    echo "✅ Found GLPK at /usr/local/lib/libglpk.dylib"
else
    echo "❌ GLPK library not found"
fi

echo "Running cargo build..."
cargo build
echo "Build completed with exit code: $?"