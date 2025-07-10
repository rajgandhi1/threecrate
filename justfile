# threecrate Development Commands
# Install just: cargo install just
# Usage: just <command>

# List all available commands
default:
    @just --list

# Build all crates in the workspace
build:
    cargo build --workspace

# Build all crates in release mode
build-release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run tests with output
test-verbose:
    cargo test --workspace -- --nocapture

# Run tests for a specific crate (e.g., just test-crate threecrate-core)
test-crate crate:
    cargo test -p {{crate}}

# Check all crates (fast compilation check)
check:
    cargo check --workspace

# Format all code
fmt:
    cargo fmt --all

# Check if code is formatted
fmt-check:
    cargo fmt --all -- --check

# Run clippy linter
clippy:
    cargo clippy --workspace --all-targets --all-features

# Run clippy with automatic fixes
clippy-fix:
    cargo clippy --workspace --all-targets --all-features --fix

# Run all quality checks (format, clippy, test)
ci: fmt-check clippy test

# Prepare for commit (format, clippy, test)
ready: fmt clippy test

# Clean build artifacts
clean:
    cargo clean

# Run the basic usage example
example:
    cargo run --bin basic_usage

# Run a specific example (e.g., just run-example basic_usage)
run-example name:
    cargo run --bin {{name}}

# Build documentation
doc:
    cargo doc --workspace --no-deps

# Build and open documentation
doc-open:
    cargo doc --workspace --no-deps --open

# Check for unused dependencies
machete:
    cargo machete

# Update dependencies
update:
    cargo update

# Install development tools
install-tools:
    cargo install cargo-machete just

# Benchmark (when benchmarks are implemented)
bench:
    cargo bench --workspace

# Profile a specific example with perf (Linux only)
profile example:
    cargo build --release --bin {{example}}
    perf record --call-graph=dwarf ./target/release/{{example}}
    perf report

# Create a new algorithm implementation template
new-algorithm name crate:
    @echo "Creating template for {{name}} in {{crate}}"
    @echo "Remember to:"
    @echo "1. Add function signature to lib.rs"
    @echo "2. Implement the algorithm with proper documentation"
    @echo "3. Add comprehensive tests"
    @echo "4. Update examples if needed"

# Quick development cycle: format, check, test
dev: fmt check test

# Full quality check before PR
pr-ready: clean fmt clippy test doc

# Help for contributors
help:
    @echo "Common development workflows:"
    @echo ""
    @echo "  just dev          - Quick development cycle (format, check, test)"
    @echo "  just pr-ready     - Full quality check before PR"
    @echo "  just ci           - Run CI checks locally"
    @echo "  just example      - Run basic usage example"
    @echo ""
    @echo "For implementing new algorithms:"
    @echo "  just new-algorithm <name> <crate> - Show implementation checklist"
    @echo ""
    @echo "See 'just --list' for all available commands" 