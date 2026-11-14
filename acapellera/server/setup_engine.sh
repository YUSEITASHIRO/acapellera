#!/usr/bin/env bash
# Rust (user install, no sudo) and the CHIMERA CLI of this repository (target/release/chimera), which renders the
# twelve voices on all the machine's cores. Run from anywhere inside the repository's checkout.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x ~/.cargo/bin/cargo ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source ~/.cargo/env
rustc -V
cargo build --release -p chimera-cli
ls -la target/release/chimera
echo ENGINE_DONE
