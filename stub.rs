//! Empty stub crate.
//!
//! HF-Hub-Ecosystem is a Python project; `pyproject.toml` holds the real
//! dependencies and `src/` holds the real code. The root `Cargo.toml`
//! exists only to satisfy the PMAT compliance checker, and it declares
//! `[lib] path = "stub.rs"`.
//!
//! That file was never committed. Because it was missing, every cargo
//! invocation that reads the manifest failed to even load it:
//!
//! ```text
//! $ cargo fmt --all -- --check
//! Error: file `.../stub.rs` does not exist
//! ```
//!
//! The PMAT pre-push hook skips non-Rust repos with `[ ! -f Cargo.toml ]`,
//! which the stub manifest defeats, so the hook ran `cargo fmt` here and
//! could never pass — leaving `git push --no-verify` as the only way to
//! push, on a repo where the hook was never meant to apply at all.
//!
//! Committing the declared file makes the manifest coherent and the gate
//! passable, so it measures formatting instead of erroring.
