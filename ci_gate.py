#!/usr/bin/env python3
"""CI gate for the Chorus continuous loop.

Exit 0 = pass. Any other exit code = the calling workflow must discard the
phase's changes (git reset --hard).

Layers
------
1. Cargo compile check: `cargo check --all-targets` — every crate/target
   must compile (the fast "syntax" layer for Rust).
2. Full test suite: `cargo test` — the unit tests embedded in
   src/{config_tests,llm,server,workflow,test_migration}.rs must pass.

--syntax-only runs layer 1 only (used by the shutdown trap).

The gate needs the Rust toolchain on PATH (or in ~/.cargo/bin). It never
touches the network beyond what cargo itself needs to fetch dependencies.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[gate] FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"[gate] ok: {msg}")


def find_cargo() -> str | None:
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    fallback = pathlib.Path.home() / ".cargo" / "bin" / "cargo"
    if fallback.exists():
        return str(fallback)
    return None


def run_cargo(args: list[str], timeout: int) -> tuple[int, str]:
    cargo = find_cargo()
    if cargo is None:
        return 127, "cargo not found on PATH or in ~/.cargo/bin"
    env = dict(os.environ)
    cargo_home = pathlib.Path.home() / ".cargo" / "bin"
    if cargo_home.exists():
        env["PATH"] = f"{cargo_home}:{env.get('PATH', '')}"
    try:
        r = subprocess.run(
            [cargo] + args, cwd=ROOT, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, f"cargo {' '.join(args)} timed out after {timeout}s"
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out


def tail_of(text: str, n: int = 15) -> list[str]:
    lines = text.strip().splitlines()
    return lines[-n:]


def main() -> int:
    syntax_only = "--syntax-only" in sys.argv

    if not (ROOT / "Cargo.toml").exists():
        fail("Cargo.toml missing at repo root — cannot gate a Rust project without it")
    else:
        # Layer 1: compile check (all targets: bins, benches, tests).
        ec, out = run_cargo(["check", "--all-targets", "--quiet"], timeout=1200)
        if ec == 127:
            fail(out)
        elif ec == 124:
            fail("cargo check timed out")
        elif ec != 0:
            fail(f"cargo check failed (exit={ec})")
            for line in tail_of(out):
                print(f"    {line}")
        else:
            ok("cargo check --all-targets: clean compile")

        # Layer 2: full unit-test suite.
        if not syntax_only and not FAILURES:
            ec, out = run_cargo(["test", "--quiet"], timeout=1800)
            if ec == 127:
                fail(out)
            elif ec == 124:
                fail("cargo test timed out")
            elif ec != 0:
                fail(f"cargo test failed (exit={ec})")
                for line in tail_of(out):
                    print(f"    {line}")
            else:
                summary = tail_of(out, 3)
                ok("cargo test: " + (" | ".join(summary) if summary else "passed"))

    if FAILURES:
        print(f"\n[gate] REJECTED — {len(FAILURES)} failure(s)")
        return 1
    print("\n[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
