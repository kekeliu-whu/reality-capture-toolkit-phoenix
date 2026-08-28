# Research and acceptance scripts

These files preserve the module-alignment investigations. Scripts named `capture_*`,
`gamma_*`, `panorama_*` and `verify_*` may refer to large source-workstation datasets or vendor
binaries; they are evidence/research tools, not production dependencies.

Portable routine entry points are:

- `code/scripts/run_all_tests.sh` — configure, build and run unit checks;
- `code/scripts/run_stage1_slam_acceptance.sh` — frozen Stage-1 solver acceptance;
- `scripts/run_frozen_acceptance.sh` — frozen Surface acceptance;
- `scripts/build_and_test.sh` — full local build and test helper;
- `scripts/update_manifest.py` — regenerate `MANIFEST.sha256` after intentional changes.
