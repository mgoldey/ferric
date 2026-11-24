<!-- stack-overview v1 2026-11-20 -->
# The stack: ferric, smeltery and attestation

Three projects, three layers. **ferric** computes (a Rust quantum chemistry
engine with Python bindings). **smeltery** applies it (a tiered drug-screening
funnel, an MCP server, job backends, dataset export). **attestation** observes
what was run (a ledger of runs, a claim checker, an MCP server, and the core
run record schema the other two can write). Each keeps its own documentation;
this page is the one map across them.

> **Mirrored page.** This exact file is kept, byte for byte, in all three
> repositories (`docs/stack.md` in each). Edit it in all three together and bump
> the version line at the top. Each repository has a test that pins that line,
> so a change made in only one repository is a reminder, not a silent drift.
> The test cannot compare against the other two copies: CI has only one
> checkout.

## The three layers

```text
  attestation   reads core run records, checks claims, serves MCP tools
       ^          (imports neither smeltery nor ferric)
       :  files only: <run dir>/attestation/run.json, dataset.json, digests.json
       :
  smeltery      funnel, MCP server, jobs, dataset export
       |          (optional extra: attestation-schema, never attestation)
       v  imports
  ferric        engine: Rust core, PyO3 bindings, QCSchema, MLIP export
                  (imports nothing above it)
```

The dependency arrows point down and stop:

- **ferric** imports nothing above it. Its Python package has a test that scans
  every module for an import of `smeltery` or `attestation`.
- **smeltery** depends on ferric. Optionally (the `attestation` extra) it
  depends on `attestation-schema`, the small stdlib-only package that defines
  the core run record. It never depends on `attestation` itself.
- **attestation** imports neither smeltery nor ferric (a boundary test checks
  it). It reads the files that producers write. smeltery is one producer;
  anything that writes a core run record, such as an MLflow exporter or a
  training callback, is another.

So the three can be used alone, in pairs, or together, and the only thing
smeltery and attestation share is a run directory on disk.

## What flows between them

1. **Build identity.** ferric stamps the compiled extension at build time:
   `ferric.__build__` is `{"git_sha": ..., "dirty": ...}` (or `None` for an
   sdist with no git metadata). smeltery reads it as a *verified* dependency
   identity and writes it into its core record. attestation's ledger turns an
   unverified or dirty identity into a caveat on `runs.detail` and
   `runs.compare`. It is reported as a caveat, never as a verdict on the
   numbers.
2. **The core run record.** After a finished run, smeltery writes
   `OUT/attestation/run.json` (schema `attestation.run/1`, frozen) beside its
   own legacy record, through `attestation-schema`'s writer, with a
   `digests.json` manifest binding every file in the run directory. attestation
   scans, ranks and checks claims against these records. A record that fails
   the schema's conformance check is quarantined rather than guessed at.
3. **Job state.** A submitted job can also write `attestation/state.json`
   (`attestation.state/1`). attestation's `runs.status` reads it at query time
   to list runs that are in flight, lagging or unrecorded.
4. **Datasets and lineage.** `smeltery export dataset` writes dataset
   directories (`attestation.dataset/1`, still a draft schema). `attest runs
   lineage` follows a training run's data digests to those datasets and back to
   the runs that produced them, by digest only, and reports broken or
   unverified links. Reading a dataset directory needs `attestation-schema`
   0.2.0 or later.
5. **ferric's own outputs.** `ferric.qcschema.run` takes and returns QCSchema
   dicts, and `ferric.mlip` writes extended XYZ and ASE databases for ML
   potential training, each frame carrying `ferric_git_sha` and `ferric_dirty`
   from the same build stamp. These are ferric-level interfaces: smeltery's
   `src` and `docs` do not use them (checked by search), so they are not part
   of the smeltery to attestation path.

## Install and wire them together

Install what you need; none of it is required by the others.

- **ferric alone:** `pip install ferric`. The wheels are abi3 (`cp310-abi3`) and
  need no compiler. See ferric's installation page.
- **smeltery:** depends on ferric. Extras `mcp` (the MCP server) and
  `attestation` (writes the core run record) are the ones that matter here.
- **attestation:** `uvx attestation install` sets up and diagnoses; `attest-mcp`
  is its MCP server.
- **Both servers in Claude Code or Claude Desktop, on one workspace:** follow
  attestation's Claude guide and smeltery's MCP page rather than copying
  commands from here. Both pin their copy of the example with a test.
- **A set of versions known to work together:** smeltery's
  `compat/stack.toml` names it, `docs/compatibility.md` shows which ferric
  versions each smeltery passed against, and a CI job (`cross-server`) drives
  both MCP servers over stdio as an agent would. As of 2026-11-20 the pinned
  set runs attestation in its own environment, because smeltery's
  `attestation` extra caps `attestation-schema` below 0.2 while attestation
  requires 0.2 or later.

Which releases are on PyPI changes. The Claude guide and `compat/stack.toml`
say what was true when they were last checked; this page does not repeat it.
Not verified when this page was written: the current PyPI versions of the three
packages.

## Where each concern is documented

Links go to `main` on GitHub so they work from any of the three repositories.

### ferric

| Concern | Page |
|---|---|
| Overview, install, first calculation | [README](https://github.com/mgoldey/ferric/blob/main/README.md) |
| Installation, wheels, build stamp (`__build__`, `build_info`) | [installation](https://github.com/mgoldey/ferric/blob/main/site/src/using/installation.md) |
| QCSchema input and output | [qcschema](https://github.com/mgoldey/ferric/blob/main/site/src/using/qcschema.md) |
| Training data for ML potentials | [mlip](https://github.com/mgoldey/ferric/blob/main/site/src/using/mlip.md) |
| Contributing and the development loop | [CONTRIBUTING](https://github.com/mgoldey/ferric/blob/main/CONTRIBUTING.md) |

### smeltery

| Concern | Page |
|---|---|
| Overview and documentation index | [README](https://github.com/mgoldey/smeltery/blob/main/README.md) |
| MCP server and Claude install | [docs/mcp.md](https://github.com/mgoldey/smeltery/blob/main/docs/mcp.md) |
| The core run record smeltery writes | [docs/core-record.md](https://github.com/mgoldey/smeltery/blob/main/docs/core-record.md) |
| Dataset export (Parquet, Arrow, dataset directories) | [docs/dataset-export.md](https://github.com/mgoldey/smeltery/blob/main/docs/dataset-export.md) |
| Submitting and following jobs | [docs/jobs.md](https://github.com/mgoldey/smeltery/blob/main/docs/jobs.md) |
| ferric version matrix and the pinned stack | [docs/compatibility.md](https://github.com/mgoldey/smeltery/blob/main/docs/compatibility.md), [compat/stack.toml](https://github.com/mgoldey/smeltery/blob/main/compat/stack.toml) |
| The two MCP servers driven together in CI | [docs/cross-server.md](https://github.com/mgoldey/smeltery/blob/main/docs/cross-server.md) |

### attestation

| Concern | Page |
|---|---|
| Overview and quickstart | [README](https://github.com/mgoldey/attestation/blob/main/README.md) |
| Claude Code and Desktop with both servers | [docs/guides/claude.md](https://github.com/mgoldey/attestation/blob/main/docs/guides/claude.md) |
| The run ledger: scan, compare, status, lineage | [docs/guides/ledger.md](https://github.com/mgoldey/attestation/blob/main/docs/guides/ledger.md) |
| Claims and citations | [docs/guides/claims-and-citations.md](https://github.com/mgoldey/attestation/blob/main/docs/guides/claims-and-citations.md) |
| The core run record schema package | [packages/attestation-schema/README.md](https://github.com/mgoldey/attestation/blob/main/packages/attestation-schema/README.md) |
| Why the record is shaped as it is | [core run record design](https://github.com/mgoldey/attestation/blob/main/docs/superpowers/specs/2026-11-19-core-run-record-design.md), [dataset records design](https://github.com/mgoldey/attestation/blob/main/docs/superpowers/specs/2026-11-19-dataset-records-design.md) |
