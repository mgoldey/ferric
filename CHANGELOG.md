# Changelog

User-visible changes, newest first. Numerical changes that move results are
listed with their size.

## Unreleased

### Changed

- **Direct J/K: bra-pair-local digest (`[scf] jk_digest`).** The direct
  four-centre J/K builders (`DirectJK` closed and open shell, full and
  incremental, rotated or not; `DirectJ`; `DirectK`; `rhf::build_jk` /
  `build_jk_with_pool`) now contract each computed shell quartet with a
  bra-pair-local digest: exchange is accumulated only into the rows of the
  two bra shells and the Coulomb blocks are written once per bra pair or
  quartet, with `J = Jh + Jhᵀ` and `K = Kb + Kbᵀ` formed once per build. The
  integrals, screening, work partition and fold order are unchanged, so
  results stay bitwise identical across thread counts; only the summation
  order differs. **J and K move by about 1e-14 relative and SCF energies in
  their last digits (about 1e-12 Ha)** against earlier releases.
  `[scf] jk_digest = "legacy"`, `RhfConfig::jk_digest = Some(JkDigest::Legacy)`
  or `FERRIC_JK_DIGEST=legacy` restores the previous 8-fold scatter bit for
  bit (Python: the environment variable). The in-core store, LinK, CFMM,
  density fitting, gradients, the open-shell response build and the
  block-CPHF multi-density build are unchanged.
