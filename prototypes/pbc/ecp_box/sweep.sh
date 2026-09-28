#!/bin/bash
# $1 = tag (output dir suffix); runs mol then box 12..36
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONPATH=../..:../../pyshim
mkdir -p out_$1; cp run_box.py basis.py out_$1/; cd out_$1; rm -f box.jsonl
PY=/home/matt/qc/ferric/.venv/bin/python
$PY run_box.py mol 2>&1 | grep -v scf-ladder > mol.log
$PY run_box.py box 12 16 20 24 28 32 36 40 44 48 2>&1 | grep -v scf-ladder > box.log
echo DONE >> box.log
