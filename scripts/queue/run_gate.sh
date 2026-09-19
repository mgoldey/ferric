#!/bin/bash
cd /home/matt/qc/ferric
export FERRIC_TERF_TABLE_DIR=$PWD/terf-tables
CI_GATE_TIMEOUT_SECS=5400 scripts/ci-gate.sh > /tmp/claude-1000/-home-matt-qc-ferric/7457bb3f-12b9-4697-af3b-345b11ccf33e/scratchpad/gate_final.log 2>&1
echo "GATE_EXIT=$?" >> /tmp/claude-1000/-home-matt-qc-ferric/7457bb3f-12b9-4697-af3b-345b11ccf33e/scratchpad/gate_final.log
