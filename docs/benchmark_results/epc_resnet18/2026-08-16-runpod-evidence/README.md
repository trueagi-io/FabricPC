# Runpod evidence: independent ePC/sPC optimization

This directory contains the externally safe raw evidence for the
2026-08-16/17 Runpod execution described in [the result
report](../2026-08-17-independent-optimization-runpod.md).

## Identity

- Source commit: `403aeaa89d4095c444820e652f37d808e10e79f2`
- Tuning run: `rp_i86i6rvo58eyq9_tune_20260816`
- Final run: `rp_i86i6rvo58eyq9_final_20260817`
- Runpod Pod: `i86i6rvo58eyq9`
- Temporary network volume: `w5mfqaowy0`

## Contents

- `tune-runpod.log` and `final-runpod.log`: complete raw benchmark streams.
- `runpod-*-metadata.txt`: launch-time intent and exact commands. Their
  `status=running` values are historical launch state; the corresponding
  `runpod-*-completion.txt` files record terminal status and outcomes.
- `environment-freeze.txt` and `nvme-environment-freeze.txt`: dependency
  manifests before and after moving the environment to local NVMe. They are
  byte-identical.
- `nvme-env-install.log`: environment construction output, including the
  installed Optuna and GPU-enabled JAX packages.
- `dataset-sha256.txt`: hashes captured and revalidated before final mode.
- `benchmark-commit.txt`: exact benchmark source identity.
- `runpod-pod.json` and `runpod-network-volume.json`: redacted provider
  resource metadata.
- `teardown.json`: verification that the temporary paid resources were
  deleted.
- `run-summary.json`: scientific and integrity fields without account or
  billing data.
- `manifest.sha256`: checksum manifest for every other file in this directory.

The provider billing responses and account-balance reconciliation are retained
in the private expense archive, not this external debugging bundle.

## Verify

From this directory:

```text
sha256sum --check manifest.sha256
```
