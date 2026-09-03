# dora-openarm-dataset-recorder

A [Dora](https://dora-rs.ai/) node that records data as an OpenArm dataset.

## Commands

The `command` input uses a small lifecycle vocabulary:

- `start`, `success`, `fail`, `cancel`, `quit`
- `metadata` for dataset-level annotations

Extensible data is a JSON object in the Dora event metadata key `payload`. `start`
accepts `episode_number`, `task_index`, and an `episode` object. Completion commands
accept an `episode` object. `metadata` accepts this shape:

```json
{
  "dataset": {"evaluation": {"checkpoint": {"path": "/models/run"}}}
}
```

The legacy flat `episode_number` and `task_index` metadata on `start` remain
supported. The recorder owns `version`, `episodes`, and episode identity fields and
writes `metadata.yaml` atomically. When the dataflow declares the optional `result`
output, every command reports a JSON result containing `command`, `ok`, an optional
`episode_id`, and an error message on failure.

Active episodes are written under `episodes/.partial-<id>` and then published under the
final episode ID. Interrupted partial episodes and published directories missing from
`metadata.yaml` are moved to `orphaned/` on the next startup. Repeated arm action
snapshots are deduplicated per input and episode using their `timestamp`.

Dataset format 0.5.0 adds nullable `chunk_id` and `blended_chunk_id` columns to action
Parquet files, plus `policy/chunks.parquet`. Episode files written by 0.4.0 remain valid;
readers should treat absent chunk columns as null.

## Legacy evaluation metadata

Runtime commands no longer patch historical episodes. Convert an older rollout that has
`eval_metadata.yaml` or episodes without `source` once before resuming it:

```bash
uv run dora-openarm-migrate-eval-metadata --dataset-dir /path/to/dataset
```

Add `--dry-run` to inspect the decisions first. The command backs up `metadata.yaml`,
merges checkpoint and episode annotations while keeping recorder identities authoritative,
adds `source: rollout` where needed, and archives the old `eval_metadata.yaml`.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
