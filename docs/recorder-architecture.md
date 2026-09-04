# OpenArm Dataset Recorder Architecture and Data Format

This document describes the current implementation of
`dora-openarm-dataset-recorder` 0.5.0. It covers runtime responsibilities,
the episode lifecycle, the Dora event protocol, persistence and recovery,
metadata ownership, Parquet schemas, and the protection boundaries that the
implementation does and does not provide.

Relevant implementation files:

- [`main.py`](../src/dora_openarm_dataset_recorder/main.py)
- [`migrate_eval_metadata.py`](../src/dora_openarm_dataset_recorder/migrate_eval_metadata.py)
- [`test_metadata.py`](../tests/test_metadata.py)
- [`test_migrate_eval_metadata.py`](../tests/test_migrate_eval_metadata.py)
- [`test_frequency_detector.py`](../tests/test_frequency_detector.py)

## 1. Design Goals and Responsibility Boundaries

The recorder has a deliberately narrow set of responsibilities:

1. Receive episode lifecycle commands from a UI or another coordinator.
2. Collect robot, camera, and optional policy-chunk events while an episode is
   active.
3. Publish each completed episode as an independent directory.
4. Act as the sole writer of `metadata.yaml`, including dataset-level metadata
   and episode results.
5. Avoid presenting incomplete data as a valid episode after ordinary
   exceptions or process interruptions.

The recorder does not:

- Decide the UI state or task-switching policy.
- Control the arms, IK, teleoperation, or the action mux.
- Resample, interpolate, sort, or time-align input streams.
- Determine whether an action was physically executed successfully.
- Automatically recover or merge data in `orphaned/`.

The overall data path is:

```text
UI command ----------------------+
arm/lifter/camera events --------+--> main event loop --> Episode buffer
optional policy_chunk -----------+                          |
optional ker_metadata -----------+                          v
                                                    EpisodeWriter
                                                         |
                                      .partial-<id> --> <id>
                                                         |
                                                         v
                                                    DatasetWriter
                                                         |
                                                    metadata.yaml

optional result output <---------------------------------+
```

Only the recorder writes `metadata.yaml`. The UI and other nodes submit patches
through the `metadata` command instead of maintaining a second episode-history
file.

## 2. Main Components

| Component | Lifetime | Responsibility |
| --- | --- | --- |
| `Episode` | Created for each `start` | Holds episode identity, result fields, and in-memory stream data |
| `EpisodeWriter` | One per active episode | Writes images and Parquet files, then publishes or cancels the episode directory |
| `DatasetWriter` | Entire recorder process | Loads and writes `metadata.yaml`, checks IDs, and isolates interrupted directories |
| `FrequencyDetector` | Startup | Derives nominal frequencies from timer paths in the Dora dataflow |
| `main()` | Entire recorder process | Parses Dora events and runs the command state machine and data routing |

There is no separate runtime state-machine class. The core state is represented
by a few variables:

```text
episode is None                    idle
episode is not None                recording
episode_writer                     writer for the current .partial-<id>
last_action_timestamps             action deduplication cache for this episode
```

The recorder therefore permits at most one active episode at a time.

## 3. Startup Configuration

Every command-line option can also be supplied through an environment variable:

| Option | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `--directory` | `DIRECTORY` | Current working directory | Parent directory of the dataset |
| `--name` | `NAME` | `dataset` | Dataset directory name |
| `--metadata-file` | `METADATA_FILE` | None | Initial metadata template |
| `--operation-type` | `OPERATION_TYPE` | `teleop` | `teleop` or `rollout` |
| `--docker-image` | `DOCKER_IMAGE` / `IMAGE` | None | Model image information for rollout collection |

The final dataset path is:

```text
<DIRECTORY>/<NAME>
```

Startup proceeds in this order:

1. Create the Dora node.
2. Read `METADATA_FILE`; an empty file is treated as `{}`.
3. Add dynamic metadata, including the operation type, model or equipment
   placeholders, and nominal frequencies.
4. Create `DatasetWriter` and load an existing `metadata.yaml`, if present.
5. Isolate directories left by an interrupted previous process.
6. Enter the Dora event loop.

Dynamic metadata follows these rules:

- The current `operation_type` is always written.
- `teleop` ensures that an `equipment.embodiments` container exists.
- `rollout` ensures that `model` exists and writes `model.docker_image` when it
  is configured.
- The `frequencies` container is rebuilt and populated from frequencies that
  can be derived from the current dataflow topology.

## 4. Dora Input and Output Protocol

### 4.1 Input Overview

Input names are selected by the dataflow, but the recorder interprets the
following exact names and prefixes:

| Input | Required | Purpose | Important metadata |
| --- | --- | --- | --- |
| `command` | Required for lifecycle control | `start/success/fail/cancel/metadata/quit` | Optional `payload` |
| `arm_right_action` | Optional | Final right-arm action | `timestamp`, optional chunk fields |
| `arm_left_action` | Optional | Final left-arm action | `timestamp`, optional chunk fields |
| `arm_*_observation` | Optional | Arm observation | `timestamp` |
| `elevation_action` | Optional | Lifter action | `timestamp` |
| `elevation_observation` | Optional | Lifter observation | `timestamp` |
| `camera_*` | Optional | Encoded image | `timestamp`, `encoding` |
| `policy_chunk` | Optional | Raw policy action chunk | `episode_attempt_id`, `interval`, and related fields |
| `ker_metadata` | Optional | KER leader firmware and hardware information | No additional requirement |

A dataflow may declare only the inputs it actually uses. In particular, a pure
teleoperation collection flow may omit `policy_chunk` entirely. Omitting it does
not affect action, observation, camera, or episode lifecycle recording.

If a dataflow retains `policy_chunk: policy-server/actions`, the referenced node
and output must exist in that dataflow. A missing source is a dataflow wiring
error, not a recorder requirement for the optional input.

### 4.2 Command Representation

The command name is the first element of the event value. Extensible command
data is stored in the Dora event metadata under `payload`. `payload` must be a
string containing a JSON object:

```yaml
value: ["start"]
metadata:
  payload: >-
    {"episode_number":12,"episode_attempt_id":"a1b2c3","task_index":0,
     "episode":{"source":"rollout","started_at":"2026-09-04T12:00:00"}}
```

Missing `payload` is treated as an empty object, `{}`. The following values are
rejected:

- A non-string payload.
- Invalid JSON.
- A JSON array, number, string, or any other non-object top-level value.

For `start`, `episode_number`, `episode_attempt_id`, and `task_index` are read
from the payload first. To preserve the recorder's general-purpose legacy API,
missing values fall back to top-level Dora event metadata, then to `0`, `None`,
and `0`, respectively.

"Top-level Dora event metadata" means fields adjacent to `payload` in the event
metadata mapping. They are transport metadata, not fields inside the JSON
payload. For example:

```yaml
metadata:
  episode_number: 12
  episode_attempt_id: a1b2c3
  task_index: 0
  payload: '{"episode":{"source":"rollout"}}'
```

The current Evaluation UI includes identity both in the start payload and in
top-level event metadata. New integrations should treat the JSON payload as the
complete business object; the fallback exists for older generic callers.

### 4.3 Command List

| Command | While idle | While recording | Data result |
| --- | --- | --- | --- |
| `start` | Creates an episode and `.partial-<id>` | Returns an error and leaves the current episode active | Does not write an episode result |
| `success` | Returns an error | Completes with `success: true` | Publishes the directory and appends a metadata result |
| `fail` | Returns an error | Completes with `success: false` | Publishes the directory and appends a metadata result |
| `cancel` | Succeeds as a no-op | Deletes the partial directory and returns to idle | Does not append a metadata result |
| `metadata` | Merges a dataset patch | Also allowed | Updates only `metadata.yaml` |
| `quit` | Exits | Completes the active episode, then exits | An active episode is recorded as failed by default |

`quit` is not equivalent to `success`. `Episode.success` starts as `False`, and
`quit` does not change it, so an episode closed by `quit` is written with
`success: false`.

### 4.4 Start Payload

Recommended structure:

```json
{
  "episode_number": 12,
  "episode_attempt_id": "8d3f42f4f0c94d498cad92f50e52ca73",
  "task_index": 0,
  "episode": {
    "source": "rollout",
    "started_at": "2026-09-04T12:00:00",
    "experiment_config": {
      "seed": 1
    }
  }
}
```

`episode` must be an object. It contains extensible episode fields but cannot
override the final `id`, `success`, or `task_index` written by the recorder. If
`start` supplies a non-empty attempt ID, the recorder also uses it to override
any `episode_attempt_id` in the extensible fields. The current Evaluation UI
always supplies this ID.

### 4.5 Completion Payload

`success`, `fail`, and `quit` while recording accept:

```json
{
  "episode": {
    "note": "clean grasp",
    "timestamp": "2026-09-04T12:00:12"
  }
}
```

Completion fields are recursively merged into the episode metadata saved at
`start`. Nested objects are deep-merged; all other values from the completion
payload replace their earlier values. The recorder then writes authoritative
identity and result fields last.

### 4.6 Dataset Metadata Payload

The `metadata` command accepts only a dataset patch:

```json
{
  "dataset": {
    "evaluation": {
      "schema_version": "0.1.0",
      "checkpoint": {
        "path": "/models/checkpoint"
      }
    }
  }
}
```

The patch is merged recursively. The recorder preserves `version` and
`episodes`, even if they appear inside the `dataset` object. A top-level
`episodes` field is interpreted as an obsolete historical-episode patch and is
rejected; callers must use the one-time migration tool instead.

### 4.7 Result Output

If the dataflow declares `result` in the recorder outputs, every command returns
a JSON string synchronously after processing:

```json
{"command":"success","ok":true,"episode_id":"12"}
```

Failure example:

```json
{
  "command": "start",
  "ok": false,
  "episode_id": "12",
  "error": "Episode 12 already exists in dataset"
}
```

| Field | Meaning |
| --- | --- |
| `command` | Command that was processed |
| `ok` | Whether the command completed successfully |
| `episode_id` | Included when an episode can be identified; encoded as a string |
| `error` | Exception text on failure |

When `result` is not declared, the recorder performs the same operation but
does not send an acknowledgment. The Evaluation UI reloads `metadata.yaml` only
after acknowledgments, so that integration should always declare `result`.

## 5. Episode Lifecycle

### 5.1 Start

After receiving a valid `start`, the recorder:

1. Verifies that there is no active episode.
2. Parses identity from the payload and compatibility metadata.
3. Verifies that the episode ID is absent from existing metadata.
4. Verifies that neither the final nor partial directory exists.
5. Creates `episodes/.partial-<id>`.
6. Installs the candidate as the active `Episode`.
7. Clears the action-timestamp deduplication cache.
8. Returns a successful result.

The active state is replaced only after both the candidate and writer have been
created successfully. A failed start validation therefore does not leave a
partially initialized active episode.

### 5.2 While an Episode Is Active

Different streams use different storage strategies:

- Arm, lifter, and policy-chunk data are buffered in Python lists or mappings
  on `Episode`.
- Camera images are written into the partial directory as they arrive.
- Dataset metadata patches may be persisted independently while an episode is
  active.
- Policy chunks unrelated to the active episode are ignored.

When `episode is None`, ordinary action, observation, and camera events are
ignored and do not update the deduplication cache. `command` and `ker_metadata`
are not subject to this restriction.

### 5.3 Success or Fail

Completion follows this order:

```text
merge start metadata + completion metadata
                  |
force id/success/task_index/attempt_id
                  |
append result to the in-memory episode list
                  |
write pending Parquet files into .partial-<id>
                  |
rename .partial-<id> to <id>
                  |
write metadata.yaml.tmp
                  |
os.replace(metadata.yaml.tmp, metadata.yaml)
                  |
return command result
```

After a successful completion, the main loop clears `episode` and
`episode_writer` and returns to idle.

### 5.4 Cancel

`cancel` recursively removes `.partial-<id>`, clears the active episode, and
does not append to the `episodes` list in `metadata.yaml`. It is intended for
discarding an incomplete collection. Use `fail` when failed data must be kept.

### 5.5 Consecutive Episodes

The recorder does not generate the next episode number and does not switch
tasks automatically. The caller should wait for a successful completion result,
reload recorder-owned metadata, and then send the next `start`.

The Evaluation UI follows this ownership model for both rollout task switches
and rollout-to-intervention transitions: persisted `metadata.yaml` is the
authority for completed episode results.

## 6. Data Event Processing

### 6.1 Timestamps

Arm, lifter, and camera events require `timestamp` in Dora event metadata.

- A `datetime.datetime` supplied by Dora is converted to POSIX nanoseconds.
- A numeric timestamp is passed directly to the Arrow `timestamp[ns]` column.
- The recorder does not correct device clocks or unify sampling across streams.

A timestamp in an output file is the timestamp attached to the corresponding
input event. It does not guarantee that a physical action had finished
executing at that time.

Policy chunks use `generated_timestamp_ns`; they fall back to `timestamp`, and
the column is null if both are missing. Policy chunks still require `interval`
in their event metadata.

### 6.2 Action Deduplication

Arm actions commonly come from the driver's `latest_command` output, which may
publish repeated snapshots of the same command. The recorder stores the last
accepted `timestamp` independently for `arm_right_action` and
`arm_left_action`:

```text
same input + same timestamp as previous accepted action -> drop
otherwise                                               -> record
```

The cache is updated only while an episode is active and is cleared on every
`start`. The final action of one episode therefore cannot suppress the first
action of the next episode.

This is not a global set of seen timestamps. The sequence `10, 11, 10` records
all three values. Its purpose is to remove duplicate latest-command snapshots,
not to reorder or clean out-of-order events.

Observation, camera, lifter, and policy-chunk events do not use this
deduplication rule.

### 6.3 Arm Value Compatibility

Arm inputs support two main representations:

1. A `StructArray`, from which the writer preserves supported `qpos`, `qvel`,
   `qtorque`, and `pose` fields.
2. A non-struct array, which is written as `qpos`.

If a `StructArray` contains `new_position`, the main loop extracts that field
first. This supports nodes whose output wraps both a command envelope and the
final position. All motion values are ultimately converted to `float32` lists.

### 6.4 Associating Policy Chunks with Execution Data

A policy server may emit a chunk such as:

```text
policy_chunk metadata:
  chunk_id
  episode_number
  episode_attempt_id
  generated_timestamp_ns or timestamp
  interval

policy_chunk value:
  [action_0, action_1, ...]
```

The recorder accepts the chunk only when:

```text
event.episode_attempt_id == active_episode.attempt_id
```

A mismatch is silently dropped so a delayed policy response from the previous
episode cannot enter the new episode. For compatibility with generic legacy
callers that supply no identity, two `None` values still compare equal.

An arm action carrying the same `chunk_id` may also provide optional execution
fields:

- `executor_received_timestamp_ns`
- `blended_chunk_id`
- `blend_policy_points`

At episode completion, each policy record is joined with those execution fields
by `chunk_id`. If multiple actions provide the same field for one chunk, the
last received value wins. If no matching action exists, execution fields remain
null.

For pure teleoperation data with no policy chunks or chunk IDs:

- `policy/chunks.parquet` is not created.
- Arm actions are still recorded normally.
- The chunk columns in action Parquet files exist but contain null values.

### 6.5 KER Metadata

The first element of a `ker_metadata` event value must be a JSON string. The
recorder maps `fw` and `hw` into:

```yaml
equipment:
  leader:
    ker:
      id: OpenArmKER
      firmware_version: <fw>
      hardware_version: <hw>
```

This update does not require an active episode and immediately rewrites
`metadata.yaml`.

## 7. File Layout

Standard dataset layout:

```text
<DIRECTORY>/<NAME>/
|-- metadata.yaml
|-- episodes/
|   |-- 0/
|   |   |-- action/
|   |   |   |-- arms/
|   |   |   |   |-- left/state.parquet
|   |   |   |   `-- right/state.parquet
|   |   |   `-- lifter/elevation.parquet
|   |   |-- obs/
|   |   |   |-- arms/
|   |   |   |   |-- left/state.parquet
|   |   |   |   `-- right/state.parquet
|   |   |   `-- lifter/elevation.parquet
|   |   |-- cameras/
|   |   |   |-- ceiling/<timestamp>.jpeg
|   |   |   |-- head_left/<timestamp>.jpeg
|   |   |   `-- wrist_right/<timestamp>.jpeg
|   |   `-- policy/chunks.parquet
|   `-- 1/
`-- orphaned/
    |-- .partial-2.<timestamp>/
    `-- 3.<timestamp>/
```

Parquet files and camera subdirectories are created only for streams that
actually produced data. `policy/chunks.parquet` is also optional.

During collection, the directory is named `.partial-<id>`. Completion uses
`os.replace` to rename the entire directory to its numeric episode ID. The
partial and final directories share the same `episodes/` parent, avoiding a
cross-filesystem move.

## 8. Data Format 0.5.0

### 8.1 Arm Action

Path:

```text
episodes/<id>/action/arms/<left|right>/state.parquet
```

| Column | Arrow type | Nullable | Meaning |
| --- | --- | --- | --- |
| `timestamp` | `timestamp[ns]` | No | Dora event timestamp |
| `qpos` | `list<float32>` | Depends on input | Joint position or non-struct input |
| `qvel` | `list<float32>` | Depends on input | Optional joint velocity |
| `qtorque` | `list<float32>` | Depends on input | Optional joint torque |
| `pose` | `list<float32>` | Depends on input | Optional pose |
| `chunk_id` | `string` | Yes | Policy chunk that produced this action |
| `blended_chunk_id` | `string` | Yes | Previous chunk associated through blending |

Only supported motion fields actually present in the input are written. Action
files produced by the standard 0.5.0 runtime always include the two nullable
chunk columns; teleoperation actions usually contain null in every row.

### 8.2 Arm Observation

Path:

```text
episodes/<id>/obs/arms/<left|right>/state.parquet
```

Motion fields match arm action files, but observations do not include
`chunk_id` or `blended_chunk_id`.

### 8.3 Lifter

Paths:

```text
episodes/<id>/action/lifter/elevation.parquet
episodes/<id>/obs/lifter/elevation.parquet
```

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `timestamp` | `timestamp[ns]` | Dora event timestamp |
| `value` | `list<float32>` | Elevation value |

### 8.4 Policy Chunks

Path:

```text
episodes/<id>/policy/chunks.parquet
```

| Column | Arrow type | Nullable | Source |
| --- | --- | --- | --- |
| `chunk_id` | `string` | Yes | Policy event metadata |
| `episode_number` | `int64` | Yes | Event metadata, falling back to the active episode |
| `episode_attempt_id` | `string` | Yes | Policy event metadata |
| `generated_timestamp_ns` | `timestamp[ns]` | Yes | `generated_timestamp_ns` or `timestamp` |
| `interval_ns` | `int64` | No | Policy event metadata `interval` |
| `chunk_received` | `list<list<float32>>` | No | Policy event value |
| `executor_received_timestamp_ns` | `timestamp[ns]` | Yes | Matching action metadata |
| `blended_chunk_id` | `string` | Yes | Matching action metadata |
| `blend_policy_points` | `int32` | Yes | Matching action metadata |

`interval_ns` is copied directly from upstream `interval` metadata. The recorder
does not convert its unit, so upstream must provide nanoseconds.

### 8.5 Camera Images

Path:

```text
episodes/<id>/cameras/<camera_name>/<timestamp>.<encoding>
```

The recorder does not re-encode images. It writes the encoded bytes from the
Arrow buffer directly, and the extension comes from event metadata
`encoding`. Images are written into the partial directory during collection,
so they do not all need to remain in memory.

### 8.6 Compatibility with 0.4.0

Version 0.5.0 adds:

- Nullable `chunk_id` in arm action files.
- Nullable `blended_chunk_id` in arm action files.
- Optional `policy/chunks.parquet`.

The recorder does not rewrite existing 0.4.0 episodes. A single dataset may
therefore contain both:

- Old action files without chunk columns.
- New action files with chunk columns whose values may all be null.

Downstream readers must materialize missing columns as null rather than assume
that every Parquet file in one dataset has an identical physical schema. This
repository writes data but does not provide a unified compatibility reader.

## 9. metadata.yaml

### 9.1 Example

```yaml
version: 0.5.0
location: Lab
operator: Tester
operation_type: rollout
tasks:
  - prompt: Pick a spoon.
evaluation:
  schema_version: 0.1.0
  checkpoint:
    path: /models/checkpoint
frequencies:
  action:
    arms:
      right: 250.0
  obs:
    arms:
      right: 250.0
  cameras:
    ceiling: 30.303
episodes:
  - id: "12"
    success: true
    task_index: 0
    episode_attempt_id: 8d3f42f4f0c94d498cad92f50e52ca73
    source: rollout
    started_at: "2026-09-04T12:00:00"
    note: clean grasp
```

Except for reserved fields, the metadata schema is an extensible mapping. See
[`examples/metadata.yaml`](../examples/metadata.yaml) for typical equipment and
task metadata.

### 9.2 Field Ownership

The recorder owns and enforces:

- Root-level `version`.
- Root-level `episodes`.
- Each episode's `id`, `success`, and `task_index`.
- `episode_attempt_id` when present.

Callers own or provide:

- `tasks`, `location`, `operator`, and `equipment`.
- `evaluation`, `model`, and experiment configuration.
- Extensible episode fields such as `source`, `started_at`, and `note`.

This split lets the UI describe an experiment without allowing a metadata patch
to rewrite the identity or result of a persisted episode.

### 9.3 Resuming an Existing Dataset

When the dataset directory already exists, the recorder:

1. Loads the existing `metadata.yaml` and episode list.
2. Compares overlapping leaf fields between startup metadata and existing
   metadata.
3. Excludes `version` and `episodes` from conflict checks.
4. Preserves additional fields that exist only in persisted metadata.
5. Fails startup if the same path contains different values instead of silently
   replacing experiment identity.
6. Preserves existing non-reserved fields, then merges current startup metadata.

This allows the recorder to retain fields added at runtime, such as KER firmware
versions, while preventing a dataset from being resumed with conflicting
location, task, or equipment configuration.

### 9.4 Writing the Metadata File

Writes use:

```text
serialize complete mapping -> metadata.yaml.tmp -> os.replace -> metadata.yaml
```

A reader therefore never observes a partially written YAML file.
`update_metadata()` performs its merge on a copy; if validation or writing
fails, the recorder restores its previous in-memory metadata.

Constructing a new `DatasetWriter` does not immediately create
`metadata.yaml`. The first successful metadata command, KER metadata update, or
episode completion performs the first actual write.

This provides atomic replacement for one metadata file. It is not a database
transaction spanning both an episode directory and metadata.

## 10. Nominal Frequency Derivation

At startup, `FrequencyDetector` starts at each declared recorder input's source
node and walks upstream through one of these inputs:

1. `tick`
2. `request_state`
3. `request_position`

If the path eventually reaches a Dora timer, it derives:

```text
dora/timer/millis/4  -> 250 Hz
dora/timer/millis/33 -> 30.303... Hz
dora/timer/secs/1    -> 1 Hz
```

Results are mapped from recorder input names into:

```yaml
frequencies:
  action:
    arms: {}
  obs:
    arms: {}
  cameras: {}
```

These values are nominal frequencies inferred from the dataflow topology, not
measured event-arrival frequencies. If an action passes through a mux that has
no upstream `tick/request_*` input to follow, derivation returns no value and
that frequency is omitted. Data events are still recorded one by one.

`policy_chunk` is not represented in the arm or camera frequency fields, so
connecting or omitting it has no effect on this metadata.

## 11. Protection and Recovery

### 11.1 Identity and Directory Conflicts

Episode creation checks all of the following:

- Whether the ID already appears in the `episodes` list in `metadata.yaml`.
- Whether `episodes/<id>` already exists.
- Whether `episodes/.partial-<id>` already exists.

Any conflict fails `start`; existing data is never overwritten.

### 11.2 Policy Attempt Isolation

An active episode retains the `episode_attempt_id` supplied at `start`. Only a
`policy_chunk` with exactly the same attempt ID is added to that episode. The
main purpose is to keep asynchronous policy responses from crossing an episode
boundary.

Strict identity filtering currently applies only to policy chunks. Arm actions,
observations, lifter data, and cameras are assigned according to whichever
episode is active when the event arrives; they do not check attempt IDs. This is
an intentional small-scope boundary and must not be interpreted as end-to-end
identity isolation for every stream.

### 11.3 Ordinary Python Exceptions

If writing Parquet or metadata raises an exception while completing an episode:

- The uncommitted result is removed from the in-memory episode list.
- If the episode directory has already been published, it is renamed back to
  its partial name.
- The command result reports `ok: false`.
- The main loop keeps the episode active, allowing the caller to retry
  completion or send `cancel`.

### 11.4 Process Interruption

SIGKILL or power loss cannot run Python rollback and may leave either:

1. `episodes/.partial-<id>`.
2. A renamed `episodes/<id>` directory whose ID is not yet present in metadata.

On the next startup, the recorder moves both forms into:

```text
orphaned/<original-name>.<YYYYmmdd-HHMMSS>[.<suffix>]
```

A numeric episode directory already registered in metadata is not moved.
`orphaned/` is only for isolation and manual inspection; its contents are never
restored automatically.

This covers the crash window between directory rename and metadata update. It
does not provide `fsync`-level durability against power loss, and it does not
repair metadata whose episode directory was deleted externally.

### 11.5 Historical Result Protection

The runtime `metadata` command rejects top-level `episodes` patches. Historical
conversion is kept separate from online collection so a UI cannot create,
replace, or repair old episode results in bulk while recording.

Old data is handled only by the one-time migration utility.

## 12. Migrating Legacy Evaluation Metadata

An old rollout dataset may contain both:

```text
metadata.yaml
eval_metadata.yaml
```

It may also contain rollout episodes in `metadata.yaml` without `source`. Before
resuming collection, run:

```bash
uv run dora-openarm-migrate-eval-metadata \
  --dataset-dir /path/to/dataset \
  --dry-run
```

Inspect the output, then remove `--dry-run`.

Migration rules:

1. `id`, `success`, and `task_index` from `metadata.yaml` take precedence.
2. Extensible fields such as note and timestamp are merged from
   `eval_metadata.yaml`.
3. The old checkpoint is moved to `evaluation.checkpoint`.
4. The old evaluation version is moved to `evaluation.schema_version`.
5. On task-definition conflicts, tasks from `metadata.yaml` are preserved and a
   diagnostic is printed.
6. Missing `source` on rollout episodes is filled with `source: rollout`.
7. An episode that exists only in evaluation metadata is restored only when its
   directory exists and it has both success and task-index fields.
8. `metadata.yaml` is backed up before writing.
9. After success, `eval_metadata.yaml` is renamed as an archive.

The migration utility does not rewrite episode Parquet files and does not
upgrade metadata version to 0.5.0 itself. A subsequent write by the 0.5.0
recorder maintains the current recorder format version.

## 13. Explicit Non-Guarantees

To keep the recorder focused, the current implementation does not:

- Filter action, observation, or camera streams by attempt ID.
- Resample, interpolate, sort, or synchronize clocks across streams.
- Verify physical execution of an action.
- Implement a database-style transaction or write-ahead log for the full
  dataset.
- Automatically recover `orphaned/`.
- Limit episode duration or in-memory sample count.
- Define a fixed schema for every extensible metadata field.
- Guarantee that nominal dataflow frequency equals actual arrival frequency.

Arm, lifter, and policy data remain in memory until episode completion, so a
very long episode consumes memory in proportion to its sample count. Camera
data is written directly to the partial directory and does not have the same
growth pattern. If the process is killed before normal completion, in-memory
motion and policy data cannot be recovered; the partial directory normally
contains only camera files already written, or files produced during a
partially completed finish operation.

These boundaries are explicit, not implied guarantees. When stronger behavior
is required, first identify the concrete failure mode, then add the smallest
mechanism at the relevant input or writer boundary instead of making the
recorder responsible for control, synchronization, and data cleaning.

## 14. Recommended Dataflow Integration

### Rollout

```yaml
- id: recorder
  path: dora-openarm-dataset-recorder
  env:
    OPERATION_TYPE: rollout
  inputs:
    command:
      source: evaluation-ui/recorder_command
      queue_size: 10
    arm_right_action: arm-right/latest_command
    arm_right_observation: arm-right/state
    arm_left_action: arm-left/latest_command
    arm_left_observation: arm-left/state
    policy_chunk: policy-server/actions
    camera_ceiling: camera-ceiling/image
  outputs:
    - result
```

### Pure Teleoperation

```yaml
- id: recorder
  path: dora-openarm-dataset-recorder
  env:
    OPERATION_TYPE: teleop
  inputs:
    command:
      source: ui/command
      queue_size: 10
    arm_right_action: arm-right/latest_command
    arm_right_observation: arm-right/state
    arm_left_action: arm-left/latest_command
    arm_left_observation: arm-left/state
    camera_ceiling: camera-ceiling/image
  outputs:
    - result
```

A teleoperation flow does not need a placeholder policy node and does not need
to send empty chunks to the recorder. Omit the `policy_chunk` input directly.

## 15. Test Coverage Priorities

Current tests cover:

- Dataset metadata patches and episode-field ownership.
- Resuming existing metadata and conflict handling.
- Rejection of historical episode patches.
- In-memory rollback after metadata write failure.
- Directory rollback after episode publication failure.
- Startup isolation of partial and unregistered final directories.
- Version 0.5.0 action chunk columns and policy-chunk files.
- Strict JSON-object parsing for command payloads.
- Legacy evaluation metadata merge, backup, archive, and dry-run behavior.
- Nominal frequency derivation through Dora timer paths.

When recorder behavior changes, extend these existing boundaries with focused
tests rather than copying UI state-machine or dataflow-specific policy into the
recorder.
