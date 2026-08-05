# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to record data from OpenArm and cameras as OpenArm dataset."""

import argparse
from dataclasses import dataclass, field
import datetime
import copy
import dora
import json
import os
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import math
from numpy.typing import ArrayLike
import shutil
import yaml


@dataclass
class Episode:
    """Episode related data."""

    number: int = 0
    success: bool = False
    task_index: int = 0
    metadata: dict = field(default_factory=dict)

    right_action_timestamps: ArrayLike = field(default_factory=list)
    right_actions: ArrayLike = field(default_factory=list)
    right_observation_timestamps: ArrayLike = field(default_factory=list)
    right_observations: ArrayLike = field(default_factory=list)
    left_action_timestamps: ArrayLike = field(default_factory=list)
    left_actions: ArrayLike = field(default_factory=list)
    left_observation_timestamps: ArrayLike = field(default_factory=list)
    left_observations: ArrayLike = field(default_factory=list)
    elevation_action_timestamps: ArrayLike = field(default_factory=list)
    elevation_actions: ArrayLike = field(default_factory=list)
    elevation_observation_timestamps: ArrayLike = field(default_factory=list)
    elevation_observations: ArrayLike = field(default_factory=list)


def extract_values(value: pa.Array, key: str) -> np.ndarray:
    """Read `key` from a length-1 StructArray, or a flat array as-is."""
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.array(value, dtype=np.float32)


class EpisodeWriter:
    """Writer an episode."""

    def __init__(self, directory, episode):
        """Initialize variables."""
        self._directory = directory
        self._episode = episode
        episodes_directory = self._directory / "episodes"
        self._base_directory = episodes_directory / f".partial-{self._episode.number}"
        self._final_directory = episodes_directory / str(self._episode.number)
        self._base_directory.mkdir(parents=True)
        self._published = False

    def write_camera_image(self, name, image, timestamp, format):
        """Write an image from a camera."""
        output_path = self._base_directory / "cameras" / name / f"{timestamp}.{format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            # Using pa.PythonFile here is for zero-copy. We can't
            # write pa.Buffer data to Python's IO directory. If we use
            # Python's IO, we need to copy data in pa.Buffer as
            # Python's bytes. We want to avoid it.
            with pa.PythonFile(output) as pa_output:
                pa_output.write(image.buffers()[1])

    def finish(self):
        """Write all pending data and publish the completed episode directory."""
        if self._episode.right_actions:
            self._write_kinematic_state(
                self._base_directory / "action" / "arms" / "right",
                self._episode.right_action_timestamps,
                self._episode.right_actions,
            )
        if self._episode.right_observations:
            self._write_kinematic_state(
                self._base_directory / "obs" / "arms" / "right",
                self._episode.right_observation_timestamps,
                self._episode.right_observations,
            )
        if self._episode.left_actions:
            self._write_kinematic_state(
                self._base_directory / "action" / "arms" / "left",
                self._episode.left_action_timestamps,
                self._episode.left_actions,
            )
        if self._episode.left_observations:
            self._write_kinematic_state(
                self._base_directory / "obs" / "arms" / "left",
                self._episode.left_observation_timestamps,
                self._episode.left_observations,
            )
        if self._episode.elevation_actions:
            self._write_positions(
                self._base_directory / "action" / "lifter" / "elevation.parquet",
                self._episode.elevation_action_timestamps,
                self._episode.elevation_actions,
            )
        if self._episode.elevation_observations:
            self._write_positions(
                self._base_directory / "obs" / "lifter" / "elevation.parquet",
                self._episode.elevation_observation_timestamps,
                self._episode.elevation_observations,
            )
        os.replace(self._base_directory, self._final_directory)
        self._published = True

    def cancel(self):
        """Cancel this episode."""
        shutil.rmtree(self._base_directory, ignore_errors=True)

    def rollback_publish(self):
        """Restore a published directory when the metadata commit fails."""
        if self._published and self._final_directory.exists():
            os.replace(self._final_directory, self._base_directory)
            self._published = False

    def _write_positions(self, output_path, timestamps, positions):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_type = pa.list_(pa.float32())
        table = pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                "value": pa.array(positions, type=list_type),
            }
        )
        pq.write_table(table, output_path)

    def _write_kinematic_state(self, base_path, timestamps, states):
        output_path = base_path / "state.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        list_type = pa.list_(pa.float32())
        first = states[0]

        if isinstance(first, pa.StructArray):
            field_names = first.type.names
            available_field_names = [
                "qpos",
                "qvel",
                "qtorque",
                "pose",
            ]  # currently only support these fields
            state_fields = {"timestamp": pa.array(timestamps, type=pa.timestamp("ns"))}
            for field_name in field_names:
                if field_name not in available_field_names:
                    continue
                state_fields[field_name] = pa.array(
                    [extract_values(s, field_name) for s in states],
                    type=list_type,
                )
            table = pa.table(state_fields)
        else:
            # if the observation is not a struct, it should be qpos.
            table = pa.table(
                {
                    "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                    "qpos": pa.array(states, type=list_type),
                }
            )
        pq.write_table(table, output_path)


class DatasetWriter:
    """Write a dataset."""

    _VERSION = "0.4.0"
    _RESERVED_ROOT_FIELDS = {"version", "episodes"}

    def __init__(self, directory, name, metadata):
        """Initialize variables."""
        self._directory = directory
        self._name = name
        self._metadata = copy.deepcopy(metadata or {})
        self._base_directory = self._directory / name
        self._episode_results = []
        if self._base_directory.exists():
            existing = self._read_existing_metadata()
            if existing is not None:
                mismatched = _metadata_mismatches(
                    self._metadata,
                    existing,
                    ignored=self._RESERVED_ROOT_FIELDS,
                )
                if mismatched:
                    raise ValueError(
                        f"Existing dataset metadata does not match: {self._base_directory / 'metadata.yaml'}\n"
                        + "\n".join(
                            f"  {path}: existing={values[0]!r}, current={values[1]!r}"
                            for path, values in mismatched.items()
                        )
                    )
                preserved = {
                    key: value
                    for key, value in existing.items()
                    if key not in self._RESERVED_ROOT_FIELDS
                }
                _deep_merge(preserved, self._metadata)
                self._metadata = preserved
        else:
            self._base_directory.mkdir(parents=True)

        self._quarantine_partial_episodes()

        for field_name in self._RESERVED_ROOT_FIELDS:
            self._metadata.pop(field_name, None)

    def create_episode_writer(self, episode):
        """Create a writer for the given episode."""
        episode_id = str(episode.number)
        if episode_id in {result["id"] for result in self._episode_results}:
            raise ValueError(f"Episode {episode_id} already exists in dataset")
        episode_directory = self._base_directory / "episodes" / episode_id
        partial_directory = self._base_directory / "episodes" / f".partial-{episode_id}"
        if episode_directory.exists():
            raise ValueError(f"Episode directory already exists: {episode_directory}")
        if partial_directory.exists():
            raise ValueError(
                f"Partial episode directory already exists: {partial_directory}"
            )
        return EpisodeWriter(self._base_directory, episode)

    def finish_episode(self, episode, metadata=None, writer=None):
        """Publish an episode and atomically add its result to metadata."""
        result = copy.deepcopy(episode.metadata)
        _deep_merge(result, metadata or {})
        result.update(
            id=str(episode.number),
            success=episode.success,
            task_index=episode.task_index,
        )
        self._episode_results.append(result)
        try:
            if writer is not None:
                writer.finish()
            self._write_metadata_file()
        except Exception:
            self._episode_results.pop()
            if writer is not None:
                writer.rollback_publish()
            raise

    def update_metadata(self, payload):
        """Merge dataset metadata from a command payload."""
        if "episodes" in payload:
            raise ValueError(
                "historical episode patches are no longer supported; "
                "run dora-openarm-migrate-eval-metadata"
            )
        dataset_patch = payload.get("dataset", {})
        if not isinstance(dataset_patch, dict):
            raise ValueError("metadata payload 'dataset' must be an object")
        dataset_patch = {
            key: value
            for key, value in dataset_patch.items()
            if key not in self._RESERVED_ROOT_FIELDS
        }
        updated_metadata = copy.deepcopy(self._metadata)
        _deep_merge(updated_metadata, dataset_patch)
        previous_metadata = self._metadata
        self._metadata = updated_metadata
        try:
            self._write_metadata_file()
        except Exception:
            self._metadata = previous_metadata
            raise

    def _quarantine_partial_episodes(self):
        """Move interrupted episode directories aside for manual inspection."""
        episodes_directory = self._base_directory / "episodes"
        if not episodes_directory.exists():
            return
        partial_directories = sorted(episodes_directory.glob(".partial-*"))
        if not partial_directories:
            return
        orphaned_directory = self._base_directory / "orphaned"
        orphaned_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for partial_directory in partial_directories:
            destination = orphaned_directory / f"{partial_directory.name}.{timestamp}"
            suffix = 1
            while destination.exists():
                destination = orphaned_directory / (
                    f"{partial_directory.name}.{timestamp}.{suffix}"
                )
                suffix += 1
            shutil.move(str(partial_directory), destination)
            print(f"Quarantined interrupted episode: {destination}")

    def set_leader_ker_metadata(self, ker_metadata):
        """Record KER leader device metadata under equipment.leader.ker."""
        equipment = self._metadata.setdefault("equipment", {})
        leader = equipment.setdefault("leader", {})
        ker = leader.setdefault("ker", {})
        ker["id"] = "OpenArmKER"
        ker["firmware_version"] = ker_metadata.get("fw")
        ker["hardware_version"] = ker_metadata.get("hw")
        self._write_metadata_file()

    def _write_metadata_file(self):
        metadata = copy.deepcopy(self._metadata)
        metadata["version"] = self._VERSION
        metadata["episodes"] = self._episode_results
        output_path = self._base_directory / "metadata.yaml"
        temporary_path = output_path.with_suffix(".yaml.tmp")
        with open(temporary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                metadata,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        os.replace(temporary_path, output_path)

    def _read_existing_metadata(self):
        metadata_path = self._base_directory / "metadata.yaml"
        if not metadata_path.exists():
            return None
        with open(metadata_path, encoding="utf-8") as f:
            existing_metadata = yaml.safe_load(f) or {}
        self._episode_results = existing_metadata.get("episodes", [])
        return existing_metadata


def _deep_merge(target, patch):
    """Recursively merge a mapping into another mapping."""
    if not isinstance(patch, dict):
        raise ValueError("metadata patch must be an object")
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _metadata_mismatches(expected, existing, *, ignored=frozenset(), prefix=""):
    """Find conflicting leaves while allowing extra fields in existing metadata."""
    mismatches = {}
    for key, expected_value in expected.items():
        if key in ignored or key not in existing:
            continue
        path = f"{prefix}.{key}" if prefix else key
        existing_value = existing[key]
        if isinstance(expected_value, dict) and isinstance(existing_value, dict):
            mismatches.update(
                _metadata_mismatches(
                    expected_value,
                    existing_value,
                    prefix=path,
                )
            )
        elif expected_value != existing_value:
            mismatches[path] = (existing_value, expected_value)
    return mismatches


def parse_command_payload(metadata):
    """Parse the optional JSON object carried in Dora event metadata."""
    payload = metadata.get("payload")
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return copy.deepcopy(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("command metadata payload must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("command metadata payload must be a JSON object")
    return parsed


def _send_command_result(node, command, ok, episode_id=None, error=None):
    """Report whether a recorder command completed successfully."""
    outputs = node.node_config().get("outputs", [])
    if "result" not in outputs:
        return
    result = {"command": command, "ok": ok}
    if episode_id is not None:
        result["episode_id"] = str(episode_id)
    if error:
        result["error"] = str(error)
    node.send_output(
        "result",
        pa.array([json.dumps(result, ensure_ascii=True, separators=(",", ":"))]),
    )


def _is_duplicate_action(event_id, metadata, last_action_timestamps):
    """Deduplicate repeated latest-command snapshots by source timestamp."""
    action_timestamp = metadata.get("timestamp")
    if action_timestamp is None:
        raise ValueError("missing timestamp")
    if last_action_timestamps.get(event_id) == action_timestamp:
        return True
    last_action_timestamps[event_id] = action_timestamp
    return False


class FrequencyDetector:
    """Detect frequency of an input."""

    def __init__(self, configs):
        """Initialize with node configurations."""
        self._configs = configs

    def detect(self, input):
        """Detect frequency of the given input."""
        if isinstance(input, dict):
            input = input["source"]
        node_id, name = input.split("/", 1)
        for config in self._configs:
            if config["id"] != node_id:
                continue
            inputs = config["inputs"]
            next_input = (
                inputs.get("tick")
                or inputs.get("request_state")
                or inputs.get("request_position")
            )
            if next_input is None:
                continue
            if next_input.startswith("dora/timer/"):
                unit, value = next_input.split("/")[2:4]
                if unit == "secs":
                    return 1.0 / int(value)
                elif unit == "millis":
                    return 1_000.0 / int(value)
                else:
                    return None
            else:
                return self.detect(next_input)


def _collect_dynamic_metadata(metadata, args, node):
    metadata["operation_type"] = args.operation_type
    if args.operation_type == "teleop":
        if "equipment" not in metadata:
            metadata["equipment"] = {}
        if "embodiments" not in metadata["equipment"]:
            metadata["equipment"]["embodiments"] = {}
        # equipment.leader.ker is filled at runtime from the KER node's
        # metadata reply (see main()'s "ker_metadata" handling).
    elif args.operation_type == "rollout":
        if "model" not in metadata:
            metadata["model"] = {}
        if args.docker_image:
            metadata["model"]["docker_image"] = args.docker_image

    metadata["frequencies"] = {
        "action": {
            "arms": {},
        },
        "obs": {
            "arms": {},
        },
        "cameras": {},
    }
    frequency_detector = FrequencyDetector(node.dataflow_descriptor()["nodes"])
    for name, input in node.node_config()["inputs"].items():
        frequency = frequency_detector.detect(input)
        if not frequency:
            continue
        if name.startswith("arm_"):
            # arm_right_action -> right, action
            side, type = name.split("_")[1:3]
            if type == "observation":
                type = "obs"
            metadata["frequencies"][type]["arms"][side] = frequency
        elif name.startswith("camera_"):
            # camera_wrist_right -> wrist_right
            camera_name = name.removeprefix("camera_")
            metadata["frequencies"]["cameras"][camera_name] = frequency


def main():
    """Collect data and record them."""
    parser = argparse.ArgumentParser(description="Record data as OpenArm dataset")
    parser.add_argument(
        "--directory",
        default=os.getenv("DIRECTORY", os.getcwd()),
        help="The output directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--docker-image",
        default=os.getenv("DOCKER_IMAGE", os.getenv("IMAGE")),
        help="The Docker image used for this rollout",
        type=str,
    )
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="The metadata file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--name",
        default=os.getenv("NAME", "dataset"),
        help="The dataset name",
        type=str,
    )
    parser.add_argument(
        "--operation-type",
        choices=["teleop", "rollout"],
        default=os.getenv("OPERATION_TYPE", "teleop"),
        help="The operation type",
        type=str,
    )
    args = parser.parse_args()

    node = dora.Node()
    if args.metadata_file is None:
        metadata = {}
    else:
        with open(args.metadata_file, encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}
    _collect_dynamic_metadata(metadata, args, node)
    dataset_writer = DatasetWriter(args.directory, args.name, metadata)
    episode = None
    episode_writer = None
    last_action_timestamps = {}

    for event in node:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id == "command":
            command = event["value"][0].as_py()
            result_episode_id = episode.number if episode is not None else None
            should_quit = False
            try:
                payload = parse_command_payload(event["metadata"])
                if command == "start":
                    if episode is not None:
                        raise ValueError("an episode is already active")
                    candidate = Episode()
                    candidate.number = payload.get(
                        "episode_number", event["metadata"].get("episode_number", 0)
                    )
                    candidate.task_index = payload.get(
                        "task_index", event["metadata"].get("task_index", 0)
                    )
                    candidate.metadata = payload.get("episode", {})
                    if not isinstance(candidate.metadata, dict):
                        raise ValueError("episode metadata must be an object")
                    candidate_writer = dataset_writer.create_episode_writer(candidate)
                    episode = candidate
                    episode_writer = candidate_writer
                    result_episode_id = episode.number
                elif command in ("success", "fail"):
                    if episode is None:
                        raise ValueError("no episode is active")
                    episode_metadata = payload.get("episode", {})
                    if not isinstance(episode_metadata, dict):
                        raise ValueError("episode metadata must be an object")
                    episode.success = command == "success"
                    dataset_writer.finish_episode(
                        episode,
                        episode_metadata,
                        writer=episode_writer,
                    )
                    episode = None
                    episode_writer = None
                elif command == "cancel":
                    if episode is not None:
                        episode_writer.cancel()
                        episode = None
                        episode_writer = None
                elif command == "metadata":
                    dataset_writer.update_metadata(payload)
                elif command == "quit":
                    if episode is not None:
                        episode_metadata = payload.get("episode", {})
                        if not isinstance(episode_metadata, dict):
                            raise ValueError("episode metadata must be an object")
                        dataset_writer.finish_episode(
                            episode,
                            episode_metadata,
                            writer=episode_writer,
                        )
                        episode = None
                        episode_writer = None
                    should_quit = True
                else:
                    raise ValueError("unknown command")
            except Exception as error:
                print(f"Ignoring {command!r} command: {error}")
                _send_command_result(
                    node,
                    command,
                    False,
                    result_episode_id,
                    error,
                )
            else:
                _send_command_result(node, command, True, result_episode_id)
                if should_quit:
                    break
            continue

        if event_id == "ker_metadata":
            # KER leader device metadata (JSON) from the KER node.
            ker_metadata = json.loads(event["value"][0].as_py())
            dataset_writer.set_leader_ker_metadata(ker_metadata)
            continue

        if event_id in {"arm_right_action", "arm_left_action"}:
            try:
                duplicate = _is_duplicate_action(
                    event_id,
                    event["metadata"],
                    last_action_timestamps,
                )
            except ValueError as error:
                print(f"Ignoring {event_id!r} input: {error}")
                continue
            if duplicate:
                continue

        # Main process
        if episode is None:
            continue
        timestamp = event["metadata"]["timestamp"]
        if isinstance(timestamp, datetime.datetime):
            # Added by dora-rs automatically.
            # Convert to POSIX timestamp in nanosecond.
            timestamp = math.ceil(timestamp.timestamp() * 1_000_000_000)
        if event_id.startswith("arm_"):
            value = event["value"]
            if isinstance(value, pa.StructArray) and "new_position" in value.type.names:
                value = value.field("new_position")

            # arm_right_action ->
            # right_action
            key_prefix = event_id.removeprefix("arm_")
            # right_action ->
            # right_actions
            values_key = f"{key_prefix}s"
            getattr(episode, values_key).append(value)
            # right_action ->
            # right_action_timestamps
            timestamps_key = f"{key_prefix}_timestamps"
            getattr(episode, timestamps_key).append(timestamp)
        elif event_id.startswith("elevation_"):
            # elevation_observation -> elevation_observations, elevation_observation_timestamps
            # elevation_action -> elevation_actions, elevation_action_timestamps
            getattr(episode, f"{event_id}s").append(event["value"])
            getattr(episode, f"{event_id}_timestamps").append(timestamp)
        elif event_id.startswith("camera_"):
            name = event_id.removeprefix("camera_")
            image = event["value"]
            format = event["metadata"]["encoding"]
            episode_writer.write_camera_image(name, image, timestamp, format)


if __name__ == "__main__":
    main()
