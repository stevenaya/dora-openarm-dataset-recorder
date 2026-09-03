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

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dora_openarm_dataset_recorder.main import (
    DatasetWriter,
    Episode,
    _is_duplicate_action,
    parse_command_payload,
)


def _read_metadata(tmp_path):
    with (tmp_path / "dataset" / "metadata.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_runtime_metadata_and_episode_fields_are_written(tmp_path):
    writer = DatasetWriter(tmp_path, "dataset", {"tasks": [{"name": "pick"}]})
    writer.update_metadata(
        {
            "dataset": {
                "evaluation": {
                    "schema_version": "0.1.0",
                    "checkpoint": {"path": "/models/checkpoint"},
                },
                "version": "ignored",
                "episodes": ["ignored"],
            }
        }
    )

    episode = Episode(
        number=4,
        success=True,
        task_index=0,
        metadata={"source": "rollout", "experiment_config": {"seed": 1}},
    )
    writer.finish_episode(
        episode,
        {
            "note": "clean grasp",
            "experiment_config": {"temperature": 0.2},
            "id": "wrong",
            "success": False,
        },
    )

    metadata = _read_metadata(tmp_path)
    assert metadata["version"] == "0.4.0"
    assert metadata["evaluation"]["checkpoint"]["path"] == "/models/checkpoint"
    assert metadata["episodes"] == [
        {
            "source": "rollout",
            "experiment_config": {"seed": 1, "temperature": 0.2},
            "note": "clean grasp",
            "id": "4",
            "success": True,
            "task_index": 0,
        }
    ]


def test_existing_runtime_metadata_is_preserved_and_patchable(tmp_path):
    initial = {"tasks": [{"name": "pick"}]}
    writer = DatasetWriter(tmp_path, "dataset", initial)
    writer.update_metadata(
        {"dataset": {"evaluation": {"checkpoint": {"path": "first"}}}}
    )
    episode = Episode(number=2, metadata={"source": "rollout"})
    writer.finish_episode(episode)

    resumed = DatasetWriter(tmp_path, "dataset", initial)
    resumed.update_metadata(
        {"dataset": {"evaluation": {"checkpoint": {"description": "candidate A"}}}}
    )

    metadata = _read_metadata(tmp_path)
    assert metadata["evaluation"]["checkpoint"] == {
        "path": "first",
        "description": "candidate A",
    }
    assert metadata["episodes"] == [
        {
            "id": "2",
            "success": False,
            "task_index": 0,
            "source": "rollout",
        }
    ]


def test_existing_nested_runtime_fields_do_not_break_resume(tmp_path):
    """Runtime additions may extend source metadata without causing a mismatch."""
    initial = {
        "equipment": {"leader": {"id": "OpenArm"}},
        "frequencies": {"action": {"arms": {}}},
    }
    writer = DatasetWriter(tmp_path, "dataset", initial)
    writer.update_metadata(
        {
            "dataset": {
                "equipment": {"leader": {"firmware_version": "1.2.3"}},
                "frequencies": {"action": {"arms": {"right": 250.0}}},
            }
        }
    )

    resumed = DatasetWriter(tmp_path, "dataset", initial)
    resumed.update_metadata({})
    metadata = _read_metadata(tmp_path)
    assert metadata["equipment"]["leader"]["firmware_version"] == "1.2.3"
    assert metadata["frequencies"]["action"]["arms"]["right"] == 250.0


def test_historical_episode_patch_is_rejected(tmp_path):
    writer = DatasetWriter(tmp_path, "dataset", {})
    with pytest.raises(ValueError, match="no longer supported"):
        writer.update_metadata({"episodes": [{"id": "8", "note": "old"}]})
    with pytest.raises(ValueError, match="no longer supported"):
        writer.update_metadata({"episodes": []})


def test_invalid_metadata_payload_is_transactional(tmp_path):
    """Validation failures must not leak a partial dataset update into memory."""
    writer = DatasetWriter(tmp_path, "dataset", {})
    with pytest.raises(ValueError, match="must be an object"):
        writer.update_metadata(
            {
                "dataset": ["invalid"],
            }
        )

    writer.update_metadata({})
    assert "evaluation" not in _read_metadata(tmp_path)


def test_metadata_write_failure_rolls_back_memory(tmp_path, monkeypatch):
    writer = DatasetWriter(tmp_path, "dataset", {"location": "Lab"})

    def fail_write():
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_metadata_file", fail_write)
    with pytest.raises(OSError, match="disk full"):
        writer.update_metadata({"dataset": {"location": "Elsewhere"}})

    assert writer._metadata["location"] == "Lab"


def test_episode_is_published_only_when_metadata_is_committed(tmp_path):
    writer = DatasetWriter(tmp_path, "dataset", {})
    episode = Episode(number=3, success=True)
    episode_writer = writer.create_episode_writer(episode)
    partial = tmp_path / "dataset" / "episodes" / ".partial-3"
    final = tmp_path / "dataset" / "episodes" / "3"

    assert partial.is_dir()
    assert not final.exists()
    writer.finish_episode(episode, writer=episode_writer)

    assert not partial.exists()
    assert final.is_dir()
    assert _read_metadata(tmp_path)["episodes"][0]["id"] == "3"


def test_metadata_failure_rolls_published_episode_back(tmp_path, monkeypatch):
    writer = DatasetWriter(tmp_path, "dataset", {})
    episode = Episode(number=4)
    episode_writer = writer.create_episode_writer(episode)

    def fail_write():
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_metadata_file", fail_write)
    with pytest.raises(OSError, match="disk full"):
        writer.finish_episode(episode, writer=episode_writer)

    assert (tmp_path / "dataset" / "episodes" / ".partial-4").is_dir()
    assert not (tmp_path / "dataset" / "episodes" / "4").exists()
    assert writer._episode_results == []


def test_interrupted_partial_episode_is_quarantined_on_resume(tmp_path):
    writer = DatasetWriter(tmp_path, "dataset", {})
    writer.update_metadata({})
    partial = tmp_path / "dataset" / "episodes" / ".partial-9"
    partial.mkdir(parents=True)
    (partial / "frame.jpeg").write_bytes(b"frame")

    DatasetWriter(tmp_path, "dataset", {})

    assert not partial.exists()
    quarantined = list((tmp_path / "dataset" / "orphaned").glob(".partial-9.*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "frame.jpeg").read_bytes() == b"frame"


def test_action_snapshots_are_deduplicated_per_input_timestamp():
    latest = {}
    assert not _is_duplicate_action("arm_right_action", {"timestamp": 10}, latest)
    assert _is_duplicate_action("arm_right_action", {"timestamp": 10}, latest)
    assert not _is_duplicate_action("arm_left_action", {"timestamp": 10}, latest)
    assert not _is_duplicate_action("arm_right_action", {"timestamp": 11}, latest)

    with pytest.raises(ValueError, match="missing timestamp"):
        _is_duplicate_action("arm_right_action", {}, latest)


def test_policy_chunks_and_action_chunk_ids_are_written(tmp_path):
    writer = DatasetWriter(tmp_path, "dataset", {})
    episode = Episode(number=1, attempt_id="attempt-1")
    episode.right_action_timestamps = [10, 20]
    episode.right_actions = [
        pa.array([{"qpos": [1.0, 2.0]}]),
        pa.array([{"qpos": [3.0, 4.0]}]),
    ]
    episode.right_action_chunk_ids = ["chunk-1", None]
    episode.right_action_blended_chunk_ids = ["chunk-0", None]
    episode.policy_chunks = [
        {
            "chunk_id": "chunk-1",
            "episode_number": 1,
            "episode_attempt_id": "attempt-1",
            "generated_timestamp_ns": 5,
            "interval_ns": 33_333_333,
            "chunk_received": [[1.0, 2.0], [3.0, 4.0]],
        }
    ]
    episode.chunk_execution = {
        "chunk-1": {
            "executor_received_timestamp_ns": 7,
            "blended_chunk_id": "chunk-0",
            "blend_policy_points": 4,
        }
    }

    episode_writer = writer.create_episode_writer(episode)
    writer.finish_episode(episode, writer=episode_writer)

    episode_dir = tmp_path / "dataset" / "episodes" / "1"
    actions = pq.read_table(
        episode_dir / "action" / "arms" / "right" / "state.parquet"
    )
    assert actions["chunk_id"].to_pylist() == ["chunk-1", None]
    assert actions["blended_chunk_id"].to_pylist() == ["chunk-0", None]

    chunks = pq.read_table(episode_dir / "policy" / "chunks.parquet")
    assert chunks["chunk_id"].to_pylist() == ["chunk-1"]
    assert chunks["interval_ns"].to_pylist() == [33_333_333]
    assert chunks["chunk_received"].to_pylist() == [
        [[1.0, 2.0], [3.0, 4.0]]
    ]
    assert chunks["blended_chunk_id"].to_pylist() == ["chunk-0"]
    assert chunks["blend_policy_points"].to_pylist() == [4]
    assert _read_metadata(tmp_path)["episodes"][0]["episode_attempt_id"] == (
        "attempt-1"
    )


def test_parse_command_payload():
    payload = {"episode_number": 3, "episode": {"source": "intervention"}}
    assert parse_command_payload({"payload": json.dumps(payload)}) == payload
    assert parse_command_payload({}) == {}

    with pytest.raises(ValueError, match="valid JSON"):
        parse_command_payload({"payload": "{"})

    with pytest.raises(ValueError, match="JSON object"):
        parse_command_payload({"payload": "[]"})
