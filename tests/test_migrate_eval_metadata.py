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

import yaml

from dora_openarm_dataset_recorder.migrate_eval_metadata import migrate


def _write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_split_evaluation_metadata_is_merged_and_archived(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    metadata_path = dataset_dir / "metadata.yaml"
    legacy_path = dataset_dir / "eval_metadata.yaml"
    _write_yaml(
        metadata_path,
        {
            "version": "0.4.0",
            "operation_type": "rollout",
            "tasks": [{"prompt": "Pick"}],
            "episodes": [{"id": "0", "success": False, "task_index": 0}],
        },
    )
    _write_yaml(
        legacy_path,
        {
            "version": "0.1.0",
            "tasks": [{"prompt": "Pick"}],
            "checkpoint": {"path": "/models/step-1"},
            "episodes": [
                {
                    "id": "0",
                    "success": True,
                    "task_index": 0,
                    "note": "legacy note",
                    "timestamp": "2026-01-01T00:00:00",
                }
            ],
        },
    )

    messages = migrate(dataset_dir)
    migrated = _read_yaml(metadata_path)

    assert migrated["evaluation"] == {
        "schema_version": "0.1.0",
        "checkpoint": {"path": "/models/step-1"},
    }
    assert migrated["episodes"] == [
        {
            "id": "0",
            "success": False,
            "task_index": 0,
            "note": "legacy note",
            "timestamp": "2026-01-01T00:00:00",
            "source": "rollout",
        }
    ]
    assert not legacy_path.exists()
    assert len(list(dataset_dir.glob("metadata.yaml.bak.*"))) == 1
    assert len(list(dataset_dir.glob("eval_metadata.yaml.migrated.*"))) == 1
    assert any("success differs" in message for message in messages)


def test_old_rollout_episodes_gain_source_without_eval_metadata(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    metadata_path = dataset_dir / "metadata.yaml"
    _write_yaml(
        metadata_path,
        {
            "operation_type": "rollout",
            "episodes": [{"id": "2", "success": True, "task_index": 1}],
        },
    )

    migrate(dataset_dir)

    assert _read_yaml(metadata_path)["episodes"][0]["source"] == "rollout"


def test_empty_legacy_file_is_archived(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    metadata_path = dataset_dir / "metadata.yaml"
    legacy_path = dataset_dir / "eval_metadata.yaml"
    _write_yaml(metadata_path, {"episodes": []})
    legacy_path.write_text("", encoding="utf-8")

    migrate(dataset_dir)

    assert not legacy_path.exists()
    assert len(list(dataset_dir.glob("eval_metadata.yaml.migrated.*"))) == 1


def test_dry_run_does_not_change_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    metadata_path = dataset_dir / "metadata.yaml"
    original = {
        "operation_type": "rollout",
        "episodes": [{"id": "0", "success": True, "task_index": 0}],
    }
    _write_yaml(metadata_path, original)

    messages = migrate(dataset_dir, dry_run=True)

    assert _read_yaml(metadata_path) == original
    assert not list(dataset_dir.glob("metadata.yaml.bak.*"))
    assert messages[-1] == "Dry run only; no files were changed."


def test_legacy_only_result_without_episode_directory_is_skipped(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_yaml(dataset_dir / "metadata.yaml", {"episodes": []})
    _write_yaml(
        dataset_dir / "eval_metadata.yaml",
        {"episodes": [{"id": "8", "success": True, "task_index": 0, "note": "stale"}]},
    )

    messages = migrate(dataset_dir)

    assert _read_yaml(dataset_dir / "metadata.yaml")["episodes"] == []
    assert any("has no episode directory" in message for message in messages)
