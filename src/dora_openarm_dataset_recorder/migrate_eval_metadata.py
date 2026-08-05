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

"""One-time migration from split evaluation metadata to dataset metadata."""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import pathlib
import shutil

import yaml


def _read_mapping(path: pathlib.Path, *, required: bool = False) -> dict:
    if not path.exists():
        if required:
            raise ValueError(f"Metadata file does not exist: {path}")
        return {}
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a mapping: {path}")
    return data


def _deep_merge(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _episode_map(episodes, source: pathlib.Path) -> tuple[list[str], dict[str, dict]]:
    if not isinstance(episodes, list):
        raise ValueError(f"'episodes' must be a list: {source}")
    order = []
    by_id = {}
    for episode in episodes:
        if not isinstance(episode, dict) or "id" not in episode:
            raise ValueError(f"Every episode must be an object with an id: {source}")
        episode_id = str(episode["id"])
        if episode_id in by_id:
            raise ValueError(f"Duplicate episode id {episode_id!r}: {source}")
        order.append(episode_id)
        by_id[episode_id] = copy.deepcopy(episode)
    return order, by_id


def build_migrated_metadata(
    dataset_dir: pathlib.Path,
    metadata: dict,
    legacy: dict,
) -> tuple[dict, list[str]]:
    """Build unified metadata while keeping recorder identities authoritative."""
    metadata_path = dataset_dir / "metadata.yaml"
    legacy_path = dataset_dir / "eval_metadata.yaml"
    recorded_order, recorded_by_id = _episode_map(
        metadata.get("episodes", []), metadata_path
    )
    legacy_order, legacy_by_id = _episode_map(legacy.get("episodes", []), legacy_path)
    messages = []

    merged = copy.deepcopy(metadata)
    legacy_tasks = legacy.get("tasks")
    if legacy_tasks:
        if merged.get("tasks") and merged["tasks"] != legacy_tasks:
            messages.append("tasks differ; keeping metadata.yaml tasks")
        else:
            merged["tasks"] = copy.deepcopy(legacy_tasks)

    if legacy:
        evaluation = {
            "schema_version": str(legacy.get("version", "0.1.0")),
            "checkpoint": copy.deepcopy(legacy.get("checkpoint", {})),
        }
        _deep_merge(evaluation, merged.get("evaluation", {}))
        merged["evaluation"] = evaluation

    is_rollout = legacy_path.exists() or merged.get("operation_type") == "rollout"
    merged_episodes = []
    for episode_id in recorded_order:
        recorded_episode = recorded_by_id[episode_id]
        legacy_episode = legacy_by_id.pop(episode_id, {})
        for field in ("success", "task_index"):
            if (
                field in legacy_episode
                and field in recorded_episode
                and legacy_episode[field] != recorded_episode[field]
            ):
                messages.append(
                    f"episode {episode_id} {field} differs; keeping metadata.yaml value"
                )
        result = copy.deepcopy(legacy_episode)
        _deep_merge(result, recorded_episode)
        result["id"] = episode_id
        if is_rollout:
            result.setdefault("source", "rollout")
        merged_episodes.append(result)

    for episode_id in legacy_order:
        legacy_episode = legacy_by_id.get(episode_id)
        if legacy_episode is None:
            continue
        episode_directory = dataset_dir / "episodes" / episode_id
        if not episode_directory.exists():
            messages.append(
                f"episode {episode_id} exists only in eval_metadata.yaml and has no "
                "episode directory; skipping it"
            )
            continue
        if "success" not in legacy_episode or "task_index" not in legacy_episode:
            messages.append(
                f"episode {episode_id} cannot be recovered without success and "
                "task_index; skipping it"
            )
            continue
        recovered = copy.deepcopy(legacy_episode)
        recovered["id"] = episode_id
        if is_rollout:
            recovered.setdefault("source", "rollout")
        merged_episodes.append(recovered)
        messages.append(
            f"episode {episode_id} recovered from eval_metadata.yaml and its directory"
        )

    merged["episodes"] = merged_episodes
    return merged, messages


def migrate(dataset_dir: pathlib.Path, *, dry_run: bool = False) -> list[str]:
    """Migrate one dataset and return informational messages."""
    dataset_dir = dataset_dir.resolve()
    metadata_path = dataset_dir / "metadata.yaml"
    legacy_path = dataset_dir / "eval_metadata.yaml"
    legacy_exists = legacy_path.exists()
    metadata = _read_mapping(metadata_path, required=True)
    legacy = _read_mapping(legacy_path)

    missing_sources = [
        str(episode.get("id", "?"))
        for episode in metadata.get("episodes", [])
        if isinstance(episode, dict) and "source" not in episode
    ]
    if not legacy_exists and not (
        metadata.get("operation_type") == "rollout" and missing_sources
    ):
        return ["No legacy evaluation metadata found; nothing to migrate."]

    migrated, messages = build_migrated_metadata(dataset_dir, metadata, legacy)
    if dry_run:
        messages.append("Dry run only; no files were changed.")
        return messages

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    metadata_backup = metadata_path.with_name(f"metadata.yaml.bak.{timestamp}")
    shutil.copy2(metadata_path, metadata_backup)

    temporary_path = metadata_path.with_suffix(".yaml.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            migrated,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    os.replace(temporary_path, metadata_path)
    messages.append(f"Backup: {metadata_backup}")

    if legacy_path.exists():
        legacy_backup = legacy_path.with_name(
            f"eval_metadata.yaml.migrated.{timestamp}"
        )
        os.replace(legacy_path, legacy_backup)
        messages.append(f"Archived legacy metadata: {legacy_backup}")
    messages.append(f"Updated: {metadata_path}")
    return messages


def main() -> None:
    """Run the metadata migration command."""
    parser = argparse.ArgumentParser(
        description="Merge legacy eval_metadata.yaml into metadata.yaml"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=pathlib.Path,
        help="Dataset directory containing metadata.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration decisions without changing files",
    )
    args = parser.parse_args()
    try:
        messages = migrate(args.dataset_dir, dry_run=args.dry_run)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from error
    for message in messages:
        print(message)


if __name__ == "__main__":
    main()
