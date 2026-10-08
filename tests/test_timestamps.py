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

import datetime
import math
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dora_openarm_dataset_recorder.main as recorder


class FakeNode:
    def __init__(self, events):
        self.events = events

    def __iter__(self):
        return iter(self.events)

    def dataflow_descriptor(self):
        return {"nodes": []}

    def node_config(self):
        return {"inputs": {}}


@pytest.fixture
def run_events(monkeypatch, tmp_path):
    monkeypatch.delenv("METADATA_FILE", raising=False)

    def run(events, name="dataset"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "recorder",
                "--directory",
                str(tmp_path),
                "--name",
                name,
                "--operation-type",
                "teleop",
            ],
        )
        monkeypatch.setattr(recorder.dora, "Node", lambda: FakeNode(events))
        recorder.main()
        return tmp_path / name / "episodes"

    return run


@pytest.fixture
def record_event(run_events):
    def record(event_id, value, metadata):
        events = [
            {
                "type": "INPUT",
                "id": "command",
                "value": pa.array(["start"]),
                "metadata": {},
            },
            {"type": "INPUT", "id": event_id, "value": value, "metadata": metadata},
            {
                "type": "INPUT",
                "id": "command",
                "value": pa.array(["success"]),
                "metadata": {},
            },
        ]
        return run_events(events) / "0"

    return record


MESSAGE_NS = 1_700_000_000_123_456_789
OBSERVATION_NS = MESSAGE_NS - 5_000_000
MESSAGE_DATETIME = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {"timestamp": MESSAGE_NS, "observation_timestamp": OBSERVATION_NS},
            OBSERVATION_NS,
        ),
        ({"timestamp": MESSAGE_NS, "observation_timestamp": 0}, 0),
        ({"timestamp": MESSAGE_NS}, MESSAGE_NS),
        (
            {"timestamp": MESSAGE_DATETIME},
            math.ceil(MESSAGE_DATETIME.timestamp() * 1_000_000_000),
        ),
        (
            {"timestamp": MESSAGE_NS, "observation_timestamp": MESSAGE_DATETIME},
            math.ceil(MESSAGE_DATETIME.timestamp() * 1_000_000_000),
        ),
    ],
)
def test_arm_observation_timestamp(record_event, side, structured, metadata, expected):
    value = (
        pa.array([{"qpos": [0.25, 0.5]}])
        if structured
        else pa.array([0.25, 0.5], type=pa.float32())
    )
    episode = record_event(f"arm_{side}_observation", value, metadata)

    table = pq.read_table(episode / "obs" / "arms" / side / "state.parquet")
    assert table["timestamp"].cast(pa.int64()).to_pylist() == [expected]
    assert table["qpos"].to_pylist() == [[0.25, 0.5]]


@pytest.mark.parametrize(
    ("event_id", "path"),
    [
        ("arm_left_action", "action/arms/left/state.parquet"),
        ("arm_right_action", "action/arms/right/state.parquet"),
        ("elevation_action", "action/lifter/elevation.parquet"),
        ("elevation_observation", "obs/lifter/elevation.parquet"),
    ],
)
def test_other_state_inputs_keep_message_timestamp(record_event, event_id, path):
    episode = record_event(
        event_id,
        pa.array([0.25], type=pa.float32()),
        {"timestamp": MESSAGE_NS, "observation_timestamp": OBSERVATION_NS},
    )
    table = pq.read_table(episode / path)
    assert table["timestamp"].cast(pa.int64()).to_pylist() == [MESSAGE_NS]


def test_camera_keeps_message_timestamp(record_event):
    episode = record_event(
        "camera_head",
        pa.array([1, 2], type=pa.uint8()),
        {
            "timestamp": MESSAGE_NS,
            "observation_timestamp": OBSERVATION_NS,
            "encoding": "jpg",
        },
    )
    directory = episode / "cameras" / "head"
    assert [path.name for path in directory.iterdir()] == [f"{MESSAGE_NS}.jpg"]
    assert (directory / f"{MESSAGE_NS}.jpg").read_bytes() == bytes([1, 2])


def command(name, number=0):
    return {
        "type": "INPUT",
        "id": "command",
        "value": pa.array([name]),
        "metadata": {"episode_number": number},
    }


@pytest.mark.parametrize("side", ["left", "right"])
def test_commanded_position_uses_dispatch_time_and_deduplicates(run_events, side):
    events = [command("start")]
    for index, dispatch in enumerate(
        (OBSERVATION_NS, OBSERVATION_NS, OBSERVATION_NS + 1)
    ):
        events.append(
            {
                "type": "INPUT",
                "id": f"arm_{side}_action",
                "value": pa.array([{"qpos": [0.25, 0.5]}]),
                "metadata": {
                    "timestamp": MESSAGE_DATETIME
                    + datetime.timedelta(milliseconds=index),
                    "observation_timestamp": MESSAGE_NS + index,
                    "dispatch_timestamp": dispatch,
                    "chunk_id": "chunk-a",
                    "blended_chunk_id": "chunk-before",
                },
            }
        )
    events.append(command("success"))
    episode = run_events(events) / "0"
    table = pq.read_table(episode / "action" / "arms" / side / "state.parquet")
    assert table["timestamp"].cast(pa.int64()).to_pylist() == [
        OBSERVATION_NS,
        OBSERVATION_NS + 1,
    ]
    assert table["qpos"].to_pylist() == [[0.25, 0.5], [0.25, 0.5]]
    assert table["chunk_id"].to_pylist() == ["chunk-a", "chunk-a"]
    assert table["blended_chunk_id"].to_pylist() == ["chunk-before", "chunk-before"]


def test_legacy_command_keeps_source_timestamp(record_event):
    episode = record_event(
        "arm_right_action",
        pa.array([{"qpos": [0.25]}]),
        {"timestamp": MESSAGE_NS, "executed_timestamp": OBSERVATION_NS},
    )
    table = pq.read_table(episode / "action/arms/right/state.parquet")
    assert table["timestamp"].cast(pa.int64()).to_pylist() == [MESSAGE_NS]


def observation(side, snapshot=True):
    metadata = {"timestamp": MESSAGE_NS}
    if snapshot:
        metadata["observation_timestamp"] = OBSERVATION_NS
    return {
        "type": "INPUT",
        "id": f"arm_{side}_observation",
        "value": pa.array([0.25], type=pa.float32()),
        "metadata": metadata,
    }


@pytest.mark.parametrize("first_side", ["left", "right"])
@pytest.mark.parametrize("snapshot", [False, True])
def test_first_observation_locks_both_arms_before_recording(
    run_events, capsys, first_side, snapshot
):
    episodes = run_events(
        [
            observation(first_side, snapshot),
            command("start"),
            observation("left"),
            observation("right"),
            command("success"),
        ]
    )
    expected = OBSERVATION_NS if snapshot else MESSAGE_NS
    for side in ("left", "right"):
        table = pq.read_table(episodes / "0" / "obs" / "arms" / side / "state.parquet")
        assert table["timestamp"].cast(pa.int64()).to_pylist() == [expected]
    field = "observation_timestamp" if snapshot else "timestamp"
    assert capsys.readouterr().out == f"Arm observation timestamp field: {field}\n"


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("recording", [False, True])
def test_missing_locked_field_fails_even_when_not_recording(
    run_events, side, recording
):
    events = [observation("left")]
    if recording:
        events.append(command("start"))
    events.append(observation(side, snapshot=False))
    with pytest.raises(
        ValueError,
        match=f"arm_{side}_observation.*locked timestamp field 'observation_timestamp'",
    ):
        run_events(events)


@pytest.mark.parametrize("end_command", ["success", "fail", "cancel"])
@pytest.mark.parametrize("snapshot", [False, True])
def test_timestamp_selection_survives_episode_changes(
    run_events, end_command, snapshot
):
    events = [
        command("start"),
        observation("left", snapshot),
        # Camera output creates the episode directory before cancellation.
        {
            "type": "INPUT",
            "id": "camera_head",
            "value": pa.array([1, 2], type=pa.uint8()),
            "metadata": {"timestamp": MESSAGE_NS, "encoding": "jpg"},
        },
        command(end_command),
        command("start", 1),
        observation("right", snapshot=not snapshot),
        command("success"),
    ]
    if snapshot:
        with pytest.raises(
            ValueError, match="locked timestamp field 'observation_timestamp'"
        ):
            run_events(events)
    else:
        episodes = run_events(events)
        table = pq.read_table(
            episodes / "1" / "obs" / "arms" / "right" / "state.parquet"
        )
        assert table["timestamp"].cast(pa.int64()).to_pylist() == [MESSAGE_NS]


def test_new_process_can_select_a_different_field(run_events, capsys):
    run_events([observation("left", snapshot=False), command("quit")], name="first")
    episodes = run_events(
        [command("start"), observation("right"), command("success")], name="second"
    )
    table = pq.read_table(episodes / "0" / "obs" / "arms" / "right" / "state.parquet")
    assert table["timestamp"].cast(pa.int64()).to_pylist() == [OBSERVATION_NS]
    assert capsys.readouterr().out.splitlines() == [
        "Arm observation timestamp field: timestamp",
        "Arm observation timestamp field: observation_timestamp",
    ]


def test_locked_arm_field_does_not_affect_other_inputs(run_events):
    outputs = {
        "arm_left_action": "action/arms/left/state.parquet",
        "arm_right_action": "action/arms/right/state.parquet",
        "elevation_action": "action/lifter/elevation.parquet",
        "elevation_observation": "obs/lifter/elevation.parquet",
    }
    events = [observation("left"), command("start")]
    for event_id in outputs:
        events.append(
            {
                "type": "INPUT",
                "id": event_id,
                "value": pa.array([0.25], type=pa.float32()),
                "metadata": {
                    "timestamp": MESSAGE_NS,
                    "observation_timestamp": OBSERVATION_NS,
                },
            }
        )
    events.append(
        {
            "type": "INPUT",
            "id": "camera_head",
            "value": pa.array([1, 2], type=pa.uint8()),
            "metadata": {
                "timestamp": MESSAGE_NS,
                "observation_timestamp": OBSERVATION_NS,
                "encoding": "jpg",
            },
        }
    )
    events.append(command("success"))
    episode = run_events(events) / "0"
    for path in outputs.values():
        table = pq.read_table(episode / path)
        assert table["timestamp"].cast(pa.int64()).to_pylist() == [MESSAGE_NS]
    assert (episode / "cameras" / "head" / f"{MESSAGE_NS}.jpg").is_file()
