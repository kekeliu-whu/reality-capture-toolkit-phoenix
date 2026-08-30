"""Small read-only subset of the ROS1 ``rosbag`` API backed by rosbags.

This keeps the production post-processing runner usable on machines where a
full ROS1 installation is unavailable.  It deliberately implements only the
read operations used by the reconstruction pipeline.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator

from rosbags.highlevel import AnyReader


class _RosTime:
    __slots__ = ("sec", "nanosec")

    def __init__(self, sec: int, nanosec: int) -> None:
        self.sec = int(sec)
        self.nanosec = int(nanosec)

    @classmethod
    def from_nsec(cls, timestamp_ns: int) -> "_RosTime":
        sec, nanosec = divmod(int(timestamp_ns), 1_000_000_000)
        return cls(sec, nanosec)

    def to_sec(self) -> float:
        return self.sec + self.nanosec * 1.0e-9

    def to_nsec(self) -> int:
        return self.sec * 1_000_000_000 + self.nanosec

    @property
    def secs(self) -> int:
        return self.sec

    @property
    def nsecs(self) -> int:
        return self.nanosec


class _MessageProxy:
    __slots__ = ("_message",)

    def __init__(self, message: object) -> None:
        self._message = message

    def __getattr__(self, name: str) -> object:
        return _wrap(getattr(self._message, name))


def _wrap(value: object) -> object:
    message_type = getattr(value, "__msgtype__", "")
    if message_type == "builtin_interfaces/msg/Time":
        return _RosTime(getattr(value, "sec"), getattr(value, "nanosec"))
    if message_type:
        return _MessageProxy(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    return value


_TypeAndTopicInfo = namedtuple("TypeAndTopicInfo", ("types", "topics"))


class Bag:
    """Read-only compatibility wrapper for :class:`rosbag.Bag`."""

    def __init__(
        self,
        filename: str,
        mode: str = "r",
        allow_unindexed: bool = False,
        **_: object,
    ) -> None:
        del allow_unindexed
        if mode not in ("r", "rb"):
            raise NotImplementedError("rosbag_compat supports read-only bags")
        self._reader = AnyReader([Path(filename)])
        self._reader.open()
        self._closed = False

    def __enter__(self) -> "Bag":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._reader.close()
            self._closed = True

    def get_type_and_topic_info(self) -> _TypeAndTopicInfo:
        topics: dict[str, object] = {}
        types: dict[str, str] = {}
        for connection in self._reader.connections:
            topics[connection.topic] = SimpleNamespace(
                msg_type=connection.msgtype,
                message_count=connection.msgcount,
            )
            types[connection.msgtype] = connection.msgdef.data
        return _TypeAndTopicInfo(types, topics)

    def read_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, object, _RosTime]]:
        selected = set(topics) if topics is not None else None
        connections = [
            connection
            for connection in self._reader.connections
            if selected is None or connection.topic in selected
        ]
        for connection, timestamp_ns, rawdata in self._reader.messages(
            connections=connections
        ):
            message = self._reader.deserialize(rawdata, connection.msgtype)
            yield connection.topic, _wrap(message), _RosTime.from_nsec(timestamp_ns)
