"""NavVis-compatible floor estimation from a post-processing trace.

The implementation mirrors the observable behaviour of the installed Cython
``floor_estimator`` package. In particular, floors are tracked sequentially;
they are not global height clusters. That distinction matters for stairwells
and for trajectories which revisit a floor several times.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CLUSTER_BIN_SIZE = 0.1
STANDARD_FLOOR_HEIGHT = 3.0
MAX_FLOOR_HEIGHT = 4.0
MIN_FLOOR_HEIGHT = 2.1
TOLERANCE = 0.03
MAX_TIME_RANGE_GAP_NS = 195_000_000
VALIDATION_PERIOD_NS = 100_000_000
TRACE_MINIMUM_INTERVAL_NS = 100_000_001
SMALL_SEGMENT_NS = 5_000_000_000
NO_FLOOR_OVERLAP_TOLERANCE = 0.12


@dataclass(frozen=True, slots=True)
class TraceSample:
    timestamp_ns: int
    x: float
    y: float
    z: float


@dataclass(slots=True)
class TimeRange:
    start_ns: int
    end_ns: int


@dataclass(slots=True)
class Floor:
    index: int
    hard_z_min: float = -math.inf
    hard_z_max: float = math.inf
    z_min: float = math.inf
    z_max: float = -math.inf
    samples: list[TraceSample] = field(default_factory=list)
    time_ranges: list[TimeRange] = field(default_factory=list)
    tolerance: float = TOLERANCE
    _active_range: TimeRange | None = field(default=None, repr=False)

    @property
    def center(self) -> float:
        if not self.samples:
            return math.nan
        values = sorted(sample.z for sample in self.samples)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return 0.5 * (values[middle - 1] + values[middle])

    @property
    def total_duration_ns(self) -> int:
        return sum(item.end_ns - item.start_ns for item in self.time_ranges)

    def accepts(self, z: float, max_z_diff: float = STANDARD_FLOOR_HEIGHT) -> bool:
        """Return the Cython ``Floor.can_add_z`` decision."""
        if not math.isfinite(z) or z < self.hard_z_min or z > self.hard_z_max:
            return False
        if not math.isfinite(self.z_min) or not math.isfinite(self.z_max):
            return True
        potential_size = max(self.z_max, z) - min(self.z_min, z)
        return (
            potential_size < MAX_FLOOR_HEIGHT
            and (
                potential_size < max_z_diff
                or self.z_min - self.tolerance <= z <= self.z_max + self.tolerance
            )
        )

    def add_z(self, sample: TraceSample) -> None:
        if not self.accepts(sample.z):
            raise ValueError(
                f"sample z={sample.z!r} is outside floor {self.index} limits "
                f"[{self.hard_z_min!r}, {self.hard_z_max!r}]"
            )
        self.samples.append(sample)
        self.z_min = min(self.z_min, sample.z)
        self.z_max = max(self.z_max, sample.z)

    def start_time_range(self, timestamp_ns: int) -> None:
        if self._active_range is not None:
            raise RuntimeError(f"floor {self.index} already has an active time range")
        self._active_range = TimeRange(timestamp_ns, timestamp_ns)

    def extend_time_range(self, timestamp_ns: int) -> None:
        if self._active_range is None:
            raise RuntimeError(f"floor {self.index} has no active time range")
        if timestamp_ns < self._active_range.end_ns:
            raise ValueError("floor timestamps must be non-decreasing")
        self._active_range.end_ns = timestamp_ns

    def finish_time_range(self) -> None:
        if self._active_range is not None:
            self.time_ranges.append(self._active_range)
            self._active_range = None


def read_trace_csv(path: str | Path) -> list[TraceSample]:
    """Read the four fields used by Floor from an official ``trace.csv``."""
    trace_path = Path(path)
    samples: list[TraceSample] = []
    with trace_path.open(newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty trace file: {trace_path}") from error
        if not header or header[0].strip() != "nsecs":
            raise ValueError(f"unsupported trace header in {trace_path}: {header!r}")
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"trace row {line_number} has fewer than four fields")
            try:
                sample = TraceSample(int(row[0]), float(row[1]), float(row[2]), float(row[3]))
            except ValueError as error:
                raise ValueError(f"invalid trace row {line_number}: {row!r}") from error
            if not all(math.isfinite(value) for value in (sample.x, sample.y, sample.z)):
                raise ValueError(f"non-finite trace coordinate at row {line_number}")
            if samples and sample.timestamp_ns <= samples[-1].timestamp_ns:
                raise ValueError(f"trace timestamps are not strictly increasing at row {line_number}")
            samples.append(sample)
    if not samples:
        raise ValueError(f"trace has no usable samples: {trace_path}")
    return samples


def _validate_samples(samples: Iterable[TraceSample]) -> list[TraceSample]:
    result = list(samples)
    if not result:
        raise ValueError("floor estimation requires at least one trace sample")
    for index, sample in enumerate(result):
        if not all(math.isfinite(value) for value in (sample.x, sample.y, sample.z)):
            raise ValueError(f"trace sample {index} has a non-finite coordinate")
        if index and sample.timestamp_ns <= result[index - 1].timestamp_ns:
            raise ValueError(f"trace sample timestamps are not strictly increasing at index {index}")
    return result


def _run_state_machine(
    trace: list[TraceSample], seed_floors: Iterable[Floor] = ()
) -> list[Floor]:
    floors = list(seed_floors)
    current: Floor | None = None

    for sample in trace:
        if current is not None and current.accepts(sample.z):
            current.add_z(sample)
            current.extend_time_range(sample.timestamp_ns)
            continue

        if current is not None:
            current.finish_time_range()

        target = next((floor for floor in floors if floor.accepts(sample.z)), None)
        if target is None:
            below = [floor.z_max for floor in floors if floor.z_max < sample.z]
            above = [floor.z_min for floor in floors if floor.z_min > sample.z]
            target = Floor(
                len(floors),
                hard_z_min=max(below, default=-math.inf),
                hard_z_max=min(above, default=math.inf),
            )
            floors.append(target)

        target.start_time_range(sample.timestamp_ns)
        target.add_z(sample)
        current = target

    if current is not None:
        current.finish_time_range()
    floors = [floor for floor in floors if floor.samples]
    floors.sort(key=lambda floor: floor.z_min)
    for index, floor in enumerate(floors):
        floor.index = index
    return floors


def simple_floor_estimator(samples: Iterable[TraceSample]) -> list[Floor]:
    """Run the recovered current/previous-floor state machine."""
    return _run_state_machine(_validate_samples(samples))


def _bin_index(z: float) -> int:
    # Cython's ``int(z / size)`` truncates towards zero. ``floor`` is wrong
    # for negative trajectories and changes split decisions.
    return int(z / CLUSTER_BIN_SIZE)


def _height_histogram(samples: list[TraceSample]) -> dict[int, list[TraceSample]]:
    bins: dict[int, list[TraceSample]] = {}
    for sample in samples:
        bins.setdefault(_bin_index(sample.z), []).append(sample)
    return bins


@dataclass(frozen=True, slots=True)
class _HistogramBin:
    average: float
    count: int
    probability: float


def _incremental_mean(values: Iterable[float]) -> float:
    average = 0.0
    count = 0
    for value in values:
        average = (average * count + value) / (count + 1)
        count += 1
    if count == 0:
        raise ValueError("cannot average an empty sequence")
    return average


def _floor_split_value(floor: Floor) -> float | None:
    """Return the Cython ``contains_two_floors`` split value, if any."""
    histogram = _height_histogram(floor.samples)
    if not histogram:
        return None
    total = len(floor.samples)
    bins = [
        _HistogramBin(
            _incremental_mean(sample.z for sample in values),
            len(values),
            len(values) / total,
        )
        for values in histogram.values()
    ]
    # FloorZHistogram.get_sorted_bin_list orders by descending probability.
    # Python's stable sort preserves first-occurrence order for ties, as does
    # the reference implementation's dict/list path.
    bins.sort(key=lambda item: item.probability, reverse=True)
    bin_count = len(bins)
    clusters = [
        index
        for index, item in enumerate(bins)
        if item.probability > 1.0 / 3.0 or item.probability * bin_count > 3.0
    ]
    if len(clusters) < 2:
        return None

    combinations = [
        (left, right)
        for offset, left in enumerate(clusters)
        for right in clusters[offset + 1 :]
    ]
    differences = [
        abs(bins[left].average - bins[right].average) for left, right in combinations
    ]
    max_index = max(range(len(differences)), key=differences.__getitem__)
    if differences[max_index] < MIN_FLOOR_HEIGHT:
        return None
    left, right = combinations[max_index]
    smaller_z = min(bins[left].average, bins[right].average)
    larger_z = max(bins[left].average, bins[right].average)
    valley = [
        item
        for item in bins
        if item.average >= smaller_z + 2.0 * CLUSTER_BIN_SIZE
        and item.average
        <= larger_z - (MIN_FLOOR_HEIGHT - 4.0 * CLUSTER_BIN_SIZE)
    ]
    if len(valley) < 2:
        return None
    valley.sort(key=lambda item: item.probability)
    return statistics.mean((valley[0].average, valley[1].average))


def _merged_time_ranges(floors: Iterable[Floor]) -> list[TimeRange]:
    ranges = sorted(
        (
            TimeRange(item.start_ns, item.end_ns)
            for floor in floors
            for item in floor.time_ranges
        ),
        key=lambda item: item.start_ns,
    )
    if not ranges:
        return []
    merged = [ranges[0]]
    for item in ranges[1:]:
        if item.start_ns - merged[-1].end_ns < MAX_TIME_RANGE_GAP_NS:
            merged[-1].end_ns = max(merged[-1].end_ns, item.end_ns)
        else:
            merged.append(item)
    return merged


def _merge_floor_pair(first: Floor, second: Floor) -> Floor:
    samples = sorted(first.samples + second.samples, key=lambda sample: sample.timestamp_ns)
    merged = Floor(
        0,
        min(first.hard_z_min, second.hard_z_min),
        max(first.hard_z_max, second.hard_z_max),
        min(first.z_min, second.z_min),
        max(first.z_max, second.z_max),
        samples,
        _merged_time_ranges((first, second)),
    )
    return merged


def _is_tiny(floor: Floor) -> bool:
    return (
        floor.z_max - floor.z_min < 2.0 * TOLERANCE
        or floor.total_duration_ns < SMALL_SEGMENT_NS
    )


def _merge_tiny_floors(floors: list[Floor]) -> list[Floor]:
    result = sorted(floors, key=lambda floor: floor.z_min)
    while len(result) > 1:
        tiny_index = next((index for index, floor in enumerate(result) if _is_tiny(floor)), None)
        if tiny_index is None:
            break
        neighbor_indices = [
            index
            for index in (tiny_index - 1, tiny_index + 1)
            if 0 <= index < len(result)
        ]
        tiny = result[tiny_index]

        def neighbor_key(index: int) -> tuple[float, int, float]:
            neighbor = result[index]
            if index < tiny_index:
                distance = max(0.0, tiny.z_min - neighbor.z_max)
            else:
                distance = max(0.0, neighbor.z_min - tiny.z_max)
            return (distance, -neighbor.total_duration_ns, neighbor.z_min)

        neighbor_index = min(neighbor_indices, key=neighbor_key)
        low, high = sorted((tiny_index, neighbor_index))
        merged = _merge_floor_pair(result[low], result[high])
        result[low : high + 1] = [merged]
    return result


def _histogram_bins(floor: Floor) -> list[_HistogramBin]:
    histogram = _height_histogram(floor.samples)
    total = len(floor.samples)
    return [
        _HistogramBin(
            _incremental_mean(sample.z for sample in values),
            len(values),
            len(values) / total,
        )
        for values in histogram.values()
    ]


def _is_cluster(item: _HistogramBin, bin_count: int) -> bool:
    return item.probability > 1.0 / 3.0 or item.probability * bin_count > 3.0


def _boundary_has_cluster(floor: Floor, boundary: float) -> bool:
    histogram = _height_histogram(floor.samples)
    boundary_index = _bin_index(boundary)
    candidates: list[_HistogramBin] = []
    for index in (boundary_index, boundary_index + 1):
        values = histogram.get(index)
        if values:
            candidates.append(
                _HistogramBin(
                    _incremental_mean(sample.z for sample in values),
                    len(values),
                    len(values) / len(floor.samples),
                )
            )
    if not candidates:
        return False
    # FloorZHistogram.get_bigger_cluster compares the two bins straddling the
    # boundary and validates only the more populated one.
    cluster = max(candidates, key=lambda item: item.count)
    return _is_cluster(cluster, len(histogram))


def _time_ranges_do_not_overlap(first: Floor, second: Floor) -> bool:
    return all(
        left.end_ns <= right.start_ns or right.end_ns <= left.start_ns
        for left in first.time_ranges
        for right in second.time_ranges
    )


def _should_merge_neighbors(below: Floor, above: Floor) -> bool:
    gap = above.z_min - below.z_max
    # The native validator permits substantial vertical overlap here.  The
    # decisive guards are the two boundary clusters and disjoint visits; the
    # upper gap limit only rules out genuinely separate levels.
    if gap >= 2.0 * CLUSTER_BIN_SIZE:
        return False
    if not _time_ranges_do_not_overlap(below, above):
        return False
    boundary = statistics.mean((below.z_max, above.z_min))
    return _boundary_has_cluster(below, boundary) and _boundary_has_cluster(
        above, boundary
    )


def _merge_adjacent_floors(floors: list[Floor]) -> list[Floor]:
    result = sorted(floors, key=lambda floor: floor.z_min)
    changed = True
    while changed:
        changed = False
        for index in range(len(result) - 1):
            if _should_merge_neighbors(result[index], result[index + 1]):
                result[index : index + 2] = [
                    _merge_floor_pair(result[index], result[index + 1])
                ]
                changed = True
                break
    return result


def _clone_floor_boundaries(floor: Floor) -> Floor:
    """Mirror ``Floor.Create_clone_boundaries`` used during a split replay."""
    return Floor(
        0,
        floor.hard_z_min,
        floor.hard_z_max,
        floor.z_min,
        floor.z_max,
        tolerance=0.0,
    )


def _split_seed_floors(
    floors: list[Floor], split_values: dict[int, float]
) -> list[Floor]:
    seeds: list[Floor] = []
    for index, floor in enumerate(floors):
        boundary = split_values.get(index)
        if boundary is None:
            seeds.append(_clone_floor_boundaries(floor))
            continue
        # Split floors deliberately receive local limits rather than the old
        # floor's limits.  Both sides overlap by 1.5 cm and use zero dynamic
        # tolerance while the complete trace is replayed.
        seeds.extend(
            (
                Floor(
                    0,
                    floor.z_min - TOLERANCE / 2.0,
                    boundary + TOLERANCE / 2.0,
                    floor.z_min,
                    boundary,
                    tolerance=0.0,
                ),
                Floor(
                    0,
                    boundary - TOLERANCE / 2.0,
                    floor.z_max + TOLERANCE / 2.0,
                    boundary,
                    floor.z_max,
                    tolerance=0.0,
                ),
            )
        )
    return seeds


def _split_double_floors(
    floors: list[Floor], trace: list[TraceSample]
) -> list[Floor]:
    result = list(floors)
    while True:
        split_values = {
            index: boundary
            for index, floor in enumerate(result)
            if (boundary := _floor_split_value(floor)) is not None
        }
        if not split_values:
            return result
        seeds = _split_seed_floors(result, split_values)
        next_result = _run_state_machine(trace, seeds)
        if len(next_result) <= len(result):
            return result
        result = next_result


def refined_floor_estimator(samples: Iterable[TraceSample]) -> list[Floor]:
    """Run the recovered four-pass FloorRefiner sequence."""
    trace = _validate_samples(samples)
    floors = _run_state_machine(trace)
    floors = _merge_tiny_floors(floors)
    floors = _merge_adjacent_floors(floors)
    floors = _split_double_floors(floors, trace)
    floors = _merge_tiny_floors(floors)
    floors.sort(key=lambda floor: floor.z_min)
    for index, floor in enumerate(floors):
        floor.index = index
    return floors


def to_official_json(floors: Iterable[Floor]) -> list[dict[str, object]]:
    """Serialize the public schema of ``artifacts/floors.json``."""
    result: list[dict[str, object]] = []
    for floor in floors:
        if floor._active_range is not None:
            raise RuntimeError(f"floor {floor.index} still has an active time range")
        if not floor.samples or not floor.time_ranges:
            raise ValueError(f"floor {floor.index} is empty")
        if any(item.start_ns > item.end_ns for item in floor.time_ranges):
            raise ValueError(f"floor {floor.index} has an invalid time range")
        result.append(
            {
                "time_ranges": [[item.start_ns, item.end_ns] for item in floor.time_ranges],
                "z_min": floor.z_min,
                "z_max": floor.z_max,
            }
        )
    if not result:
        raise ValueError("floor estimation produced no floors")
    return result


def to_json(floors: Iterable[Floor]) -> list[dict[str, object]]:
    """Backward-compatible name for the official serializer."""
    return to_official_json(floors)
