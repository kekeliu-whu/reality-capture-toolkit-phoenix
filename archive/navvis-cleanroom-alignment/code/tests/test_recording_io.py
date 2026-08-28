"""Unit checks for shared rec-v4 calibration and bag discovery."""

from pathlib import Path
import tempfile

import numpy as np

from navvis_recon.recording_io import numeric_laser_bags, read_laser_poses


def test_numeric_laser_bags_uses_numeric_suffix_order():
    with tempfile.TemporaryDirectory(prefix="recording-io-test-") as directory:
        dataset = Path(directory)
        bags = dataset / "internal" / "bags"
        bags.mkdir(parents=True)
        for name in (
            "bag_laser_horiz_10.bag",
            "bag_laser_horiz_2.bag",
            "bag_laser_horiz_invalid.bag",
            "bag_laser_vert_0.bag",
        ):
            (bags / name).touch()

        result = numeric_laser_bags(dataset, "horiz")

        assert [path.name for path in result] == [
            "bag_laser_horiz_2.bag",
            "bag_laser_horiz_10.bag",
        ]


def test_read_laser_poses_returns_worker_schema():
    model = """
      <VelodyneLaserModel>
        <SensorName>{name}</SensorName>
        <Pose>
          <position><x>{x}</x><y>{y}</y><z>{z}</z></position>
          <orientation><w>1</w><x>0</x><y>0</y><z>0</z></orientation>
        </Pose>
      </VelodyneLaserModel>
    """
    xml = "<SensorFrame>" + model.format(
        name="laser_horiz", x=0, y=0, z=0
    ) + model.format(name="laser_vert", x=1, y=2, z=3) + "</SensorFrame>"
    with tempfile.TemporaryDirectory(prefix="recording-io-test-") as directory:
        path = Path(directory) / "sensor_frame.xml"
        path.write_text(xml)

        poses = read_laser_poses(path)

    assert set(poses) == {"laser_horiz", "laser_vert", "laser_vert_box"}
    for pose in poses.values():
        values = np.asarray([float(value) for value in pose.split(",")])
        assert values.shape == (7,)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(np.linalg.norm(values[3:]), 1.0)
    assert poses["laser_vert_box"] == "1.0,2.0,3.0,0.0,0.0,0.0,1.0"


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().copy().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
