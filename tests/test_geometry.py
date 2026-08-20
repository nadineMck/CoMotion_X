import pytest

from comotion_x.safety.geometry import Capsule, Sphere, primitive_distance, segment_distance


def test_crossing_segments_have_zero_centerline_distance() -> None:
    distance = segment_distance(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    assert distance == pytest.approx(0.0)


def test_parallel_segments_have_expected_distance() -> None:
    distance = segment_distance(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (1.0, 0.5, 0.0),
    )

    assert distance == pytest.approx(0.5)


def test_primitive_clearance_accounts_for_radii() -> None:
    sphere = Sphere("hand", (0.0, 0.0, 0.0), 0.1)
    capsule = Capsule("arm", (0.5, -1.0, 0.0), (0.5, 1.0, 0.0), 0.2)

    distance = primitive_distance(sphere, capsule)

    assert distance.centerline_distance_m == pytest.approx(0.5)
    assert distance.clearance_m == pytest.approx(0.2)
    assert not distance.overlaps


def test_overlapping_spheres_have_negative_clearance() -> None:
    first = Sphere("first", (0.0, 0.0, 0.0), 0.2)
    second = Sphere("second", (0.25, 0.0, 0.0), 0.1)

    distance = primitive_distance(first, second)

    assert distance.clearance_m == pytest.approx(-0.05)
    assert distance.overlaps

