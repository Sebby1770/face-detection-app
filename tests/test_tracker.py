from detectors import Detection
from tracker import IoUTracker


def detection(x=10, y=10, width=20, height=20, score=0.9):
    return Detection(x, y, width, height, score)


def test_new_track_is_visible_on_its_first_frame():
    tracker = IoUTracker()

    tracks = tracker.update([detection()])

    assert len(tracks) == 1
    assert tracks[0].id == 1
    assert tracks[0].age == 1
    assert tracks[0].misses == 0


def test_matching_detection_keeps_id_and_resets_misses():
    tracker = IoUTracker(iou_threshold=0.2)
    original = tracker.update([detection()])[0]
    tracker.update([])

    matched = tracker.update([detection(x=12, y=11)])[0]

    assert matched.id == original.id
    assert matched.misses == 0
    assert matched.age == 3


def test_stale_tracks_expire_only_after_configured_misses():
    tracker = IoUTracker(max_misses=1)
    tracker.update([detection()])

    assert tracker.update([])[0].misses == 1
    assert tracker.update([]) == []


def test_reset_clears_tracks_and_restarts_ids():
    tracker = IoUTracker()
    tracker.update([detection()])
    tracker.reset()

    tracks = tracker.update([detection(x=50)])

    assert len(tracks) == 1
    assert tracks[0].id == 1
