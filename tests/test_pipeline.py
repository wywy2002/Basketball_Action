import unittest

from basketball_pose.keypoints import coco17_to_h36m17
from basketball_pose.court import point_in_normalized_polygon, pose_foot_point
from basketball_pose.quality import inspect_track
from basketball_pose.roles import UniformRules, classify_uniform
from basketball_pose.schema import Pose2DRecord
from basketball_pose.tracking import OcclusionAwareTracker, PoseDetection, mark_overlaps
from basketball_pose.tracks import group_tracks, interpolate_short_gaps


def make_record(frame_id: int, x_offset: float = 0.0) -> Pose2DRecord:
    keypoints = tuple((20.0 + x_offset + i, 30.0 + i, 0.9) for i in range(17))
    return Pose2DRecord(
        video_id="video_1",
        shot_id="shot_1",
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        track_id="7",
        image_size=(1920, 1080),
        bbox_xyxy=(0.0, 0.0, 200.0, 300.0),
        keypoints_2d=keypoints,
    )


class PipelineTests(unittest.TestCase):
    def test_schema_round_trip(self) -> None:
        record = make_record(3)
        self.assertEqual(Pose2DRecord.from_dict(record.to_dict()), record)

    def test_schema_rejects_wrong_keypoint_count(self) -> None:
        data = make_record(0).to_dict()
        data["keypoints_2d"] = data["keypoints_2d"][:-1]
        with self.assertRaises(ValueError):
            Pose2DRecord.from_dict(data)

    def test_group_tracks_rejects_duplicate_frames(self) -> None:
        with self.assertRaises(ValueError):
            group_tracks([make_record(1), make_record(1)])

    def test_interpolates_short_frame_gap(self) -> None:
        result = interpolate_short_gaps([make_record(0), make_record(2, 2.0)], 1)
        self.assertEqual([item.frame_id for item in result], [0, 1, 2])
        self.assertAlmostEqual(result[1].keypoints_2d[0][0], 21.0)

    def test_does_not_interpolate_long_gap(self) -> None:
        result = interpolate_short_gaps([make_record(0), make_record(4)], 2)
        self.assertEqual([item.frame_id for item in result], [0, 4])

    def test_coco_to_h36m_mapping(self) -> None:
        result = coco17_to_h36m17(make_record(0).keypoints_2d)
        self.assertEqual(len(result), 17)
        self.assertEqual(result[1], make_record(0).keypoints_2d[12])
        self.assertEqual(result[4], make_record(0).keypoints_2d[11])

    def test_quality_detects_large_jump(self) -> None:
        issues = inspect_track([make_record(0), make_record(1, 180.0)], jump_threshold=0.2)
        self.assertTrue(any(issue["type"] == "keypoint_jump" for issue in issues))

    def test_quality_handles_fully_occluded_frames(self) -> None:
        hidden = make_record(1).to_dict()
        hidden["keypoints_2d"] = [[x, y, 0.0] for x, y, _ in hidden["keypoints_2d"]]
        issues = inspect_track([make_record(0), Pose2DRecord.from_dict(hidden)])
        self.assertTrue(any(issue["type"] == "low_confidence" for issue in issues))

    def test_court_filter_uses_ankles(self) -> None:
        record = make_record(0)
        foot = pose_foot_point(list(record.keypoints_2d), record.bbox_xyxy)
        polygon = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]
        self.assertTrue(point_in_normalized_polygon(foot, record.image_size, polygon))

    def test_referee_rule_is_explicit(self) -> None:
        rules = UniformRules()
        self.assertEqual(classify_uniform((0.0, 20.0, 100.0), rules), "referee_candidate")
        self.assertEqual(classify_uniform((0.0, 20.0, 160.0), rules), "team_light")

    def test_tracker_keeps_two_overlapping_people_separate(self) -> None:
        def detection(x: float) -> PoseDetection:
            record = make_record(0, x)
            return PoseDetection(
                bbox_xyxy=(x, 0.0, x + 100.0, 200.0),
                keypoints=record.keypoints_2d,
                score=0.9,
            )

        tracker = OcclusionAwareTracker(max_missed=2)
        first_ids = tracker.update([detection(0.0), detection(60.0)])
        second_ids = tracker.update([detection(5.0), detection(65.0)])
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(set(second_ids)), 2)
        self.assertEqual(mark_overlaps([detection(0.0), detection(60.0)]), [True, True])


if __name__ == "__main__":
    unittest.main()
