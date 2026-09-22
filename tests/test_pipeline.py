import unittest

from basketball_pose.keypoints import coco17_to_h36m17
from basketball_pose.postprocess import (
    assign_appearance_segments,
    annotate_roi_tracks,
    cluster_two_teams,
    hsv_team_feature,
    mark_short_stationary_boundary_tracks,
    mark_single_frame_uncertain,
)
from basketball_pose.auto_config import shot_for_frame
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

    def test_tracker_does_not_learn_contaminated_appearance_during_overlap(self) -> None:
        def detection(x: float, hue: float, overlap: bool = False) -> PoseDetection:
            record = make_record(0, x)
            return PoseDetection(
                bbox_xyxy=(x, 0.0, x + 100.0, 200.0),
                keypoints=record.keypoints_2d,
                score=0.9,
                appearance_hsv=(hue, 200.0, 200.0),
                metadata={"overlap": overlap},
            )

        tracker = OcclusionAwareTracker(max_missed=2)
        ids = tracker.update([detection(0.0, 20.0), detection(140.0, 110.0)])
        tracker.update([detection(15.0, 20.0, True), detection(115.0, 20.0, True)])
        self.assertEqual(tracker.tracks[ids[0]].appearance_hsv[0], 20.0)
        self.assertEqual(tracker.tracks[ids[1]].appearance_hsv[0], 110.0)

    def test_roi_track_recovers_short_edge_excursion(self) -> None:
        records = [
            {"track_id": "4", "frame_id": 0, "on_court": True},
            {"track_id": "4", "frame_id": 1, "on_court": True},
            {"track_id": "4", "frame_id": 2, "on_court": False},
        ]
        annotate_roi_tracks(records, max_edge_gap=3)
        self.assertEqual(records[2]["roi_status"], "outside_recovered")
        self.assertTrue(records[2]["edge_track_recovery"])

    def test_roi_track_rejects_single_frame_outside_object(self) -> None:
        records = [{"track_id": "9", "frame_id": 5, "on_court": False}]
        annotate_roi_tracks(records)
        self.assertEqual(records[0]["rejection_reason"], "single_frame_outside_track")
        self.assertFalse(records[0]["keep_after_roi"])

    def test_single_frame_inside_track_is_audited_as_uncertain(self) -> None:
        records = [{"track_id": "9", "frame_id": 5, "on_court": True}]
        annotate_roi_tracks(records)
        mark_single_frame_uncertain(records)
        self.assertEqual(records[0]["rejection_reason"], "single_frame_inside_uncertain")
        self.assertEqual(records[0]["review_status"], "uncertain")
        self.assertFalse(records[0]["keep_after_roi"])

    def test_short_stationary_horizontal_boundary_track_is_rejected(self) -> None:
        records = [
            {
                "track_id": "12", "frame_id": frame_id, "on_court": True,
                "bbox_xyxy": [820.0 + frame_id, 250.0, 950.0 + frame_id, 330.0],
            }
            for frame_id in range(8)
        ]
        annotate_roi_tracks(records)
        mark_single_frame_uncertain(records)
        mark_short_stationary_boundary_tracks(records, (1000, 600))
        self.assertTrue(all(not record["keep_after_roi"] for record in records))
        self.assertTrue(all(
            record["rejection_reason"] == "short_stationary_boundary_track"
            for record in records
        ))

    def test_team_clustering_separates_yellow_and_purple(self) -> None:
        assignments = cluster_two_teams({
            1: hsv_team_feature((28.0, 220.0, 220.0)),
            2: hsv_team_feature((31.0, 205.0, 210.0)),
            3: hsv_team_feature((145.0, 190.0, 150.0)),
            4: hsv_team_feature((150.0, 210.0, 140.0)),
        })
        self.assertEqual(assignments[1][0], assignments[2][0])
        self.assertEqual(assignments[3][0], assignments[4][0])
        self.assertNotEqual(assignments[1][0], assignments[3][0])

    def test_team_clustering_keeps_insufficient_evidence_unknown(self) -> None:
        assignments = cluster_two_teams({1: hsv_team_feature((20.0, 200.0, 200.0))})
        self.assertEqual(assignments[1][0], "unknown")

    def test_team_clustering_rejects_a_severely_unbalanced_split(self) -> None:
        features = {
            **{index: hsv_team_feature((25.0, 220.0, 220.0)) for index in range(9)},
            9: hsv_team_feature((145.0, 220.0, 160.0)),
        }
        assignments = cluster_two_teams(features)
        self.assertTrue(all(team_id == "unknown" for team_id, _ in assignments.values()))

    def test_shot_config_selects_its_own_roi_by_frame(self) -> None:
        config = {"shots": [
            {"shot_id": "shot_0000", "start_frame": 0, "end_frame": 9},
            {"shot_id": "shot_0001", "start_frame": 10, "end_frame": 19},
        ]}
        self.assertEqual(shot_for_frame(config, 14)["shot_id"], "shot_0001")
        self.assertIsNone(shot_for_frame(config, 20))

    def test_appearance_segment_splits_white_to_saturated_uniform_shift(self) -> None:
        records = [
            {
                "shot_id": "shot_0000", "track_id": "6", "frame_id": frame_id,
                "keep_after_roi": True, "overlap": False, "final_role": "team_light",
                "uniform_hsv": [8.0, 25.0, 165.0],
            }
            for frame_id in range(3)
        ] + [
            {
                "shot_id": "shot_0000", "track_id": "6", "frame_id": 3,
                "keep_after_roi": True, "overlap": False, "final_role": "team_dark",
                "uniform_hsv": [8.0, 160.0, 150.0],
            }
        ]
        assign_appearance_segments(records)
        self.assertEqual([record["appearance_segment"] for record in records], [0, 0, 0, 1])
        self.assertEqual(records[-1]["identity_review_reason"], "appearance_shift_possible_id_switch")


if __name__ == "__main__":
    unittest.main()
