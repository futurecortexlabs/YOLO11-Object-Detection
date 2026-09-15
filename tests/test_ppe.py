import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import ppe


class PPEChecks(unittest.TestCase):
    def test_sample_has_three_people_and_two_reviews(self):
        _, detections = ppe.sample_scene()
        rows = ppe.assess(detections, ["ヘルメット", "安全ベスト"])
        self.assertEqual([r[3] for r in rows], ["装備検出", "要確認", "要確認"])
        self.assertEqual(rows[1][1], "未着用候補")
        self.assertEqual(rows[2][2], "未確認")

    def test_missing_gear_is_unknown_not_violation(self):
        rows = ppe.assess([["human", .9, 0, 0, 100, 200]], ["ヘルメット"])
        self.assertEqual(rows[0][1:4], ["未確認", "未確認", "要確認"])

    def test_gear_cannot_be_assigned_to_two_overlapping_people(self):
        detections = [["human",.9,0,0,100,200],["human",.9,10,0,110,200],["helmet",.9,40,10,60,30]]
        self.assertTrue(all(r[1] == "未確認" for r in ppe.assess(detections,["ヘルメット"])))

    def test_conflicting_helmet_labels_require_review(self):
        rows = ppe.assess([["human",.9,0,0,100,200],["helmet",.9,40,10,60,30],["no-helmet",.9,40,10,60,30]],["ヘルメット"])
        self.assertEqual(rows[0][1], "競合・要確認")

    def test_helmet_at_feet_is_not_associated(self):
        rows = ppe.assess([["human",.9,0,0,100,200],["helmet",.9,40,170,60,190]],["ヘルメット"])
        self.assertEqual(rows[0][1], "未確認")

    def test_selected_equipment_changes_review(self):
        _, det = ppe.sample_scene()
        rows = ppe.assess(det,["ヘルメット"])
        self.assertEqual(rows[2][3],"装備検出")

    def test_sample_export_is_labelled_and_no_model_called(self):
        with tempfile.TemporaryDirectory() as directory, patch('ppe.model') as model:
            out = ppe.run(None,.25,["ヘルメット","安全ベスト"],True,directory)
            model.assert_not_called()
            self.assertEqual(len(out[2]),3)
            self.assertIn("シミュレーション",out[4])
            self.assertTrue(Path(out[5]).read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertIn("シミュレーション",Path(out[5]).read_text(encoding='utf-8-sig'))
            self.assertIn("シミュレーション",Path(out[6]).read_text(encoding='utf-8'))

    def test_validation_clears_all_exports(self):
        for image, conf, required in [(None,.25,["ヘルメット"]),(Image.new('RGB',(10,10)),2,["ヘルメット"]),(Image.new('RGB',(10,10)),.25,[])]:
            out = ppe.run(image,conf,required,False,"unused")
            self.assertIsNone(out[0])
            self.assertEqual(out[-2:],(None,None))

    def test_inference_error_is_not_reported_as_success(self):
        with patch('ppe.model',side_effect=RuntimeError('private path')):
            out = ppe.run(Image.new('RGB',(10,10)),.25,["ヘルメット"],False,"unused")
        self.assertIsNone(out[0])
        self.assertIn("失敗",out[4])
        self.assertNotIn('private path',out[4])

    def test_sample_threshold_can_produce_no_people(self):
        with tempfile.TemporaryDirectory() as directory:
            out = ppe.run(None,1,["ヘルメット"],True,directory)
            self.assertTrue(out[2].empty)
            self.assertIn("確認はできません",out[4])


if __name__ == '__main__':
    unittest.main()
