import unittest
import os
from blacklist_db import init_blacklist_db, add_to_blacklist, remove_from_blacklist, get_all_blacklisted
from fuzzy_matcher import levenshtein_distance, ocr_weighted_distance, fuzzy_match_plate
from alert_dispatcher import AlertDispatcher

class TestBlacklistAlertingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db = "test_blacklist.db"
        init_blacklist_db(cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("KA03NA5278", "KA03NA5278"), 0)
        self.assertEqual(levenshtein_distance("KA03NA5278", "KA03NA5279"), 1)
        self.assertEqual(levenshtein_distance("KA03NA5278", "KA03NA527"), 1)

    def test_ocr_typo_fuzzy_matching(self):
        """Test OCR typo confusions: 0/O, 8/B, 1/I"""
        # KA03NA5278 with 8 -> B typo
        dist_8_b = ocr_weighted_distance("KA03NA527B", "KA03NA5278")
        self.assertLess(dist_8_b, 0.5)

        # KA03NA5278 with 0 -> O typo (GJO1DA1234 -> GJ01DA1234)
        dist_0_o = ocr_weighted_distance("GJO1DA1234", "GJ01DA1234")
        self.assertLess(dist_0_o, 0.5)

        # DL10CC8821 with 1 -> I typo (DL1OCC882I)
        dist_1_i = ocr_weighted_distance("DL1OCC882I", "DL10CC8821")
        self.assertLess(dist_1_i, 0.5)

    def test_fuzzy_match_plate_function(self):
        records = get_all_blacklisted(self.test_db)

        # 1. Exact match
        is_match, rec, conf, dist = fuzzy_match_plate("KA03NA5278", records)
        self.assertTrue(is_match)
        self.assertEqual(rec["plate_number"], "KA03NA5278")
        self.assertEqual(conf, 1.0)

        # 2. Fuzzy match with OCR typo (8 -> B)
        is_match, rec, conf, dist = fuzzy_match_plate("KA03NA527B", records)
        self.assertTrue(is_match)
        self.assertEqual(rec["plate_number"], "KA03NA5278")
        self.assertGreater(conf, 0.85)

        # 3. Clean plate (Not blacklisted)
        is_match, rec, conf, dist = fuzzy_match_plate("MH12XX9999", records)
        self.assertFalse(is_match)

    def test_db_crud_operations(self):
        add_res = add_to_blacklist("MH99TEST1234", "Test Reason", "HIGH", db_path=self.test_db)
        self.assertEqual(add_res["plate_number"], "MH99TEST1234")

        records = get_all_blacklisted(self.test_db)
        plates = [r["plate_number"] for r in records]
        self.assertIn("MH99TEST1234", plates)

        rem_success = remove_from_blacklist("MH99TEST1234", db_path=self.test_db)
        self.assertTrue(rem_success)

    def test_alert_dispatcher_formatting(self):
        dispatcher = AlertDispatcher(camera_location="North Toll Plaza")
        mock_rec = {
            "plate_number": "KA03NA5278",
            "reason": "Stolen Car",
            "priority": "CRITICAL",
            "added_by": "Police HQ",
            "notes": "Armed"
        }
        payload = dispatcher.dispatch_alert("KA03NA527B", mock_rec, confidence=0.95, distance=0.2)
        self.assertEqual(payload["matched_blacklist_plate"], "KA03NA5278")
        self.assertEqual(payload["ocr_read_plate"], "KA03NA527B")
        self.assertEqual(payload["camera_location"], "North Toll Plaza")

if __name__ == "__main__":
    unittest.main()
