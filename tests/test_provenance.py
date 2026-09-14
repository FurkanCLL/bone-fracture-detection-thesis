from __future__ import annotations

import unittest

import cv2
import numpy as np

from bone_fracture_audit.provenance import ImageFeatures, compare_image_pair


class ProvenanceTests(unittest.TestCase):
    def test_classifies_small_rotation_as_derivative(self) -> None:
        random_generator = np.random.default_rng(4)
        first = random_generator.integers(0, 80, size=(240, 240), dtype=np.uint8)
        cv2.circle(first, (80, 110), 45, 230, 4)
        cv2.rectangle(first, (130, 45), (200, 180), 170, 5)
        matrix = cv2.getRotationMatrix2D((120, 120), 8, 1.0)
        second = cv2.warpAffine(first, matrix, (240, 240), borderValue=0)
        second = cv2.convertScaleAbs(second, alpha=0.9, beta=12)

        detector = cv2.ORB_create(nfeatures=1800, fastThreshold=8)
        first_points, first_descriptors = detector.detectAndCompute(first, None)
        second_points, second_descriptors = detector.detectAndCompute(second, None)
        result = compare_image_pair(
            ImageFeatures(first, first_points, first_descriptors),
            ImageFeatures(second, second_points, second_descriptors),
        )

        self.assertIn(result["classification"], {"high_confidence_derivative", "probable_derivative"})
        self.assertGreater(result["aligned_correlation"], 0.8)
        self.assertAlmostEqual(abs(result["rotation_degrees"]), 8, delta=2)


if __name__ == "__main__":
    unittest.main()
