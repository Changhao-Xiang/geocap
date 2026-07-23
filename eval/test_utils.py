import unittest

from eval.utils import extract_range_or_num


class ExtractRangeOrNumTest(unittest.TestCase):
    def test_english_integer(self) -> None:
        self.assertEqual(extract_range_or_num("seven"), [7.0])

    def test_mixed_half(self) -> None:
        self.assertEqual(extract_range_or_num("five and one-half"), [5.5])
        self.assertEqual(extract_range_or_num("six and a half"), [6.5])

    def test_mixed_half_range(self) -> None:
        self.assertEqual(extract_range_or_num("five and one-half to six"), [5.5, 6.0])
        self.assertEqual(
            extract_range_or_num("four and one-half to five, rarely up to six"),
            [4.5, 5.0, 6.0],
        )

    def test_standalone_half_range(self) -> None:
        self.assertEqual(extract_range_or_num("one-half to six"), [0.5, 6.0])

    def test_micrometer_conversion_is_preserved(self) -> None:
        self.assertEqual(extract_range_or_num("120 to 340 μm"), [0.12, 0.34])
        self.assertEqual(extract_range_or_num("190u"), [0.19])


if __name__ == "__main__":
    unittest.main()
