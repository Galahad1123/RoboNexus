import unittest

import numpy as np

from core.mask_codec import decode_mask, decode_rgb, encode_mask, encode_rgb


class CodecTest(unittest.TestCase):
    def test_rgb_roundtrip(self):
        rgb = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        np.testing.assert_array_equal(decode_rgb(encode_rgb(rgb)), rgb)

    def test_mask_roundtrip_is_binary(self):
        mask = np.zeros((8, 9), np.uint8)
        mask[2:5, 3:7] = 3
        expected = (mask > 0).astype(np.uint8)
        np.testing.assert_array_equal(decode_mask(encode_mask(mask)), expected)


if __name__ == "__main__":
    unittest.main()
