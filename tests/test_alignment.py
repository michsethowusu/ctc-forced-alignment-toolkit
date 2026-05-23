import unittest
import tempfile
import numpy as np
import torch
import torchaudio

from utils import load_audio, read_tokens, text_to_token_ids, forced_align_emissions


class TestAlignment(unittest.TestCase):
    def test_text_to_token_ids(self):
        token2id = {"a": 1, "b": 2}
        ids = text_to_token_ids("ab", token2id)
        self.assertEqual(ids, [1, 2])

    def test_forced_align_simple(self):
        # Dummy emissions: 5 frames, 3 classes (blank=0)
        emissions = torch.tensor([[[0.9, 0.05, 0.05]] * 5]).squeeze(1)  # all blank
        target = [1, 2]
        alignment, scores = forced_align_emissions(emissions, target, blank_id=0)
        self.assertEqual(alignment.shape[0], emissions.shape[0])
        # Alignment should be a path through target with blanks
        self.assertTrue(torch.all(alignment >= 0))

if __name__ == "__main__":
    unittest.main()