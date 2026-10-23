"""drama_expert_mask.py: which experts a ratio keeps, ties, whole layers, and refused mass files."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import drama_expert_mask as m

TOOL = Path(__file__).with_name("drama_expert_mask.py")


class KeepTests(unittest.TestCase):
    def test_a_ratio_keeps_the_most_mass_first_and_the_lower_id_on_a_tie(self):
        mass = [0.0] * 256
        for e, v in {9: 5, 3: 5, 200: 4, 7: 3, 1: 2, 2: 2, 250: 1, 100: 1, 99: 1, 5: 0.5}.items():
            mass[e] = v
        mask = m.keep({"4": mass}, 9 / 256, set())
        # 9 kept: the two 5s, the 4, the 3, both 2s, and of the three 1s the lowest ids, 99 and 100
        self.assertEqual(mask, {"4": [1, 2, 3, 7, 9, 99, 100, 200, 250]})

    def test_a_layer_keeps_at_least_the_experts_a_row_picks_and_a_whole_layer_keeps_all(self):
        mask = m.keep({"0": list(range(256)), "5": list(range(256))}, 0.001, {0})
        self.assertEqual(mask["0"], list(range(256)))
        self.assertEqual(mask["5"], list(range(248, 256)))

    def test_bad_layers_and_masses_are_refused(self):
        for mass in ({"40": [1] * 256}, {"03": [1] * 256}, {"x": [1] * 256}, {"1": [1] * 255}, {"1": [1] * 255 + [-1]}, {"1": [1] * 255 + ["a"]}):
            with self.assertRaises(ValueError):
                m.keep(mass, 0.5, set())

    def test_the_tool_writes_a_mask_the_engine_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            mass, out = Path(folder, "mass.json"), Path(folder, "mask.json")
            mass.write_text(json.dumps({"3": [e % 7 for e in range(256)], "36": [1] * 256}))
            run = subprocess.run([sys.executable, str(TOOL), str(mass), "--ratio", "0.4", "--whole", "36", "--out", str(out)], capture_output=True, text=True, check=True)
            mask = json.loads(out.read_text())
            self.assertEqual(len(mask["3"]), 103)
            self.assertTrue(all(e % 7 >= 5 or e % 7 == 4 for e in mask["3"]))
            self.assertEqual(mask["36"], list(range(256)))
            self.assertIn(f"{103 + 39 * 256} of 10240 experts kept", run.stdout)
            bad = subprocess.run([sys.executable, str(TOOL), str(mass), "--ratio", "0", "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
