"""P05 public-only gate invariants and exact attack scoring semantics."""
import unittest
import pandas as pd
from run_p05 import public_gates,attack_power,A,B,P01,P03_FROM

class PublicGatesTest(unittest.TestCase):
    def _data(self):
        feats=pd.DataFrame({
          "team":[3,7,13,18,20,23,29],
          "rows":[1049]*7,
          "public_A_full13_rows":[0,0,0,1049,0,0,1049],
          "Q_sexmatched_QI5_RMS_lt005":[2,2,0,1,0,2,1],
          "Q_sexmatched_QI5_RMS_median":[.28,.1705,.189,.269,.254,.15,.27]
        }).set_index("team")
        gr=pd.DataFrame({
          "team":[3,7,13,18,20,23,29],
          "aia_imprint_high":[1,1,1,1,1,1,0],
          "C_median_numeric_unique_fraction":[.999,.995,1.,1.,1.,.019,.88]
        }).set_index("team")
        return feats,gr
    def test_background_copy_is_generic_public_flag(self):
        f,g=self._data()
        for t,m in [(18,B),(29,P01)]:
            result,reason=public_gates(t,m,f,g,.199,
                  enable_copy=True,enable_close=False)
            self.assertEqual(result,P01)
            self.assertEqual(reason,"public_background_copy")
        result,reason=public_gates(20,B,f,g,.199,
                     enable_copy=True,enable_close=False)
        self.assertEqual(result,B)
        self.assertEqual(reason,"P04_preserved")
    def test_q_close_gate_requires_multiple_independent_public_conditions(self):
        f,g=self._data()
        selected={}
        for t,m in [(3,A),(7,A),(13,A),(23,A),(18,B),(20,B)]:
            selected[t]=public_gates(t,m,f,g,.199,
                                     enable_copy=False,enable_close=True)[0]
        self.assertEqual([t for t,m in selected.items() if m==P03_FROM],[7])
        self.assertEqual(selected[13],A)
        self.assertEqual(selected[20],B)
    def test_q_rms_threshold_sensitivity_uses_only_public_data(self):
        f,g=self._data()
        weak=public_gates(7,A,f,g,.199,rms_ratio=.80,
                           enable_copy=False,enable_close=True)[0]
        good=public_gates(7,A,f,g,.199,rms_ratio=1.0,
                           enable_copy=False,enable_close=True)[0]
        self.assertEqual(weak,A)
        self.assertEqual(good,P03_FROM)
    def test_attack_power_rare_mia_domination(self):
        mia={1:{"rare_tpr":.5},2:{"rare_tpr":0}}
        a={1:.25,2:0}
        # attack_power convention is fixed 20-target denominator.
        self.assertAlmostEqual(attack_power({1:.2,2:0},mia,a),(.5+.25)/20)
        self.assertAlmostEqual(attack_power({1:.4,2:0},mia,a),(.5+.25)/20)
        self.assertAlmostEqual(attack_power({1:.8,2:0},mia,a),(.8+.25)/20)

if __name__=="__main__":
    unittest.main()
