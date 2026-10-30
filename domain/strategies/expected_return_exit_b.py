"""Formal daily B plugin; legacy native v3 and research R1 stay available."""
import json

from domain.strategies.expected_return_exit_v3 import (
    EREM_V3_MODEL_VERSION, EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS, ExpectedReturnExitV3Strategy,
)
from domain.strategies.erem_soft_reduction import FORMAL_VERSION, SoftReductionFormalPolicy

DAILY_EREM_STRATEGY_NAME = "expected_return_exit_v3_b1"
DAILY_EREM_MODEL_VERSION = EREM_V3_MODEL_VERSION + ":" + FORMAL_VERSION
DAILY_EREM_OUTPUT_COLUMNS = EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS + [
    "EREM Model Version", "EREM Exit Policy Version", "EREM Exit Policy",
    "EREM Native Decision", "EREM Reduction Status", "EREM Reduction Cooldown",
]


class ExpectedReturnExitBStrategy(ExpectedReturnExitV3Strategy):
    description = "User-adopted B: native EREM scores, 2-observation REDUCE confirmation/cooldown; immediate EXIT"

    def __init__(self):
        super().__init__(soft_reduction_policy=SoftReductionFormalPolicy())

    def run_with_features(self, data, features, *, latest_only=False):
        result = super().run_with_features(data, features, latest_only=latest_only)
        result["EREM Model Version"] = self.version
        result["EREM Exit Policy"] = json.dumps(self.soft_reduction_policy.record(),
            ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return result
