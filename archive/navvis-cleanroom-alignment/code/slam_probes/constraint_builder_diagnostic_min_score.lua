-- Diagnostic-only override used to expose the exact high-resolution grid
-- score of candidates rejected by the production 0.55 gate.
include "constraint_builder.lua"

CONSTRAINT_BUILDER.log_matches = true
CONSTRAINT_BUILDER.constraint_candidate_save_mode = "SAVE_ALL_CANDIDATES"
CONSTRAINT_BUILDER.min_score = 0.0

return CONSTRAINT_BUILDER
