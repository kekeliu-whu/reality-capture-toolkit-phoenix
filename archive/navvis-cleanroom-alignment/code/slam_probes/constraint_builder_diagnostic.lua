-- Read-only binary diagnostic override.  The installed encrypted base file
-- defines CONSTRAINT_BUILDER; this wrapper changes only result retention and
-- logging so failed candidates expose their exact rejection reason.
include "constraint_builder.lua"

CONSTRAINT_BUILDER.log_matches = true
CONSTRAINT_BUILDER.constraint_candidate_save_mode = "SAVE_ALL_CANDIDATES"

return CONSTRAINT_BUILDER
