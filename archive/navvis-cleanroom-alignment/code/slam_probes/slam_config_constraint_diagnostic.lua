-- Full offline SurveyorSLAM diagnostic.  It changes result retention and
-- logging only; all estimator and constraint-builder thresholds stay at the
-- installed G11 offline values.
include "slam_config_offline.lua"

options.serialize_inter_constraints = true
options.serialize_trajectory_nodes = true
options.serialize_submaps = true
options.trajectory_builder.sparse_pose_graph.constraint_builder.log_matches = true
options.trajectory_builder.sparse_pose_graph.constraint_builder.constraint_candidate_save_mode =
    "SAVE_ALL_CANDIDATES"

return options
