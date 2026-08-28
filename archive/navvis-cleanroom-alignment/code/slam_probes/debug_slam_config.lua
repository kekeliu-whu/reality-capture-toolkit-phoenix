include "slam_config_offline.lua"

-- Probe-only override: exercise the binary's existing named debug cloud
-- boundary.  No estimator setting or data source is changed.
options.enable_debug_publishing = true
options.serialize_scan_match_data = true
options.serialize_trajectory_nodes = true
options.serialize_submaps = true

return options
