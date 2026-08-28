-- Deterministic read-only binary probe: make the first Exact IMU factor
-- evaluation belong to the first consecutive trajectory-node pair.
OPTIMIZATION_PROBLEM.use_new_imu_cost_function = true
OPTIMIZATION_PROBLEM.ceres_solver_options.num_threads = 1
