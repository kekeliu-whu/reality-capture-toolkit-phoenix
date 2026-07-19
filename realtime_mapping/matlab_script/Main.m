function Main()
file_path = input('Please input the file path you want to slam：','s');
file_path = [file_path,'/'];
file_path_par = [file_path,'calib/'];
lidar_inner_par       = yaml.loadFile([file_path_par,'lidar.yaml'] ,                "ConvertToArray", true);
imu_cal_vector        = yaml.loadFile([file_path_par,'imu.yaml'],                   "ConvertToArray", true);
lidar_to_motor_vector = yaml.loadFile([file_path_par,'extrinsic_motor_lidar.yaml'], "ConvertToArray", true);
motor_to_imu_vector   = yaml.loadFile([file_path_par,'extrinsic_imu_motor.yaml'],   "ConvertToArray", true);

imu_cal_matrix        = vector_to_matrix(imu_cal_vector);
lidar_to_motor_matrix = vector_to_matrix(lidar_to_motor_vector);
motor_to_imu_matrix   = vector_to_matrix(motor_to_imu_vector);

sensor = Sensor(file_path,lidar_inner_par.e,imu_cal_matrix, lidar_to_motor_matrix.transform, motor_to_imu_matrix.transform);
map = Map(file_path);
odometery = Odometery(sensor, map);

% break_id = 18000;
% [init_state,init_P, local_map] = get_break_state(map.file_path, break_id);
% odometery = Odometery(sensor, map, break_id, init_state,init_P, local_map);
odometery = odometery.run();
end
%% shit code
function [init_state,init_P, local_map] = get_break_state(file_path,break_id)
load([file_path,'states.mat']);
init_state = state_update(find(state_update.scan_id == break_id), 2:end);
init_P = diag(all_update_P(find(all_update_P(:,1) == break_id),3:end));
lasReader = lasFileReader([file_path,'target_',num2str(break_id),'.las']);
[local_map, attri]= readPointCloud(lasReader,'Attributes','GPSTimeStamp');
xyz = local_map.Location;
timestamp = seconds(attri.GPSTimeStamp);
intensity = local_map.Intensity;
N = length(intensity);
ring = ones(N,1);
frame_id = break_id*ones(N,1);
angle = zeros(N,1);
local_map = table(timestamp,xyz,intensity,ring,frame_id,angle);
end


