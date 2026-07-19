classdef Sensor
    properties (Constant)
        human_box = 1.2;% human bounding box

    end
    properties(Access = public)
        lidar_frame_intevel = 0;
    end
    properties(Access = private)
        file_path_;
        % calibration parameter

        % raw data
        lidar_topics_
        encoder_data_
        imu_data_

        %
        lidar_inner_par_
        imu_inner_par_
        encoder_to_imu_
        lidar_to_motor_

        %
        current_scan_indics_
        current_imu_data_
        current_encoder_data_
    end
    methods(Access = public)
        function this = Sensor(file_path,lidar_inner_par,imu_inner_par, lidar_to_motor, encoder_to_imu)

            this.lidar_inner_par_ = lidar_inner_par;
            this.encoder_to_imu_ = encoder_to_imu;
            this.lidar_to_motor_ = lidar_to_motor;
            this.imu_inner_par_ = imu_inner_par;
            this.file_path_ = file_path;
            this.encoder_data_ = get_encoder_data(file_path);
            this.imu_data_ = get_imu_data(file_path);
            this.lidar_topics_ = get_lidar_topics(file_path);
            this.lidar_frame_intevel = round(mean(diff(this.lidar_topics_.MessageList.Time))/0.05) *0.05;

        end
        function init(this)

            % this.last_update_moment_ = current_update_moment;
            % last_update_moment = min(lidar_data.timestamp);
        end
        function [imu,lidar_base_imu] = get_sweep_data(this, scan_indics, last_update_moment)
            imu = [];
            lidar_base_imu= [];
            lidar_data = get_lidar_data(this.lidar_topics_,scan_indics,this.human_box, Odometery.max_measure_dis);
            if(isempty(lidar_data))
                return;
            end
            if (nargin == 2)
                last_update_moment = min(lidar_data.timestamp);
            end
            current_update_moment = max(lidar_data.timestamp);
            current_encoder_data = get_encoder_in_sweep_time(this.encoder_data_,last_update_moment, current_update_moment);
            if(isempty(current_encoder_data))
                return;
            end
            lidar_data.angle = get_lidar_encoder_angle(lidar_data.timestamp, current_encoder_data);
            lidar_base_imu = get_point_base_imu(lidar_data, this.lidar_inner_par_, this.lidar_to_motor_, this.encoder_to_imu_);
            imu = get_imu_in_lidar_time(this.imu_data_,last_update_moment,current_update_moment);
            imu = get_imu_correction(imu,this.imu_inner_par_);
        end
    end

end
function lidar_topic = get_lidar_topics(file_path)
file_name = dir([file_path, '/*.hbc']);
bag = rosbag(fullfile(file_path, file_name.name));
lidar_topic = select(bag,'Topic','/hesai/pandar');
end

function index = find_neighbors_inout_radius(xyz,radius,flag)
if(flag)
    index = find(vecnorm(xyz,2,2)<radius);
else
    index = find(vecnorm(xyz,2,2)>radius);
end
end

%% 这个地方需要缓存
function lidar_data = get_lidar_data(lidar_topic,indics,box_radis, out_radis)
if(indics(end) > size(lidar_topic.MessageList,1))
    lidar_data = [];
    return;
end
lidar_message = readMessages(lidar_topic,indics);
for i = 1:length(indics)
    xyz_cell{i,1} =  readXYZ(lidar_message{i,1});
    time_cell{i,1} = readField(lidar_message{i,1},'timestamp');
    intensity_cell{i,1} = uint8(readField(lidar_message{i,1},'intensity'));
    ring_cell{i,1} = uint8(readField(lidar_message{i,1},'ring'));
    frame_id_cell{i,1} = uint32(ones(length(time_cell{i,1}),1));
end
xyz = cell2mat(xyz_cell);
timestamp =  cell2mat(time_cell);
ring = cell2mat(ring_cell);
intensity = cell2mat(intensity_cell);
frame_id = cell2mat(frame_id_cell);

lidar_data = table(timestamp, xyz, intensity, ring, frame_id);
indices_inbox = find_neighbors_inout_radius(xyz, box_radis, true);
indices_outbox = find_neighbors_inout_radius(xyz,out_radis, false);
lidar_data([indices_inbox;indices_outbox],:) = [];
end

function [inter_angle] = get_lidar_encoder_angle(lidar_time, encoder)
N = length(encoder.angle);
inter_angle = NaN(length(lidar_time),1);
motor_q = quaternion([encoder.angle,zeros(N,2)],'eulerd','ZYX','frame'); %unit
interp_index = discretize(lidar_time, encoder.timestamp);
indics = ~isnan(interp_index);

current_index = interp_index(indics);
next_index = interp_index(indics)+1;
radio = (lidar_time(indics) - encoder.timestamp(current_index))./(encoder.timestamp(next_index) - encoder.timestamp(current_index));
lidar_q = slerp(motor_q(current_index), motor_q(next_index), radio);
lidar_angle = quat2eul(lidar_q,'ZYX');
inter_angle(indics) = rad2deg(lidar_angle(:,1));
end

function lidar = get_point_base_imu(lidar_data, lidar_inner_par, lidar_motor_matrix, encoder_imu_matrix)
tform_lidar_to_motor = rigidtform3d(lidar_motor_matrix);
[yaw,pitch,roll] = dcm2angle(tform_lidar_to_motor.R');
xyz_ele_cal = single(elevation_offset(lidar_data.xyz,lidar_inner_par));
xyz_trans_var = get_point_ref_motor(xyz_ele_cal, lidar_data.angle, tform_lidar_to_motor.Translation, [yaw,roll]);
point_cloud = pctransform(pointCloud(xyz_trans_var), rigidtform3d(encoder_imu_matrix));
lidar = lidar_data;
lidar.xyz = point_cloud.Location;
end

function current_encoder = get_encoder_in_sweep_time(encoder_data,start_time,end_time)
assert(start_time<end_time,'lidar sweep timestamp error');
index1 = find(encoder_data.timestamp<start_time,1,'last');
index2 = find(encoder_data.timestamp>end_time,1,'first');
current_encoder = encoder_data(index1:index2,:);
end

function currrent_imu = get_imu_in_lidar_time(imu_data, start_time, end_time)
assert(start_time<end_time,'lidar sweep timestamp error');
index1 = find(imu_data.timestamp<=start_time,1,'last');
index2 = find(imu_data.timestamp>=end_time,1,'first');

currrent_imu = imu_data(index1:index2,:);
if(start_time ~= currrent_imu.timestamp(1))
    imu_interp = interp1(currrent_imu.timestamp(1:2),[currrent_imu.acc(1:2,:),currrent_imu.gyro(1:2,:)],start_time);   
    currrent_imu.timestamp(1)= start_time;
    currrent_imu.acc(1,:)= imu_interp(1:3);
    currrent_imu.gyro(1,:)= imu_interp(4:6);
end

end
function encoder =  get_encoder_data(file_path)
data = get_ulog_topic(file_path,'EncoderData');
timestamp = double(data.collect_timestamp)/10^6;
angle = data.angle;
encoder = table(timestamp,angle);
end
function imu =  get_imu_data(file_path)
data = get_ulog_topic(file_path,'ImuData');
timestamp = double(data.collect_timestamp)/10^6 - 0.0025;
acc = [data.accel_x, data.accel_y,data.accel_z];
gyro = [data.gyro_x, data.gyro_y,data.gyro_z];
imu = table(timestamp,acc,gyro);
end

function imu_correct = get_imu_correction(imu, par)
timestamp = double(imu.timestamp);
acc =  double((par.Ta*par.Ka*(imu.acc' - par.Ba))');
gyro = double((par.Tg*par.Kg*(imu.gyro'- par.Bg))');
imu_correct = table(timestamp,acc,gyro);
end