clc;clear all;close all;
% poses head: timestamp	x	y	z	q0	q1	q2	q3	v1	v2	v3
% gnss head: timestamp	nav_status	latitude	longitude	altitude coordinate
% undulation lat_variance long_variance height_variance bestpos_lat_delta
% bestpos_long_delta bestpos_height_delta

% load(['/media/sandyyu/SSD/test_yu_lio/line_bike/up_go/LT/local_map/','states.mat']);

data_ulog_path = '/media/sandyyu/SSD/test_yu_lio/line_bike/up_go/LT/data.ulg';
result_ulog_path = '/home/sandyyu/Desktop/test/lio.ulg';
state = get_ulog_topic(result_ulog_path,'FullState');
state_update = table();
state_update.timestamp = seconds(state.timestamp);
state_update.pos = [state.pos_x,state.pos_y,state.pos_z];
state_update.vel = [state.vel_x,state.vel_y,state.vel_z];
state_update.rotvec = [state.rot_x,state.rot_y,state.rot_z];
poses = state_update;
rtk = get_ulog_topic(data_ulog_path,'GnssData');
rtk.collect_timestamp = double(rtk.collect_timestamp)/1e6;
% dt = 0;
% rtk.timestamp = rtk.timestamp +dt;
antenna_location = [0.008243, 0.05794, 0.351]';%长支架
result = get_inter_poses(rtk, poses, antenna_location);
% writematrix(result.rtk_truth,'rtk_truth.csv')
% writematrix(result.lio_inter,'lio_inter.csv')
error = result.rtk_truth - result.lio_inter;
error_norm = vecnorm(error,2,2);
error_rmse = sum(error_norm)/length(error_norm);
error_with_flag = zeros(length(error_norm),3);
error_with_flag(result.valid_flag == 1,:) = 10*error(result.valid_flag == 1,:);
%%
figure(1)
plot3(result.rtk_truth(:,1),result.rtk_truth(:,2),result.rtk_truth(:,3),'-*');hold on;
plot3(result.lio_inter(:,1),result.lio_inter(:,2),result.lio_inter(:,3),'-o');
quiver3(result.rtk_truth(:,1),result.rtk_truth(:,2),result.rtk_truth(:,3),...
         error_with_flag(:,1),error_with_flag(:,2),error_with_flag(:,3),0);
tilt_invalid_index = find(result.tilt_valid_flag == 0);
plot3(result.rtk_truth(tilt_invalid_index,1),result.rtk_truth(tilt_invalid_index,2),result.rtk_truth(tilt_invalid_index,3),'g*')
hold off;

legend("rtk truth",'lio location')
axis equal;
xlabel('x');ylabel('y');zlabel('z');

figure(3)

plot(result.rtk_truth(:,1),result.rtk_truth(:,2),'-*');hold on;
plot(result.lio_inter(:,1),result.lio_inter(:,2),'-o');
quiver(result.rtk_truth(:,1),result.rtk_truth(:,2),...
         error_with_flag(:,1),error_with_flag(:,2),0);
axis equal
legend("rtk truth",'lio location')
figure(2)
subplot(311)
lio_time = result.timestamp - result.timestamp(1);
rtk_time = rtk.timestamp - result.timestamp(1);
plot(lio_time,error(:,1),lio_time,error(:,2),lio_time,error(:,3));hold on;
plot(lio_time,result.valid_flag);hold off;
% plot(rtk_time,rtk.navsat_status);

legend('error x','error y','error z','rtk valid flag')
xlabel('time(s)');ylabel('error(m)')
subplot(312)
plot(rtk_time, rtk.latSigma, ...
     rtk_time, rtk.lonSigma, ...
     rtk_time, rtk.hgtSigma);hold on;
plot(rtk_time, rtk.hdop, ...
     rtk_time, rtk.vdop);hold on;
legend('sigma x','sigma y', 'sigma z', 'hdop', 'vdop');
subplot(313)
plot(lio_time,error_norm);xlabel('time(s)');ylabel('error(m)')



%% 获取雷达时间
function ulog_data = get_ulog_topic(ulog_path,topic_name)
data = readTopicMsgs(ulogreader(ulog_path));
TopicNames = [data.TopicNames];
i = TopicNames == topic_name;
ulog_data = data.TopicMessages{i,1};
end



%% 获取lio转到RTK位置以及RTK依据lio时间戳插值的位置
function  [result] = get_inter_poses(rtk, poses, antenna_location)

[rtk, poses] = get_same_time_range(rtk, poses);
% NED coordinate。
lla = [rtk.latitude, rtk.longitude, double(rtk.altitude+rtk.undulation)];
enu_xyz = lla2enu(lla,lla(1,:),'ellipsoid');
poses.rtk_truth = interp1(rtk.collect_timestamp, enu_xyz, poses.timestamp);
useless_index = isnan(poses.rtk_truth(:,1));
poses(useless_index,:) = [];
dcm = quat2dcm(quaternion(poses.rotvec,'rotvec'));
for i = 1:length(dcm)
    imu_to_rtk(i,:) = (dcm(:,:,i)'*antenna_location)'; %正常
end
poses.lio_to_rtk = poses.pos+ imu_to_rtk;
invalid_index = get_lio_invalid_index(poses,rtk);
valid_index = setdiff(1:size(poses,1),invalid_index);
% [tform,inlier_index] = estgeotform3d(poses.lio_to_rtk(valid_index,:),poses.rtk_truth(valid_index,:),'rigid');
% poses.lio_inter = (tform.R*poses.lio_to_rtk' + tform.Translation')';
N = floor(length(valid_index));
[tform,~] = estgeotform2d(double(poses.lio_to_rtk(valid_index(1:N),1:2)),poses.rtk_truth(valid_index(1:N),1:2),'rigid','MaxNumTrials',10000,'Confidence',99.9);
poses.lio_inter = [(tform.R*poses.lio_to_rtk(:,1:2)')',poses.lio_to_rtk(:,3)];% tform.Translation'
poses.lio_inter = poses.lio_inter - poses.lio_inter(1,:);
poses.valid_flag(1:size(poses,1),1) = 1;
poses.valid_flag(invalid_index,1) = 0;
timestamp = poses.timestamp;
rtk_truth = poses.rtk_truth;
lio_inter = poses.lio_inter;

valid_flag = poses.valid_flag;

tilt_invalid_index = get_invalid_tilt_index(poses);
poses.tilt_valid_flag(1:size(poses,1),1) = 1;
poses.tilt_valid_flag(tilt_invalid_index,1) = 0;
tilt_valid_flag = poses.tilt_valid_flag;
result = table(timestamp,rtk_truth,lio_inter, valid_flag,tilt_valid_flag);
end
%% 获取两者相同时间段内的数据
function [rtk, imu] = get_same_time_range(rtk_data, poses_data)
time_A = max([min(poses_data.timestamp), min(rtk_data.collect_timestamp)]);
time_B = min([max(poses_data.timestamp), max(rtk_data.collect_timestamp)]);
rtk_index = rtk_data.collect_timestamp>=time_A & rtk_data.collect_timestamp<=time_B;
imu_index = poses_data.timestamp>=time_A & poses_data.timestamp<=time_B;
rtk_data = rtk_data(rtk_index,:);
poses_data = poses_data(imu_index,:);
rtk = rtk_data;
imu = poses_data;
end
%% 获取状态不为固定解,以及标准差>0.07的索引
function invalid_index = get_invalid_rtk_index(rtk)
invalid_index = find(rtk.navsat_status ~= 4 |...
    rtk.latSigma >0.07 | rtk.lonSigma >0.07 | rtk.hgtSigma>0.07) ;
end
%% 去除无效的索引
function lio_invalid_index = get_lio_invalid_index(poses,rtk)
    rtk_invalid_index = get_invalid_rtk_index(rtk);
    if(isempty(rtk_invalid_index))
        lio_invalid_index = [];
    else
        lio_invalid_index = discretize(rtk.timestamp(rtk_invalid_index),poses.timestamp);
        lio_invalid_index(isnan(lio_invalid_index)) = [];
    end
    tilt_invalid_index = get_invalid_tilt_index(poses);
    lio_invalid_index = sort(unique (union(tilt_invalid_index, lio_invalid_index)));
end

function tilt_invalid_index = get_invalid_tilt_index(poses)
    delta = deg2rad(15);
    tilt_invalid_index = find(abs(poses.rotvec(:,1))>delta | abs(poses.rotvec(:,2))>delta);
end