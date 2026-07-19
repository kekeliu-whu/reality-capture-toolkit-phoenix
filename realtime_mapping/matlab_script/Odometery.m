classdef Odometery
    properties(Constant)
        imu_time_interval = 0.005;
        use_state_G_flag = false;
        init_offset_time =  13.5; % unit(s)
        init_time_intevel = 4; % unit(s)
        sweep_base_dis      = 0.2;       % 基准降采样距离，用于更新local map； unit(s)
        max_sweep_calculate_dis = 1.5;   % 最大计算降采样距离；unit(s)
        localmap_downsample_distance = 0.1;% unit(s)
        flatness_limit = 0.3;
        max_iter_num = 5;

        min_normal_distance = 0.1;
        normal_distance_slope = 0.003;% 重要参数，跟IMU时间同步性以及编码器精度以及时间同步性有关系。
        max_measure_dis = 300;
        ref_source_point = 4000;
        min_match_point = 40;

        % standard deviation
        gyro_std_dev      = 0.0017; % discrete value, unit: rad/s.
        acc_std_dev       = 0.014;  % discrete value, unit: m/s^2
        gyro_bias_std_dev = 0.0001;% discrete value, unit: rad/s
        acc_bias_std_dev  = 0.0002; % discrete value, unit: m/s^2
        tolerance = [0.001, 0.0001]; % 平移、旋转容差。unit:m, rad
        G = 9.80;% TODO 基于世界各地实际重力加速度设计。
        milli_radio = 1e6;
    end
    properties(Access = public)
        current_scan_indics_ = 0;
        combine_num = 0;
        % sensors
        sensor_ ;

        %map
        map_
        % states
        state_update_;

        milli_Q;
        milli_P;
    end
    methods(Access = public)
        function [this] = Odometery(sensor, map, break_frame, init_state, init_P, init_map)
            this.sensor_ = sensor;
            this.map_ = map;
            this.combine_num = round(this.map_.COMBINE_TIME/this.sensor_.lidar_frame_intevel);
            if(nargin == 2)
                this = init(this);
            else
                this = init_from_interrupt(this, break_frame, init_state, init_P, init_map);
            end
        end
        function [this] = run(this)
            all_states = [];
            all_dx_states = [];
            all_attribute = [];
            all_update_P = [];
            state_update = [];
            while (true)
                this = update_current_indics(this);
                scan_id = this.current_scan_indics_(end)
                [imu, lidar] = this.sensor_.get_sweep_data(this.current_scan_indics_, this.state_update_.timestamp(end));
                if(isempty(imu) || isempty(lidar))
                    this.map_.save_final_map();
                    break;
                end
                % 预测值第一个包含前一个时刻的状态更新。
                [state_predict,this.milli_P, predict_attribute] = predict(imu, this.state_update_(end,:), this.milli_P, this.milli_Q);
                sweep_rot = norm(so3_minus(state_predict.rotvec(end,:),state_predict.rotvec(1,:)));
                [lidar.xyz, state_predict]= undistort(lidar.timestamp, lidar.xyz, state_predict);
                source_base       = point_down_sample(lidar, Odometery.sweep_base_dis, true);
                dis = get_down_sample_dis(pointCloud(source_base.xyz),Odometery.ref_source_point, ...
                    Odometery.sweep_base_dis, Odometery.max_sweep_calculate_dis);
                source_calculate  = point_down_sample(source_base, dis, true);
                [this.state_update_(end+1,:), dx, this.milli_P, use_id, attribute] = IESKF(source_calculate.xyz, this.map_.local_map_downsample.xyz, state_predict(end,:), this.milli_P, this.tolerance);
                %% debug info, shit code
                attribute = [table(scan_id),attribute,table(dis),table(sweep_rot)];
                all_update_P = [all_update_P; [scan_id, dx.timestamp,  diag(this.milli_P)']];
                all_dx_states = vertcat(all_dx_states, [table(scan_id), dx]);
                id = repmat(scan_id,height(state_predict),1);
                % all_states: update_true + half predict + predict
                all_states = vertcat(all_states,[table(id), state_predict, predict_attribute]);
                all_attribute = vertcat(all_attribute, attribute);
                state_update = vertcat(state_update, [table(scan_id), this.state_update_(end,:)]);
                save([this.map_.file_path,'states.mat'],'all_states','all_dx_states','all_attribute','all_update_P','state_update','-mat');

                T = rigidtform3d(rotvec2mat3d(this.state_update_.rotvec(end,:)), this.state_update_.pos(end,:));
                point_world_base = pc_table_transform(source_base, T);
                % point_world_cal = pc_table_transform(source_calculate, T);
                % point_world_cal_use = pc_table_transform(source_calculate(use_id,:), T);
                % T_predict = rigidtform3d(rotvec2mat3d(state_predict.rotvec(end,:)), state_predict.pos(end,:));
                % point_world_predict = pc_table_transform(source_calculate, T_predict);
                % write_las(this.map_.file_path, point_world_predict,            'sweep_cal_world_predict', scan_id);
                % write_las(this.map_.file_path, source_base,                    'sweep_base_body',     scan_id)
                % write_las(this.map_.file_path, point_world_cal,                'sweep_cal_world',     scan_id);
                % write_las(this.map_.file_path, point_world_cal_use,            'sweep_cal_use_world', scan_id);
                if(mod(scan_id,100) == 0)
                    write_las(this.map_.file_path, this.map_.local_map_downsample, 'target',              scan_id);
                end
                %%
                this.map_ = this.map_.update_sub_map(point_world_base);
            end
        end


        function this = update_current_indics(this)
            start_index = this.current_scan_indics_(end)+1;
            this.current_scan_indics_ = start_index: start_index + this.combine_num - 1;
        end
        function [this] = init(this)
            start_frame = this.init_offset_time/this.sensor_.lidar_frame_intevel+1;
            init_frame_num = this.init_time_intevel/this.sensor_.lidar_frame_intevel;
            end_frame = start_frame + init_frame_num - 1;
            this.current_scan_indics_ = start_frame:end_frame;
            [imu_static, lidar_base_imu] = this.sensor_.get_sweep_data(this.current_scan_indics_);
            assert(~isempty(imu_static) && ~isempty(lidar_base_imu),'ERROR:init error, init frame length is smaller than need');
            tform_imu_to_world = get_tform_imu_to_world(imu_static.acc);
            map0 = point_down_sample(lidar_base_imu, Odometery.localmap_downsample_distance, false);
            map0 = pc_table_transform(map0, tform_imu_to_world);
            lidar_end_time = max(lidar_base_imu.timestamp);
            this = init_par(this);
            this = init_state(this, imu_static, lidar_end_time);
            this = init_map(this, map0);
            % pcwrite(pointCloud(init_map.xyz),'init_local_map.ply','Encoding','binary');
        end

        function this = init_from_interrupt(this,break_frame,init_state, init_P, map_init)
            this.current_scan_indics_ = break_frame - this.combine_num +1 :break_frame;
            this.milli_P = init_P;
            cov = [Odometery.acc_std_dev, Odometery.gyro_std_dev, Odometery.gyro_bias_std_dev, Odometery.acc_bias_std_dev].^2;
            this.milli_Q  = diag(reshape(repmat(cov,3,1),12,1)*Odometery.milli_radio);
            this.state_update_ = init_state;
            this.map_ = this.map_.update_sub_map(map_init);
        end

        function [this] = init_state(this, imu_static, lidar_end_time)
            timestamp = 0;
            [pos, vel, rotvec, gyro_bias, acc_bias] = deal(zeros(1,3,'double'));
            gravity = [0, 0, -Odometery.G];
            if(Odometery.use_state_G_flag)
                this.state_update_ = table(timestamp, pos, vel, rotvec, gyro_bias, acc_bias, gravity);
            else
                this.state_update_ = table(timestamp, pos, vel, rotvec, gyro_bias, acc_bias);
            end
            tform_imu_to_world = get_tform_imu_to_world(imu_static.acc);
            init_R = tform_imu_to_world.R;
            this.state_update_.rotvec = rotmat2vec3d(init_R);
            this.state_update_.acc_bias = mean(imu_static.acc) +(init_R'*[0;0; -Odometery.G])';
            this.state_update_.gyro_bias  = mean(imu_static.gyro);
            this.state_update_.timestamp = lidar_end_time;
        end
        function this = init_map(this, init_map)
            this.map_ = this.map_.update_sub_map(init_map);
        end
        function this = init_par(this)
            cov = [Odometery.acc_std_dev, Odometery.gyro_std_dev, Odometery.gyro_bias_std_dev, Odometery.acc_bias_std_dev].^2;
            temp_Q = reshape(repmat(cov,3,1),12,1)*Odometery.milli_radio;
            this.milli_Q  = diag(temp_Q);
            if(Odometery.use_state_G_flag)
                this.milli_P  = diag([ones(3,1); ones(3,1); ones(3,1); 0.0018*ones(3,1); 0.16*ones(3,1);18*ones(3,1)]);
            else
                this.milli_P  = diag([ones(3,1); ones(3,1); ones(3,1); 0.0018*ones(3,1); 0.16*ones(3,1);]);
            end
        end
    end
end
function  write_las(file_path, data, name, id)
file_name_las = [file_path,name,'_', num2str(id),'.las'];
las_file_writer = lasFileWriter(file_name_las);
total_timestamp = seconds(data.timestamp);
attr = lidarPointAttributes('GPSTimeStamp',total_timestamp);
writePointCloud(las_file_writer,pointCloud(single(data.xyz)),attr);
end

function [x_states, dx_states, P, use_id, attribute] = IESKF(source_point, target_point, x0_states, P_pred, tolerance)
x0 = table2array(x0_states(1,2:end));
x = x0;% 不带时间的向量
dim = length(x0);

for i = 1: Odometery.max_iter_num
    x_pos = x(1:3);
    x_rotvec = x(7:9);
    [H, z, std_dev, use_id, observe_attri] = get_observe_jacobi(source_point, target_point, x_rotvec, x_pos, dim);
    if(length(z) < Odometery.min_match_point)
        disp(['Lio failed, match point num is:',num2str(length(z))]);
        break;
    end
    V = std_dev^2*Odometery.milli_radio; %milli 方差
    J_theta = inv(A_metrix(so3_minus(x_rotvec, x0_states.rotvec)))';
    J = blkdiag(eye(3), eye(3), J_theta, eye(3), eye(3));
    if(dim == 18)
        J = blkdiag(J, eye(3));
    end
    P = J\P_pred*inv(J)';% 这一项跟论文一样，但是跟fast-lio 代码不一样
    d = state_minus(x, x0);
    K = (inv(P)+H'/V*H)\H'/V;
    dx= -(K*z)' - d*((eye(dim) - K*H)/J)';% 最后一项有疑问
    x = state_plus(x, dx);
    pos_tol = norm(dx(1:3));
    rot_tol = norm(dx(7:9));
    if(pos_tol <tolerance(1) && rot_tol < tolerance(2)) % 平移容差，角度容差
        break;
    end
end
% 迭代次数，平移容差，姿态容差，最终标准差，
timestamp = x0_states.timestamp;
iter_num = i;
iterate_attri = table(timestamp, iter_num, pos_tol, rot_tol, std_dev);
attribute = [iterate_attri, observe_attri];
P = (eye(dim) - K*H)*P;
x_states = table(timestamp, x(1:3), x(4:6), x(7:9), x(10:12), x(13:15));
dx = state_minus(x, x0);
dx_states = table(timestamp, dx(1:3), dx(4:6), dx(7:9), dx(10:12), dx(13:15));
if(dim == 18)
    x_states.gravity = x(16:18);
    dx_states.gravity = dx(16:18);
end
% point_world = pctransform(pointCloud(source_point(use_id,:)), rigidtform3d(rotvec2mat3d(x(7:9)), x(1:3))).Location;
% quiver3(point_world(:,1),point_world(:,2),point_world(:,3),H(:,1),H(:,2),H(:,3),0);
% axis equal;
end


%         optimize time stamp
%--|------------|--------------|---------------
%^__^----------^__^-----------^__^-----------
%         lidar time stamp
function [point_body,states_predict]= undistort(lidar_timestamp, xyz, states_predict)

assert((lidar_timestamp(1) >= states_predict.timestamp(1)) &&...
    (lidar_timestamp(end) <= states_predict.timestamp(end)),...
    'Lidar time is not in the range of imu');
imu_timestamp = states_predict.timestamp;
pos = states_predict.pos;
quat = quaternion(states_predict.rotvec,'rotvec');
interp_index = discretize(lidar_timestamp, imu_timestamp);
indics = ~isnan(interp_index);
cur_id = interp_index(indics);
next_id = interp_index(indics)+1;
%
radio = (lidar_timestamp(indics) - imu_timestamp(cur_id))./(imu_timestamp(next_id) - imu_timestamp(cur_id));
quat_inter = slerp(quat(cur_id), quat(next_id), radio);
pos_inter = pos(cur_id,:) + (pos(next_id,:) - pos(cur_id,:)).*radio;

% transform to world;
point_world =  rotatepoint(quat_inter,xyz) + pos_inter;
% transform to body;
point_body = rotatepoint(quatinv(quat_inter(end)), point_world - pos_inter(end,:));

% 预测的时间大于雷达本身时间，因此需要将预测状态更新。这个地方可能有bug，因为降采样之后有些点没有用到。
states_predict.timestamp(end) =  lidar_timestamp(end);
states_predict.pos(end,:) =  pos_inter(end,:);
states_predict.rotvec(end,:) = rotvec(quat_inter(end));

end

function [state_predict,P, attribute] = predict(imu, x0, P0, Q)
state_predict = x0;
dim = length(table2array(x0(1,2:end)));
P = P0;
dt = 0;
theta_limit = 0.003;
[d_theta, d_v, gyro_true, acc_true, gyro_world, acc_world] = deal(zeros(1,3));
attribute = table(dt, d_theta, d_v, gyro_true, acc_true, gyro_world, acc_world);
for i = 1:height(imu) - 1
    [state_predict(i+1,:), attribute(i+1,:)] = predict_state(imu(i,:), imu(i+1,:), state_predict(i,:), attribute(i,:), dim);
    d_theta_norm = norm(attribute.d_theta(i+1,:));
    Q_use = Q;
    if(d_theta_norm>theta_limit)
        Q_use(4:6,4:6) = (d_theta_norm/theta_limit)^2 * Q(4:6,4:6);
    end
    [P] = get_P_state(rotvec2mat3d(state_predict.rotvec(i,:)), attribute(i+1,:), P, Q_use, dim);
end
end

function [state_next, attribute] = predict_state(imu_pre, imu_cur, pre_state, pre_attribute, dim)
%
dt = imu_cur.timestamp - imu_pre.timestamp;
% state:pos, vel, rot, gyro_bias, acc_bias, grav;

R = rotvec2mat3d(pre_state.rotvec);
gyro_bias = pre_state.gyro_bias;
acc_bias = pre_state.acc_bias;

acc_true = imu_cur.acc - acc_bias;
gyro_true = imu_cur.gyro - gyro_bias;


d_theta  = ((imu_pre.gyro + imu_cur.gyro)/2 - gyro_bias)  * dt;
d_v      = ((imu_pre.acc  + imu_cur.acc) /2  - acc_bias)  * dt;

if(dim == 15)
    gravity = [0, 0, -Odometery.G];
else
    gravity = pre_state.gravity;
end
% 这个地方单精度非常容易溢出，因为Q的数值太小了。


acc_world = (R * acc_true')' + gravity;
gyro_world = (R * gyro_true')';
delta_v_temp = d_v + 0.5*cross(d_theta, d_v) + (cross(pre_attribute.d_theta, d_v) + cross(pre_attribute.d_v, d_theta))/12;
vel    = pre_state.vel + (R * delta_v_temp')' + gravity* dt;
pos    = pre_state.pos + (vel + pre_state.vel)/2 * dt;
rotvec = rotmat2vec3d(R * rotvec2mat3d(d_theta + cross(pre_attribute.d_theta, d_theta)/12));


timestamp = imu_cur.timestamp;
state_next = table(timestamp, pos, vel, rotvec, gyro_bias, acc_bias);
attribute = table(dt, d_theta, d_v, gyro_true, acc_true, gyro_world, acc_world);
if(dim == 18)
    state_next.gravity = gravity;
end
end

function [P_pred] = get_P_state(R, attribute, P, Q, dim)
dt = attribute.dt;
delta_v =  attribute.d_v;
delta_theta = attribute.d_theta;
eye_3 = eye(3,3);
zeros_3 =  zeros(3,3);
F = [eye_3,      eye_3*dt,      zeros_3,                              zeros_3,        zeros_3,    zeros_3; % 正确
    zeros_3,     eye_3,         -R*skew_symmetric(delta_v),           zeros_3,        -R*dt,      eye_3*dt;% 正确
    zeros_3,     zeros_3,       rotvec2mat3d(-delta_theta),           -eye_3*dt,      zeros_3,    zeros_3; % 可能用A_metrix
    zeros_3,     zeros_3,       zeros_3,                              eye_3,          zeros_3,    zeros_3;
    zeros_3,     zeros_3,       zeros_3,                              zeros_3,        eye_3,      zeros_3;
    zeros_3,     zeros_3,       zeros_3,                              zeros_3,        zeros_3,    eye_3;];
% Q 的顺序是 acc_cov, gyro_cov, gyro_bias_cov, acc_bias_cov;
Fw = [zeros_3,                     zeros_3,         zeros_3,      zeros_3;% 这个地方faster-lio 没有用A_metrix,并且关于标准差定义有歧义
    -R*dt,                         zeros_3,         zeros_3,      zeros_3;
    zeros_3,     -A_metrix(delta_theta)*dt,         zeros_3,      zeros_3;
    zeros_3,                       zeros_3,         eye_3*dt,     zeros_3;
    zeros_3,                       zeros_3,         zeros_3,      eye_3*dt;
    zeros_3,                       zeros_3,         zeros_3,      zeros_3;];

if(dim == 15)
    F = double(F(1:dim,1:dim));
    Fw = double(Fw(1:dim,:));
end
P_pred = F*P*F' + Fw*Q*Fw';
end



function [source_index, target_index] = KNN_search(source,target,K,dis_limit)
[InlierIndices, inlierDists] =multiQueryKNNSearchImpl(pointCloud(target), source.Location, K);


source_index = find(sum(inlierDists<dis_limit, 1) == K);
target_index = InlierIndices(:,source_index);

% pcwrite(pointCloud(source.Location(source_index,:)), 'source.ply','Encoding','binary')
% pcwrite(pointCloud(target.Location(target_index,:)), 'neibor.ply','Encoding','binary')
end

function  planes = get_plane_coef_multi(point_cloud,base_point)
% 功能：利用SVD拟合平面
% 输入：point_cloud  - M*3*K的pages矩阵
% 输出：planes - 拟合所得平面参数 p(1)*x+P(2)*y+P(3)*z+p(4) = 0;

% 去质心
[M,N,K] = size(point_cloud);

A = point_cloud - repmat(mean(point_cloud), M, 1);
[~,sigular,V] = pagesvd(A,'econ','vector');

% 平面度公式：S(3)^2/(S(1)*S(2));其中S(3)为最小奇异值
S = squeeze(sigular)';
flatness = sqrt(S(:,3).^2./(S(:,1).*S(:,2)));
normal = V(:,3,:);
% dtmp = squeeze(mean(pagemtimes(point_cloud, normal)));
% 求法向量方向应当基于当前的base点来确定向外的方向，点云法向量方向与base点连线向外。
dtmp = squeeze(mean(pagemtimes(point_cloud, normal)));
dtmp_base = squeeze(mean(pagemtimes(point_cloud - base_point, normal)));
norm = squeeze(normal)';
planes = [norm.*sign(dtmp_base), -dtmp.*sign(dtmp_base), flatness];
end

function  planes = get_plane_coef(point_cloud)
% 功能：利用SVD批量拟合平面
% 输入：data   - 原始数据(m*3)
% 输出：planes - 拟合所得平面参数 p(1)*x+P(2)*y+P(3)*z+p(4) = 0;
points = point_cloud(:,1:3);
% 去质心
M = points - repmat(mean(points),size(points,1),1);
[~,~,V] = svd(M,'econ');

normal = V(:,3)';
dtmp = mean(points*normal');
planes(1:3) = normal'*sign(dtmp);
planes(4) = -dtmp*sign(dtmp);
end

function [h_x, h, std_dev,use_id, attribute] = get_observe_jacobi(source_point, target_point, rotvec0, T0, dim)
% 返回值均为double类型。
% 雅克比表示的是在经过了rotvec0 以及T0 变换后h的距离以及导数。
R = rotvec2mat3d(rotvec0);
point_world = pctransform(pointCloud(source_point), rigidtform3d(R, T0));
% [source_index, neighbor_index] = KNN_search(point_world, target_point, Map.NEIGHBOR_NUM, Map.MAX_NEIGHBOR_DIS);
[source_index, neighbor_index] = knn_search(point_world, target_point, Odometery.localmap_downsample_distance, Map.NEIGHBOR_NUM, Map.MAX_NEIGHBOR_DIS);

% todo: 根据距离比例判断有效性
% todo: 根据点云对应的平面设置h_x 以及h的权重。
[m,n] = size(neighbor_index);
neighbor_points_2d = target_point(reshape(neighbor_index,[m*n,1]),:)';
neighbor_points_3d = pagetranspose(reshape(neighbor_points_2d,3,m,n));

planes = get_plane_coef_multi(neighbor_points_3d, T0);
init_point_body  = source_point(source_index,:);
init_point_world = point_world.Location(source_index,:);
% 策略1 获取扁平的平面,小于平面度阈值
id1 = find(planes(:,end) < Odometery.flatness_limit);
useful_point_body = init_point_body(id1,:);
useful_point_world = init_point_world(id1,:);
usefull_planes = planes(id1,:);
plane_norm = usefull_planes(:,1:3);
flatness = usefull_planes(:,end);

skew = skew_symmetric_multi(useful_point_body);
[M,~] = size(useful_point_body);
h_x = zeros(M,dim,'double');
% 策略2  对平面度低的赋予低权重
weight = (1 - flatness);
h_x(:,1:3) = plane_norm.*weight;
% A = -norm * R * skew(points);
h_x(:,7:9) = squeeze(pagemtimes(reshape((- plane_norm * R)',[1,3,M]),skew))'.*weight;
dis_to_plane = sum(usefull_planes(:,1:4).*[useful_point_world, ones(M,1,'single')],2);
h = double(dis_to_plane.*weight);
% 策略3 去除距离平面过远的h以及h_x
point_norm = vecnorm(useful_point_body,2,2);
% point_norm/Odometery.max_measure_dis * (max_normal_distance-Odometery.min_normal_distance) + Odometery.min_normal_distance)
id2 = find((abs(dis_to_plane)    < point_norm * Odometery.normal_distance_slope + Odometery.min_normal_distance) & ...
     (abs(dis_to_plane)*10 < point_norm));
h_x = h_x(id2,:);
h = h(id2,:);
std_dev = double(std(h));

% 获取有效点的奇异值。
point_eig = svd(useful_point_world(id2,:)-T0,'econ','vector')';
flat_ness = sqrt(point_eig(3)^2/(point_eig(1)*point_eig(2)));
% 获取有效点的法相量奇异值
norm_eig = svd(plane_norm(id2,:),'econ','vector')';
smoothness = sqrt(norm_eig(3)^2/(norm_eig(1)*norm_eig(2)));
% 获取有效点数量
use_point_num = length(id2);
% 获取重叠率
total_point_num = point_world.Count;
overlap_radio = use_point_num/total_point_num;
attribute = table(point_eig, flat_ness, norm_eig, smoothness, use_point_num, total_point_num, overlap_radio);
%TODO: 去掉入射角过大的点。
use_id = source_index(id1(id2));
end



function res = A_metrix (v)
norm = vecnorm(v,2,2);
if (norm<1e-10)
    res = eye(3);
else
    res = eye(3) + ...
        (1 - cos(norm)) * skew_symmetric(v) / norm^2 +...
        (1 - sin(norm) / norm) * skew_symmetric(v)^2/norm^2;
end
end

function state = state_plus(state0, delta)
assert(sum(size(state0) == size(delta)) == 2,'states dimention is not equal');
state = [state0(1:6) + delta(1:6),...
    so3_plus(state0(7:9), delta(7:9)),...
    state0(10:end) + delta(10:end)];
end

function c = so3_plus(a,b)
% c  = a + b; a b c type is rotvec in so3
assert(isa(a,'double') && isa(b,'double'),'Errror: so3 plus input type is not double');
c = rotmat2vec3d(rotvec2mat3d(a)*rotvec2mat3d(b));
end

function state = state_minus(state0, delta)
assert(sum(size(state0) == size(delta)) == 2,'states dimention is not equal');
state = [state0(1:6) - delta(1:6),...
    so3_minus(state0(7:9), delta(7:9)),...
    state0(10:end) - delta(10:end)];
end
function c = so3_minus(a,b)
assert(isa(a,'double') && isa(b,'double'),'Errror: so3 plus input type is not double');
c = rotmat2vec3d(rotvec2mat3d(b)'*rotvec2mat3d(a));
end

function dis = get_down_sample_dis(point, total_point_num, min_dis, max_dis)
downsample = pcdownsample(point,"gridAverage",1.5);
S = svd(downsample.Location,'econ','vector');
square = 2*(S(1)*S(3)+S(2)*S(3))/downsample.Count;
dis_init = sqrt(square/total_point_num);
downsample =  pcdownsample(point,"gridAverage",dis_init);
dis = sqrt(downsample.Count/total_point_num)*dis_init;

if(dis > max_dis)
    dis  =   max_dis;
elseif(dis < min_dis)
    dis  =   min_dis;
end
end
% 功能同rotvec2mat3d
% function R = so3_exp(psi)
%
% % psi 1*3
% theta = norm(psi,2);
% if(theta ~= 0)
%     a = psi/theta;
% else
%     a = zeros(1,3);
% end
% R = cos(theta)*eye(3) + (1-cos(theta)) * a' * a + sin(theta) * skew_symmetric(a);
% end
