clc;clear all;

timestamp = [0:0.005:4]';
N = length(timestamp);
V = 8;
R = 0.5;
A = V^2/R;
W = V/R
acc = [A*ones(N,1),zeros(N,1),9.8*ones(N,1)];
gyro = [zeros(N,1),zeros(N,1),W*ones(N,1)];
imu  = table(timestamp,acc,gyro);

pos = [0,0,0]
vel = [0,-V,0]
rotvec = [0,0,0];
acc_bias = [0,0,0];
gyro_bias = [0,0,0];
timestamp = 0;
x0 = table(timestamp,pos,vel,rotvec,acc_bias,gyro_bias)

[state_predict, attribute] = predict_temp(imu, x0);
plot(state_predict.pos(:,1),state_predict.pos(:,2));
xlabel('x');
ylabel('y');
axis equal
function [state_predict, attribute] = predict_temp(imu, x0)
state_predict = x0;
dim = length(table2array(x0(1,2:end)));
% P = P0;
dt = 0;
theta_limit = 0.003;
[d_theta, d_v, gyro_true, acc_true, gyro_world, acc_world] = deal(zeros(1,3));
attribute = table(dt, d_theta, d_v, gyro_true, acc_true, gyro_world, acc_world);
for i = 1:height(imu) - 1
    [state_predict(i+1,:), attribute(i+1,:)] = predict_state(imu(i,:), imu(i+1,:), state_predict(i,:), attribute(i,:), dim);
    d_theta_norm = norm(attribute.d_theta(i+1,:));
    % Q_use = Q;
    % if(d_theta_norm>theta_limit)
    %     Q_use(1:6,1:6) = (d_theta_norm/theta_limit)^2 * Q(1:6,1:6);
    % end
    % [P] = get_P_state(rotvec2mat3d(state_predict.rotvec(i,:)), attribute(i+1,:), P, Q_use, dim);
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
    gravity = [0, 0, -9.8];
else
    gravity = pre_state.gravity;
end
% 这个地方单精度非常容易溢出，因为Q的数值太小了。


acc_world = (R * acc_true')' + gravity;
gyro_world = (R * gyro_true')';
delta_v_temp = d_v + 0.5*cross(d_theta, d_v) + (cross(pre_attribute.d_theta, d_v) + cross(pre_attribute.d_v, d_theta))/12;
vel    = pre_state.vel + (R * delta_v_temp')' + gravity* dt;
pos    = pre_state.pos + (vel + pre_state.vel)/2 * dt;
rotvec = rotmat2vec3d(R * rotvec2mat3d(d_theta+ cross(pre_attribute.d_theta, d_theta)/12 ));%


timestamp = imu_cur.timestamp;
state_next = table(timestamp, pos, vel, rotvec, gyro_bias, acc_bias);
attribute = table(dt, d_theta, d_v, gyro_true, acc_true, gyro_world, acc_world);
if(dim == 18)
    state_next.gravity = gravity;
end
end

