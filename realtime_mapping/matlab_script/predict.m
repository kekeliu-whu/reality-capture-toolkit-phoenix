clc;clear all;
load test.mat
gyro_std_dev      = 0.0017; % discrete value, unit: rad/s.
acc_std_dev       = 0.014;  % discrete value, unit: m/s^2
gyro_bias_std_dev = 0.0008;% discrete value, unit: rad/s
acc_bias_std_dev  = 0.007; % discrete value, unit: m/s^2
cov = [gyro_std_dev,acc_std_dev,gyro_bias_std_dev,acc_bias_std_dev].^2;
milli_Q  =  diag(reshape(repmat(cov,3,1),12,1))*1e6;
P0 = eye(15,15);
tic;
state_next = predict_state_multi(imu, this.state_update_(end,:), P0, milli_Q);
toc;

function [state_next, P_pred] = predict_state_multi(imu, state0, P0, Q)
dim = size(P0,1);

timestamp = imu.timestamp;
dt = diff([state0.timestamp;imu.timestamp]);
N = length(imu.timestamp);
% state:pos, vel, rot, gyro_bias, acc_bias, grav;
gyro_bias = state0.gyro_bias;
acc_bias = state0.acc_bias;
gyro_true = imu.gyro - gyro_bias;
acc_true = imu.acc - acc_bias;
gravity = [0, 0, -Odometery.G];

R = rotvec2mat3d(state0.rotvec).*cumprod(rotmat(quaternion(gyro_true .* dt,'rotvec'),'point'), 3);
acc_world = squeeze(pagemtimes(R,reshape(acc_true',[3,1,N])))';
vel = state0.vel + cumsum(acc_world.*dt + gravity .* dt);
pos = state0.pos + cumsum(vel .* dt + (0.5 * acc_world .* dt.^2) + 0.5 * gravity .* dt.^2);
rot = rotvec(quaternion(R,'rotmat','point'));


eye_3   = repmat(eye(3,3),[1,1,N]);
zeros_3 = repmat(zeros(3,3),[1,1,N]);
page_dt = reshape(dt,[1,1,N]);


F = [eye_3,      pagemtimes(eye_3, page_dt),      zeros_3,                             zeros_3,                             zeros_3,                     zeros_3; % 正确
    zeros_3,     eye_3,         pagemtimes(-R, skew_symmetric_multi(acc_true.*dt)),    zeros_3,                             pagemtimes(-R, page_dt),     pagemtimes(eye_3,page_dt);% 正确
    zeros_3,     zeros_3,       rotmat(quaternion(-gyro_true .* dt,'rotvec'),'point'), pagemtimes(-eye_3, page_dt),         zeros_3,                     zeros_3; % 可能用A_metrix
    zeros_3,     zeros_3,       zeros_3,                                               eye_3,                               zeros_3,                     zeros_3;
    zeros_3,     zeros_3,       zeros_3,                                               zeros_3,                             eye_3,                       zeros_3;
    zeros_3,     zeros_3,       zeros_3,                                               zeros_3,                             zeros_3,                     eye_3;];


Fw = [zeros_3,                                      zeros_3,                      zeros_3,                        zeros_3;% 这个地方faster-lio 没有用A_metrix,并且关于标准差定义有歧义
    zeros_3,                                       pagemtimes(-R, page_dt),      zeros_3,                        zeros_3;
    pagemtimes(-A_metrix_multi(gyro_true .* dt),page_dt), zeros_3,                      zeros_3,                        zeros_3;
    zeros_3,                                        zeros_3,                      pagemtimes(eye_3, page_dt),     zeros_3;
    zeros_3,                                        zeros_3,                      zeros_3,                        pagemtimes(eye_3, page_dt);
    zeros_3,                                        zeros_3,                      zeros_3,                        zeros_3;];
if(dim == 15)
    F = double(F(1:dim,1:dim,:));
    Fw = double(Fw(1:dim,:,:));
    gravity = [0, 0, -Odometery.G];
else
    gravity = state0.gravity;
end

P_pred = pagemtimes(pagemtimes(F,P0),pagetranspose(F)) + pagemtimes(pagemtimes(Fw,Q),pagetranspose(Fw));
gyro_bias = repmat(state0.gyro_bias,[N,1]);
acc_bias = repmat(state0.acc_bias,[N,1]);
state_next = table(timestamp, pos, vel, rot,gyro_bias,acc_bias );
if(dim == 18)
    state_next.gravity = repmat(gravity,[1,1,N]);
end
end


function res = A_metrix_multi (v)
norm = vecnorm(v,2,2);
N =  length(norm);
reshape_size = [1,1,N];
part1 = pagemtimes(reshape((1 - cos(norm)),  reshape_size), skew_symmetric_multi(v));
part2 = pagemtimes(reshape((1 - sin(norm)),  reshape_size), skew_symmetric_multi(v).^2);
res = repmat(eye(3,3),[1,1,N]);
res(:,:,norm >1e-10) = eye(3) + ...
    pagemtimes(part1,reshape(1./norm.^2,reshape_size)) + ...
    pagemtimes(part2, reshape(1./norm.^3,reshape_size));
end



