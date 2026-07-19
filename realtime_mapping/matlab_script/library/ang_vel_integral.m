function [Qk_plus1]=ang_vel_integral(q0,dt,w0,w1)
% RK4
% conference: A Robust and Easy to implement method for imu
% calibration without External Equipments

q1=q0; 
k1=0.5*omega_matrix(w0)*q1;
q2=q0+dt*0.5*k1;
k2=0.5*omega_matrix((w0+w1)/2)*q2;
q3=q0+dt*0.5*k2;
k3=0.5*omega_matrix((w0+w1)/2)*q3;
q4=q0+dt*k3;
k4=0.5*omega_matrix(w1)*q4;

Qk_plus1=q0+dt*(k1/6+k2/3+k3/3+k4/6);
Qk_plus1=Qk_plus1/norm(Qk_plus1);
end

function [omega]=omega_matrix(w)

wx=w(1);
wy=w(2);
wz=w(3);
omega=[0  , -wx , -wy , -wz ;...
       wx ,  0  ,  wz , -wy ;...
       wy , -wz ,  0  ,  wx ;...
       wz ,  wy , -wx ,  0   ];

end