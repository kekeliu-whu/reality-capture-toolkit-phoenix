function down_sample = point_down_sample(data, dis, minus_mean_flag)
% lidar_table is table variable or vector;
if(isa(data,'table'))
    raw_point = data.xyz;
elseif(isa(data,'pointCloud'))
    raw_point = data.Location;
else
    raw_point = data;
end
% 这个地方降采样的原理是防止每次点云在一个地方出现，提高点云均匀度。
if(minus_mean_flag == true)
    point_zoom = (raw_point - mean(raw_point,1))/dis;% 
else
    point_zoom = (raw_point)/dis;
end

point_grid = round(point_zoom);
hash = hash_voxel(int64(point_grid));
center_distance = vecnorm(point_zoom - point_grid, 2, 2);

[~,index] = sort(center_distance,'ascend');
hash = hash(index);

[~,ia] = unique(hash,'first');
id = index(ia);
down_sample = data(id, :);
end