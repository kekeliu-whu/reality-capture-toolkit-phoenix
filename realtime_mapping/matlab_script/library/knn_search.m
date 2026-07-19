function  [source_index, neighbor_k_index] = knn_search(source, target, grid_dis, K, d)
max_range = round(d/grid_dis);
N = 100;
if(isa(source,'pointCloud'))
    source = source.Location;
end
if(isa(target,'pointCloud'))
    target = target.Location;
end

neighbor_key = find_near_grid_key(source, grid_dis, max_range);
map = get_local_map_dic(target, grid_dis);
[M,~] = size(neighbor_key);
[~, Locb] = ismember(neighbor_key , map(:,1));

index = sort(Locb,2,'descend');
index = index(:,1:N);
index(index == 0) = 1;

target_point_id = map(:,2);
pt_id = target_point_id(index);
point_neighbor  = target(pt_id,:);
source_rep = repmat(source, [N,1]);
distance = reshape(vecnorm(point_neighbor - source_rep,2,2),M,N);
[distance_sort, id] = sort(distance,2,'ascend');
row = repmat((1:M)',1,K);
col = id(:,1:K);
sub_id = sub2ind([M,N], row,col);
neighbor_k_index = pt_id(sub_id);
source_index = find(sum(distance_sort<d, 2) > K);
neighbor_k_index = neighbor_k_index(source_index,:);

source_index = source_index';
neighbor_k_index = neighbor_k_index';
end

function [key_index]= get_local_map_dic(local_map, grid_dis)
key = get_point_key(local_map, grid_dis);
[unique_key,index] = unique(key);
key_index = [unique_key, index];
end

function neighbor_key =  find_near_grid_key(source, grid_dis, N)
grid = xyz_to_grid(source,grid_dis);
% [m,~] = size(grid);
[x,y,z] = deal(int64(-N:1:N));
[xx,yy,zz] = meshgrid(x,y,z);
neighbor = [reshape(xx,[numel(xx),1]),reshape(yy,[numel(yy),1]),reshape(zz,[numel(zz),1])];
[M,~] = size(neighbor);
grid_all = repmat(grid,[1,1,M]);
neighbor = pagetranspose(reshape(neighbor',[3,1,M]));
temp = hash_voxel(grid_all + neighbor);
neighbor_key = squeeze(temp);

end

function key = get_point_key(xyz, grid_dis)
xyz_grid = xyz_to_grid(xyz,grid_dis);
key = hash_voxel(xyz_grid);
end

function grid = xyz_to_grid(xyz,grid_dis)
grid = int64(round(xyz/grid_dis));
end

