function hash = hash_voxel(xyz_grid)
P = int64([73856093,19349663,83492791]);
k = size(xyz_grid,3);
xyz_grid = int64(xyz_grid);
type = 'int64';
temp = xyz_grid.*P;
if(k == 1)
    hash = mod(bitxor(bitxor(temp(:,1),temp(:,2),type),temp(:,3),type),Map.MAX_HASH_NUM);
else
    hash = mod(bitxor(bitxor(temp(:,1,:),temp(:,2,:),type),temp(:,3,:),type),Map.MAX_HASH_NUM);
end
end