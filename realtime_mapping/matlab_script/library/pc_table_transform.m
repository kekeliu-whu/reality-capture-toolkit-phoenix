function [output] = pc_table_transform(input,T)
output = input;
output.xyz = pctransform(pointCloud(input.xyz), T).Location;
end

