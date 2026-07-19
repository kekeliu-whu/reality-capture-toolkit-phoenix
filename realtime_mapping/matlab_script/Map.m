classdef Map
    properties(Constant)
        MAX_HASH_NUM = 1e10;% 为啥是这个值？
        %
        SUB_MAP_TIME = 40; % unit(s)
        NEIGHBOR_NUM = 6;
        % 最大搜索距离跟降采样率有很大关联关系。
        MAX_NEIGHBOR_DIS = Odometery.localmap_downsample_distance*4;
        COMBINE_TIME = 0.1;% unit(s)
    end

    properties(Access = public)
        local_map_downsample;
        file_path;
    end
    properties(Access = private)
 
        sub_map_sweep_num;
        global_map ;
        %
        local_map_sweep_vec;
        local_map_time_vec;
        local_map_sweep_num;
        %
        last_update_sweep_index;
        current_sweep_index;
    end


    methods (Access = public)
        function this = Map(file_path)
            mkdir([file_path,'local_map']);
            this.file_path = [file_path,'local_map/'];
            this.sub_map_sweep_num = floor(this.SUB_MAP_TIME/this.COMBINE_TIME);
            this.global_map = pointCloud(zeros(1,3));
            this.local_map_sweep_vec = repmat(pointCloud(zeros(1,3)),this.sub_map_sweep_num,1);
            this.local_map_sweep_num = 0;
            this.last_update_sweep_index = 0;
            this.current_sweep_index = 0;
        end

        function this = update_sub_map(this, sweep)
            if(this.local_map_sweep_num + 1 > this.sub_map_sweep_num)
                this.local_map_sweep_vec(1) = [];
                this.local_map_time_vec(1,:) = [];
            else
                this.local_map_sweep_num = this.local_map_sweep_num + 1;
            end
            this.local_map_sweep_vec(this.local_map_sweep_num) = pointCloud(sweep.xyz,'Intensity',sweep.intensity);
            this.local_map_time_vec{this.local_map_sweep_num,1} = sweep.timestamp;

            this =  update_local_map_downsample(this, sweep);
            this.current_sweep_index = this.current_sweep_index + 1;
            if(this.current_sweep_index - this.last_update_sweep_index == this.sub_map_sweep_num )
                local_map_point = pccat(this.local_map_sweep_vec);
                file_name_las = [this.file_path,'local_map_',num2str(this.current_sweep_index),'.las'];
                las_file_writer = lasFileWriter(file_name_las);
                total_timestamp = seconds(cell2mat(this.local_map_time_vec));
                attr = lidarPointAttributes('GPSTimeStamp',total_timestamp);
                writePointCloud(las_file_writer,local_map_point,attr);
                this.last_update_sweep_index = this.current_sweep_index;
            end
        end

        function this = update_local_map_downsample(this, sweep)
            downsample_dis =  Odometery.localmap_downsample_distance;
            if(this.local_map_sweep_num == 1)
                this.local_map_downsample = point_down_sample(sweep,downsample_dis,false);
            else
                max_timestamp = sweep.timestamp(end);
                valid_id = max_timestamp - this.local_map_downsample.timestamp   < Map.SUB_MAP_TIME;
                this.local_map_downsample = this.local_map_downsample(valid_id,:);
                this.local_map_downsample = point_down_sample(vertcat(this.local_map_downsample, sweep),downsample_dis,false);
            end
        end

        function save_final_map(this)
            file_name = [this.file_path,'local_map_',num2str(this.current_sweep_index),'.ply'];
            if(this.current_sweep_index>this.sub_map_sweep_num)
                point = pccat(this.local_map_sweep_vec(this.current_sweep_index - this.last_update_sweep_index: end));
            else
                point = pccat(this.local_map_sweep_vec(1:this.current_sweep_index));
            end
            pcwrite(point,file_name,'Encoding','binary');
        end

    end

    methods (Access = private)

        function this = global_map_update(this)
            this.global_map = pccat([this.global_map, this.local_map_point]);
            % las_writer =  lasFileWriter('map.las');
            % writePointCloud(las_writer,this.local_map);
        end
    end
end

