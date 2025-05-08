%% data_module.m - 数据处理模块
classdef data_module
    methods(Static)
        function [data_raw, data_processed, valid_rows] = load_and_preprocess(filename)
            % 加载并预处理数据
            % 输入:
            %   filename - 数据文件名
            % 输出:
            %   data_raw - 原始数据
            %   data_processed - 处理后的数据
            %   valid_rows - 有效行索引
            
            % 加载数据
            t_start = toc;
            [data_raw, success, msg] = data_module.load_data(filename);
            if ~success
                logger.log_message('error', msg);
                error(msg);
            end
            t_end = toc;
            logger.log_message('info', sprintf('成功加载数据，样本数：%d，变量数：%d，耗时：%.2f秒', ...
                size(data_raw, 1), size(data_raw, 2), t_end - t_start));
            
            % 预处理数据
            t_start = toc;
            [data_processed, valid_rows] = data_module.preprocess_data(data_raw);
            t_end = toc;
            logger.log_message('info', sprintf('数据预处理完成，有效样本数：%d，耗时：%.2f秒', ...
                length(valid_rows), t_end - t_start));
        end
        
        function [data, success, message] = load_data(filename)
            % 加载数据文件
            % 输入:
            %   filename - 数据文件名
            % 输出:
            %   data - 加载的数据
            %   success - 是否成功加载
            %   message - 成功或错误消息
            
            success = false;
            message = '';
            data = [];
            
            try
                % 使用try-catch捕获可能的错误
                try
                    % 尝试直接加载变量名为'data'的数据
                    s = load(filename);
                    if isfield(s, 'data')
                        data = s.data;
                    else
                        % 如果没有'data'字段，尝试获取第一个字段
                        fn = fieldnames(s);
                        if ~isempty(fn)
                            data = s.(fn{1});
                        else
                            error('数据文件中没有找到有效变量');
                        end
                    end
                catch ME
                    % 如果上面的方法失败，尝试无变量名加载
                    data = load(filename);
                end
                
                % 检查数据类型并转换
                if istable(data)
                    data = table2array(data);
                elseif ~isnumeric(data)
                    message = '数据必须是数值矩阵或表格';
                    return;
                end
                
                % 检查数据有效性
                if isempty(data)
                    message = '加载的数据为空';
                    return;
                end
                
                % 数据成功加载
                success = true;
                message = '数据加载成功';
            catch ME
                message = sprintf('数据文件 %s 加载失败，请检查文件路径或内容！错误信息：%s', filename, ME.message);
            end
        end
        
        function [data_processed, valid_rows] = preprocess_data(data)
            % 数据清洗与预处理
            % 输入:
            %   data - 原始数据
            % 输出:
            %   data_processed - 处理后的数据
            %   valid_rows - 有效行索引
            
            % 定义有效行和需要反转的项目
            rows = 1:375;
            exclude_rows = [6, 10, 42, 74, 124, 127, 189, 252, 285, 298, 326, 331, 339];
            valid_rows = setdiff(rows, exclude_rows);
            reverse_items = [12, 19, 23];
            max_score = 5;
            
            % 预分配目标数组提高性能
            data_processed = data;
            
            % 反转指定列
            data_processed(:, reverse_items) = max_score + 1 - data_processed(:, reverse_items);
            
            % 选择有效行 - 直接索引比使用setdiff每次都计算更高效
            data_processed = data_processed(valid_rows, :);
            
            % 后处理：检查数据有效性
            if any(isnan(data_processed(:)))
                % 填充NaN值
                data_processed = fillmissing(data_processed, 'linear');
                logger.log_message('warning', '检测到NaN值并使用线性插值填充');
            end
            
            % 如果数据很大且符合GPU处理条件，则尝试GPU加速
            if numel(data_processed) > 1e6
                data_processed = utils.toGPU(data_processed);
            end
        end
        
        function [X, y, var_names, group_means] = prepare_variables(data)
            % 准备自变量和因变量
            % 输入:
            %   data - 预处理后的数据
            % 输出:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   var_names - 变量名称
            %   group_means - 分组均值
            
            t_start = toc;
            
            % 提取因变量(第29列)
            y = data(:, 29);
            
            % 检查因变量范围
            if any(y < 1 | y > 4)
                error('因变量中存在异常值，请检查！');
            end
            
            % 将因变量二元化(>2为1，<=2为0)
            y = double(y > 2);
            
            % 定义分组
            groups = {
                [1, 2, 3, 4, 5, 6, 12, 19, 23],  % 组1
                [7, 8, 9],                       % 组2
                [10, 11],                        % 组3
                [13, 14],                        % 组4
                [15, 17, 18, 20, 21],            % 组5
                [22, 24],                        % 组6
                [25, 26],                        % 组7
                [27, 28]                         % 组8
            };
            
            % 标准化并计算各组均值
            X = zeros(size(data, 1), length(groups));
            group_means = cell(length(groups), 1);
            var_names = cell(length(groups), 1);
            
            % 分块计算参数 - 增大分块大小利用64GB内存
            n_samples = size(data, 1);
            BLOCK_SIZE = 10000;  % 更大的块大小
            n_blocks = ceil(n_samples / BLOCK_SIZE);
            use_blocks = n_samples > BLOCK_SIZE * 3; % 只有超大数据才分块
            
            % 使用parfor并行处理各组
            parfor i = 1:length(groups)
                % 获取当前组的列
                group_cols = groups{i};
                
                % 提取当前组的数据
                group_data = data(:, group_cols);
                
                % 标准化处理
                if use_blocks && n_samples > 100000
                    % 先计算整体统计量
                    mu = mean(group_data);
                    sigma = std(group_data);
                    
                    % 初始化标准化后的数据
                    group_data_std = zeros(size(group_data));
                    
                    % 分块标准化 - 仅对超大数据集使用
                    for b = 1:n_blocks
                        start_idx = (b-1)*BLOCK_SIZE + 1;
                        end_idx = min(b*BLOCK_SIZE, n_samples);
                        block_data = group_data(start_idx:end_idx, :);
                        
                        % 标准化
                        group_data_std(start_idx:end_idx, :) = (block_data - mu) ./ sigma;
                    end
                else
                    % 对于一般大小数据集，直接标准化
                    group_data_std = zscore(group_data);
                end
                
                % 计算标准化后的均值
                group_mean = mean(group_data_std, 2);
                
                % 存储原始均值（未标准化）
                group_orig_mean = mean(group_data, 2);
                
                % 返回结果
                X(:, i) = group_mean;
                group_means{i} = group_orig_mean;
                var_names{i} = sprintf('Group%d', i);
            end
            
            % 将X从GPU移回CPU(如果需要)
            if isa(X, 'gpuArray')
                X = gather(X);
            end
            if isa(y, 'gpuArray')
                y = gather(y);
            end
            
            t_end = toc;
            logger.log_message('info', sprintf('变量准备完成，自变量数：%d，耗时：%.2f秒', ...
                size(X, 2), t_end - t_start));
        end
    end
end