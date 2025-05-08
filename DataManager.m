classdef DataManager < handle
    % 数据管理类：负责所有数据相关操作
    % 包括加载、预处理、清洗和变量准备
    
    properties (Access = private)
        Config
        Logger
    end
    
    methods (Access = public)
        function obj = DataManager(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
        end
        
        function [data, success, message] = LoadData(obj, filename)
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
                obj.Logger.Log('info', sprintf('正在加载数据文件: %s', filename));
                
                % 检查文件是否存在
                if ~exist(filename, 'file')
                    message = sprintf('文件不存在: %s', filename);
                    obj.Logger.Log('error', message);
                    return;
                end
                
                % 获取文件信息
                file_info = dir(filename);
                file_size_mb = file_info.bytes / (1024^2);
                obj.Logger.Log('debug', sprintf('文件大小: %.2f MB', file_size_mb));
                
                % 根据文件扩展名选择加载方法
                [~, ~, ext] = fileparts(filename);
                
                switch lower(ext)
                    case '.mat'
                        data = obj.LoadMatFile(filename);
                    case {'.csv', '.txt'}
                        data = obj.LoadTextFile(filename);
                    case {'.xlsx', '.xls'}
                        data = obj.LoadExcelFile(filename);
                    otherwise
                        % 尝试作为.mat文件加载
                        data = obj.LoadMatFile(filename);
                end
                
                % 验证数据
                [isValid, validationMessage] = obj.ValidateData(data);
                if ~isValid
                    message = sprintf('数据验证失败: %s', validationMessage);
                    obj.Logger.Log('error', message);
                    return;
                end
                
                success = true;
                message = sprintf('数据加载成功，样本数: %d，变量数: %d', size(data, 1), size(data, 2));
                obj.Logger.Log('info', message);
                
            catch ME
                message = sprintf('数据文件 %s 加载失败: %s', filename, ME.message);
                obj.Logger.LogException(ME, 'LoadData');
            end
        end
        
        function [data_processed, valid_rows] = PreprocessData(obj, data)
            % 数据清洗与预处理
            % 输入:
            %   data - 原始数据
            % 输出:
            %   data_processed - 处理后的数据
            %   valid_rows - 有效行索引
            
            obj.Logger.Log('info', '开始数据预处理');
            
            try
                % 1. 反转指定列
                obj.Logger.Log('debug', sprintf('反转项目: %s', mat2str(obj.Config.ReverseItems)));
                data_processed = data;
                data_processed(:, obj.Config.ReverseItems) = obj.Config.MaxScore + 1 - data_processed(:, obj.Config.ReverseItems);
                
                % 2. 选择有效行
                valid_rows = obj.Config.ValidRows;
                data_processed = data_processed(valid_rows, :);
                obj.Logger.Log('info', sprintf('有效样本数: %d', length(valid_rows)));
                
                % 3. 检查缺失值
                [data_processed, missing_stats] = obj.HandleMissingValues(data_processed);
                obj.Logger.Log('info', sprintf('缺失值处理: 填充了 %d 个值', missing_stats.filled_count));
                
                % 4. 异常值检测
                [data_processed, outlier_stats] = obj.DetectAndHandleOutliers(data_processed);
                obj.Logger.Log('info', sprintf('异常值检测: 检测到 %d 个异常值', outlier_stats.total_outliers));
                
                % 5. 数据验证
                obj.ValidateProcessedData(data_processed);
                
                % 6. 数据摘要
                summary = obj.GenerateDataSummary(data_processed);
                obj.LogDataSummary(summary);
                
            catch ME
                obj.Logger.LogException(ME, 'PreprocessData');
                rethrow(ME);
            end
        end
        
        function [X, y, var_names, group_means] = PrepareVariables(obj, data)
            % 准备自变量和因变量
            % 输入:
            %   data - 预处理后的数据
            % 输出:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   var_names - 变量名称
            %   group_means - 分组均值
            
            obj.Logger.Log('info', '开始变量准备');
            
            try
                % 1. 提取因变量
                y = data(:, obj.Config.TargetColumn);
                
                % 验证因变量
                obj.ValidateDependentVariable(y);
                
                % 2. 二元化因变量
                y = double(y > 2);
                obj.Logger.Log('info', sprintf('因变量二元化完成，正类比例: %.2f%%', mean(y) * 100));
                
                % 3. 处理分组变量
                groups = obj.Config.VariableGroups;
                n_groups = length(groups);
                X = zeros(size(data, 1), n_groups);
                group_means = cell(n_groups, 1);
                var_names = cell(n_groups, 1);
                
                % 并行处理各组
                parfor i = 1:n_groups
                    group_cols = groups{i};
                    [X(:, i), group_means{i}, var_names{i}] = obj.ProcessGroup(data, group_cols, i);
                end
                
                obj.Logger.Log('info', sprintf('变量准备完成，自变量数: %d', size(X, 2)));
                
                % 4. 验证准备好的变量
                obj.ValidatePreparedVariables(X, y);
                
            catch ME
                obj.Logger.LogException(ME, 'PrepareVariables');
                rethrow(ME);
            end
        end
        
        function summary = GenerateDataSummary(obj, data)
            % 生成数据摘要
            % 输入:
            %   data - 数据矩阵
            % 输出:
            %   summary - 数据摘要结构体
            
            summary = struct();
            summary.dimensions = size(data);
            summary.missing_count = sum(isnan(data(:)));
            summary.missing_percentage = summary.missing_count / numel(data) * 100;
            
            % 统计信息
            summary.stats = struct();
            summary.stats.mean = mean(data, 'omitnan');
            summary.stats.std = std(data, 'omitnan');
            summary.stats.min = min(data, [], 'omitnan');
            summary.stats.max = max(data, [], 'omitnan');
            summary.stats.median = median(data, 'omitnan');
            
            % 数据类型分析
            summary.data_types = struct();
            for i = 1:size(data, 2)
                col_data = data(:, i);
                unique_vals = length(unique(col_data(~isnan(col_data))));
                total_vals = sum(~isnan(col_data));
                
                if unique_vals <= 2
                    summary.data_types.(['col_' num2str(i)]) = 'binary';
                elseif unique_vals / total_vals < 0.05
                    summary.data_types.(['col_' num2str(i)]) = 'categorical';
                else
                    summary.data_types.(['col_' num2str(i)]) = 'continuous';
                end
            end
        end
    end
    
    methods (Access = private)
        function data = LoadMatFile(obj, filename)
            % 加载.mat文件
            try
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
                
                % 转换为数值矩阵
                if istable(data)
                    data = table2array(data);
                elseif ~isnumeric(data)
                    error('数据必须是数值矩阵或表格');
                end
                
            catch ME
                % 尝试无变量名加载
                data = load(filename);
                if ~isnumeric(data)
                    error('无法加载数值数据：%s', ME.message);
                end
            end
        end
        
        function data = LoadTextFile(obj, filename)
            % 加载文本文件（CSV等）
            try
                % 检测分隔符
                delimiter = obj.DetectDelimiter(filename);
                
                % 使用readmatrix（MATLAB R2019a+）或csvread
                if exist('readmatrix', 'file')
                    data = readmatrix(filename, 'Delimiter', delimiter);
                else
                    % 回退到csvread（对于旧版本）
                    data = csvread(filename);
                end
                
            catch ME
                obj.Logger.Log('warning', sprintf('使用标准方法失败，尝试备用方法: %s', ME.message));
                
                % 尝试使用importdata
                imported = importdata(filename);
                if isnumeric(imported)
                    data = imported;
                elseif isstruct(imported) && isfield(imported, 'data')
                    data = imported.data;
                else
                    error('无法从文本文件中提取数值数据');
                end
            end
        end
        
        function data = LoadExcelFile(obj, filename)
            % 加载Excel文件
            try
                % 获取所有工作表名称
                [~, sheets] = xlsfinfo(filename);
                
                if isempty(sheets)
                    error('Excel文件中没有找到工作表');
                end
                
                obj.Logger.Log('debug', sprintf('找到 %d 个工作表，使用第一个', length(sheets)));
                
                % 读取第一个工作表
                data = xlsread(filename, sheets{1});
                
                if isempty(data)
                    error('Excel文件中没有找到数值数据');
                end
                
            catch ME
                error('加载Excel文件失败: %s', ME.message);
            end
        end
        
        function delimiter = DetectDelimiter(obj, filename)
            % 检测文本文件的分隔符
            fid = fopen(filename, 'r');
            if fid == -1
                error('无法打开文件');
            end
            
            % 读取前几行
            first_lines = cell(5, 1);
            for i = 1:5
                line = fgetl(fid);
                if ~ischar(line)
                    break;
                end
                first_lines{i} = line;
            end
            fclose(fid);
            
            % 测试不同的分隔符
            delimiters = {',', ';', '\t', ' ', '|'};
            max_count = 0;
            best_delimiter = ',';
            
            for i = 1:length(delimiters)
                count = 0;
                for j = 1:length(first_lines)
                    if ~isempty(first_lines{j})
                        count = count + length(regexp(first_lines{j}, delimiters{i}));
                    end
                end
                
                if count > max_count
                    max_count = count;
                    best_delimiter = delimiters{i};
                end
            end
            
            delimiter = best_delimiter;
            obj.Logger.Log('debug', sprintf('检测到分隔符: "%s"', delimiter));
        end
        
        function [isValid, message] = ValidateData(obj, data)
            % 验证数据
            isValid = true;
            message = '';
            
            % 检查数据是否为空
            if isempty(data)
                isValid = false;
                message = '数据为空';
                return;
            end
            
            % 检查数据类型
            if ~isnumeric(data)
                isValid = false;
                message = '数据不是数值类型';
                return;
            end
            
            % 检查维度
            if size(data, 2) < obj.Config.TargetColumn
                isValid = false;
                message = sprintf('数据列数不足，需要至少 %d 列', obj.Config.TargetColumn);
                return;
            end
            
            % 检查数据范围
            if any(data(:) < 0) || any(data(:) > obj.Config.MaxScore)
                obj.Logger.Log('warning', '数据中存在超出预期范围的值');
            end
        end
        
        function [data_processed, missing_stats] = HandleMissingValues(obj, data)
            % 处理缺失值
            missing_stats = struct();
            missing_stats.initial_count = sum(isnan(data(:)));
            missing_stats.filled_count = 0;
            
            data_processed = data;
            
            if missing_stats.initial_count > 0
                obj.Logger.Log('warning', sprintf('检测到 %d 个缺失值', missing_stats.initial_count));
                
                % 使用多种方法填充缺失值
                for j = 1:size(data, 2)
                    col_data = data(:, j);
                    missing_mask = isnan(col_data);
                    
                    if any(missing_mask)
                        % 根据数据分布选择填充方法
                        if sum(missing_mask) / length(col_data) < 0.05
                            % 缺失值少于5%，使用均值填充
                            fill_value = mean(col_data, 'omitnan');
                            method = '均值';
                        elseif sum(missing_mask) / length(col_data) < 0.15
                            % 缺失值少于15%，使用中位数填充
                            fill_value = median(col_data, 'omitnan');
                            method = '中位数';
                        else
                            % 缺失值较多，使用线性插值
                            data_processed(:, j) = fillmissing(col_data, 'linear');
                            missing_stats.filled_count = missing_stats.filled_count + sum(missing_mask);
                            continue;
                        end
                        
                        data_processed(missing_mask, j) = fill_value;
                        missing_stats.filled_count = missing_stats.filled_count + sum(missing_mask);
                        obj.Logger.Log('debug', sprintf('列 %d 使用%s填充 %d 个缺失值', j, method, sum(missing_mask)));
                    end
                end
            end
        end
        
        function [data_processed, outlier_stats] = DetectAndHandleOutliers(obj, data)
            % 检测和处理异常值
            outlier_stats = struct();
            outlier_stats.total_outliers = 0;
            outlier_stats.outliers_by_column = zeros(1, size(data, 2));
            
            data_processed = data;
            
            % 使用改进的Tukey方法检测异常值
            for j = 1:size(data, 2)
                col_data = data(:, j);
                
                % 计算四分位数
                Q1 = prctile(col_data, 25);
                Q3 = prctile(col_data, 75);
                IQR = Q3 - Q1;
                
                % 计算异常值阈值（使用1.5倍IQR）
                lower_bound = Q1 - 1.5 * IQR;
                upper_bound = Q3 + 1.5 * IQR;
                
                % 检测异常值
                outlier_mask = col_data < lower_bound | col_data > upper_bound;
                n_outliers = sum(outlier_mask);
                
                if n_outliers > 0
                    outlier_stats.total_outliers = outlier_stats.total_outliers + n_outliers;
                    outlier_stats.outliers_by_column(j) = n_outliers;
                    
                    % 处理异常值（Winsorization）
                    data_processed(col_data < lower_bound, j) = lower_bound;
                    data_processed(col_data > upper_bound, j) = upper_bound;
                    
                    obj.Logger.Log('debug', sprintf('列 %d 检测到 %d 个异常值', j, n_outliers));
                end
            end
            
            if outlier_stats.total_outliers > 0
                obj.Logger.Log('info', sprintf('总计处理 %d 个异常值', outlier_stats.total_outliers));
            end
        end
        
        function ValidateProcessedData(obj, data)
            % 验证预处理后的数据
            try
                % 检查是否仍有缺失值
                if any(isnan(data(:)))
                    obj.Logger.Log('warning', '预处理后仍存在缺失值');
                end
                
                % 检查数据范围
                min_val = min(data(:));
                max_val = max(data(:));
                
                if min_val < 0 || max_val > obj.Config.MaxScore
                    obj.Logger.Log('warning', sprintf('数据超出预期范围 [0, %d]: [%.2f, %.2f]', ...
                        obj.Config.MaxScore, min_val, max_val));
                end
                
                % 检查方差
                col_vars = var(data, 'omitnan');
                zero_var_cols = sum(col_vars < 1e-10);
                
                if zero_var_cols > 0
                    obj.Logger.Log('warning', sprintf('%d 个变量方差接近零', zero_var_cols));
                end
                
            catch ME
                obj.Logger.LogException(ME, 'ValidateProcessedData');
            end
        end
        
        function ValidateDependentVariable(obj, y)
            % 验证因变量
            unique_vals = unique(y);
            
            % 检查值的范围
            if any(y < 1) || any(y > 4)
                error('因变量中存在异常值，请检查数据！');
            end
            
            % 检查分布
            value_counts = histcounts(y, 1:5);
            obj.Logger.Log('info', sprintf('因变量分布: 1(%d), 2(%d), 3(%d), 4(%d)', value_counts));
            
            % 检查是否过于不平衡
            min_count = min(value_counts);
            max_count = max(value_counts);
            
            if min_count / max_count < 0.05
                obj.Logger.Log('warning', '因变量分布极不平衡，可能影响模型性能');
            end
        end
        
        function [X_col, group_mean, var_name] = ProcessGroup(obj, data, group_cols, group_idx)
            % 处理单个变量组
            group_data = data(:, group_cols);
            
            % 标准化处理
            group_data_std = zscore(group_data);
            
            % 计算标准化后的均值
            X_col = mean(group_data_std, 2);
            
            % 存储原始均值（未标准化）
            group_mean = mean(group_data, 2);
            
            % 生成变量名
            var_name = sprintf('Group%d', group_idx);
        end
        
        function ValidatePreparedVariables(obj, X, y)
            % 验证准备好的变量
            try
                % 检查维度匹配
                if size(X, 1) ~= length(y)
                    error('自变量和因变量样本数不匹配');
                end
                
                % 检查变量范围
                if any(isnan(X(:))) || any(isinf(X(:)))
                    obj.Logger.Log('warning', '自变量中存在NaN或Inf');
                end
                
                % 检查相关性
                try
                    R = corr(X);
                    if any(isnan(R(:))) || any(isinf(R(:)))
                        obj.Logger.Log('warning', '相关矩阵计算有问题');
                    end
                catch
                    obj.Logger.Log('warning', '无法计算相关矩阵');
                end
                
                % 记录变量统计信息
                var_stats = struct();
                var_stats.mean = mean(X);
                var_stats.std = std(X);
                var_stats.min = min(X);
                var_stats.max = max(X);
                
                obj.Logger.Log('debug', '变量统计信息验证完成');
                
            catch ME
                obj.Logger.LogException(ME, 'ValidatePreparedVariables');
            end
        end
        
        function LogDataSummary(obj, summary)
            % 记录数据摘要到日志
            obj.Logger.Log('info', '=== 数据摘要 ===');
            obj.Logger.Log('info', sprintf('数据维度: %d x %d', summary.dimensions(1), summary.dimensions(2)));
            obj.Logger.Log('info', sprintf('缺失值: %d (%.2f%%)', summary.missing_count, summary.missing_percentage));
            
            % 记录基本统计
            obj.Logger.Log('debug', '基本统计信息:');
            mean_stats = summary.stats.mean;
            std_stats = summary.stats.std;
            
            for i = 1:min(5, length(mean_stats))  % 只记录前5个变量
                obj.Logger.Log('debug', sprintf('  变量 %d: 均值=%.2f, 标准差=%.2f', i, mean_stats(i), std_stats(i)));
            end
            
            if length(mean_stats) > 5
                obj.Logger.Log('debug', '  ... (更多变量统计信息省略)');
            end
        end
    end
end