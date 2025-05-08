%% utils.m - 工具函数模块
classdef utils
    methods(Static)
        function save_figure(fig, dir_path, file_name, varargin)
            % 保存图形为多种格式
            % 输入:
            %   fig - 图形句柄
            %   dir_path - 保存目录
            %   file_name - 文件名模板（可包含%s占位符用于方法名）
            %   varargin - 可选参数:
            %       'MethodName' - 方法名称（用于格式化文件名）
            %       'Formats' - 保存格式数组，例如{'png', 'svg'}
            
            % 提取可选参数
            p = inputParser;
            addParameter(p, 'MethodName', '');
            addParameter(p, 'Formats', {'png', 'svg'});
            parse(p, varargin{:});
            
            method_name = p.Results.MethodName;
            formats = p.Results.Formats;
            
            % 格式化文件名
            if ~isempty(method_name)
                formatted_name = sprintf(file_name, method_name);
            else
                formatted_name = file_name;
            end
            
            % 获取完整路径
            path_temp = fullfile(dir_path, formatted_name);
            
            try
                % 保存为每种格式
                for i = 1:length(formats)
                    format = formats{i};
                    file_path = [path_temp, '.', format];
                    
                    % 根据格式类型保存
                    switch lower(format)
                        case 'png'
                            print(fig, file_path, '-dpng', '-r300');
                        case 'svg'
                            print(fig, file_path, '-dsvg', '-vector');
                        case 'pdf'
                            print(fig, file_path, '-dpdf', '-bestfit');
                        case 'eps'
                            print(fig, file_path, '-depsc', '-tiff');
                        case 'fig'
                            savefig(fig, file_path);
                        otherwise
                            logger.log_message('warning', sprintf('未知图形格式: %s', format));
                    end
                end
                
                logger.log_message('info', sprintf('图形已保存为 %s', formatted_name));
            catch ME
                logger.log_message('error', sprintf('保存图形时出错: %s', ME.message));
            end
        end
        
        function result = normalize_values(values, min_val, max_val)
            % 归一化数值到指定范围
            % 输入:
            %   values - 输入值
            %   min_val - 最小输出值
            %   max_val - 最大输出值
            % 输出:
            %   result - 归一化后的值
            
            % 检查输入值范围
            if max(values) == min(values)
                result = ones(size(values)) * (min_val + max_val) / 2;
                return;
            end
            
            % 执行归一化
            result = (values - min(values)) / (max(values) - min(values)) * (max_val - min_val) + min_val;
        end
        
        function data = load_data_safe(file_path, sheet_name)
            % 安全加载数据文件（Excel或CSV）
            % 输入:
            %   file_path - 文件路径
            %   sheet_name - 工作表名称（可选，仅用于Excel）
            % 输出:
            %   data - 加载的数据
            
            % 检查文件存在
            if ~exist(file_path, 'file')
                error('文件不存在: %s', file_path);
            end
            
            % 获取文件扩展名
            [~, ~, ext] = fileparts(file_path);
            
            try
                % 根据文件类型加载数据
                switch lower(ext)
                    case '.csv'
                        data = readtable(file_path, 'ReadVariableNames', true, 'TreatAsEmpty', {'NA', 'N/A', ''});
                        
                    case {'.xls', '.xlsx'}
                        % 检查是否提供了工作表名称
                        if nargin < 2 || isempty(sheet_name)
                            data = readtable(file_path, 'ReadVariableNames', true, 'TreatAsEmpty', {'NA', 'N/A', ''});
                        else
                            data = readtable(file_path, 'Sheet', sheet_name, 'ReadVariableNames', true, 'TreatAsEmpty', {'NA', 'N/A', ''});
                        end
                        
                    otherwise
                        error('不支持的文件格式: %s', ext);
                end
                
                % 替换可能的空值
                varnames = data.Properties.VariableNames;
                for i = 1:length(varnames)
                    if iscell(data.(varnames{i}))
                        empty_idx = cellfun(@isempty, data.(varnames{i}));
                        data.(varnames{i})(empty_idx) = {NaN};
                    end
                end
                
                logger.log_message('info', sprintf('成功加载数据文件: %s', file_path));
                
            catch ME
                % 处理错误
                logger.log_message('error', sprintf('加载数据文件时出错: %s', ME.message));
                rethrow(ME);
            end
        end
        
        function result = get_optimal_partition(data_size, k_fold)
            % 获取最优数据划分方案
            % 输入:
            %   data_size - 数据集大小
            %   k_fold - 折数
            % 输出:
            %   result - 包含索引的划分方案
            
            % 创建随机索引
            indices = randperm(data_size);
            
            % 计算每一折的大小
            fold_size = floor(data_size / k_fold);
            remainder = mod(data_size, k_fold);
            
            % 初始化结果
            result = cell(k_fold, 1);
            
            % 分配索引
            start_idx = 1;
            for i = 1:k_fold
                if i <= remainder
                    end_idx = start_idx + fold_size;
                else
                    end_idx = start_idx + fold_size - 1;
                end
                
                result{i} = indices(start_idx:end_idx);
                start_idx = end_idx + 1;
            end
        end
        
        function table_out = standardize_table(table_in)
            % 标准化表格中的数值列
            % 输入:
            %   table_in - 输入表格
            % 输出:
            %   table_out - 标准化后的表格
            
            % 获取变量名称
            var_names = table_in.Properties.VariableNames;
            
            % 复制表格
            table_out = table_in;
            
            % 标准化每列
            for i = 1:length(var_names)
                var_name = var_names{i};
                
                % 检查是否为数值列
                if isnumeric(table_in.(var_name)) && ~islogical(table_in.(var_name))
                    % 计算标准差和均值
                    col_std = std(table_in.(var_name), 'omitnan');
                    col_mean = mean(table_in.(var_name), 'omitnan');
                    
                    % 防止除以零
                    if col_std > 0
                        table_out.(var_name) = (table_in.(var_name) - col_mean) / col_std;
                    end
                end
            end
        end
        
        function result = calculate_combination_stats(combinations, reference)
            % 计算变量组合与参考组合的相似度
            % 输入:
            %   combinations - 变量组合的cell数组
            %   reference - 参考组合
            % 输出:
            %   result - 相似度统计
            
            n_combos = length(combinations);
            similarity = zeros(n_combos, 1);
            jaccard = zeros(n_combos, 1);
            
            for i = 1:n_combos
                combo = combinations{i};
                
                % 计算相似度（相同元素的百分比）
                n_common = length(intersect(combo, reference));
                similarity(i) = n_common / length(reference);
                
                % 计算Jaccard相似度
                jaccard(i) = n_common / length(union(combo, reference));
            end
            
            % 返回结果
            result = struct();
            result.average_similarity = mean(similarity);
            result.max_similarity = max(similarity);
            result.average_jaccard = mean(jaccard);
            result.max_jaccard = max(jaccard);
        end
        
        function confidence_interval = bootstrap_confidence_interval(data, n_bootstrap, alpha)
            % 使用Bootstrap方法计算置信区间
            % 输入:
            %   data - 输入数据
            %   n_bootstrap - Bootstrap样本数
            %   alpha - 显著性水平（默认0.05，对应95%置信区间）
            % 输出:
            %   confidence_interval - [下限, 上限]
            
            % 设置默认参数
            if nargin < 3
                alpha = 0.05;
            end
            
            % 移除NaN值
            data = data(~isnan(data));
            
            % 检查数据是否足够
            if length(data) < 2
                confidence_interval = [NaN, NaN];
                return;
            end
            
            % 执行Bootstrap
            n_data = length(data);
            bootstrap_samples = zeros(n_bootstrap, 1);
            
            for i = 1:n_bootstrap
                % 有放回抽样
                sample_idx = randi(n_data, n_data, 1);
                bootstrap_samples(i) = mean(data(sample_idx));
            end
            
            % 计算置信区间
            sorted_samples = sort(bootstrap_samples);
            lower_idx = round(alpha/2 * n_bootstrap);
            upper_idx = round((1-alpha/2) * n_bootstrap);
            
            % 防止索引越界
            lower_idx = max(1, lower_idx);
            upper_idx = min(n_bootstrap, upper_idx);
            
            confidence_interval = [sorted_samples(lower_idx), sorted_samples(upper_idx)];
        end
        
        function similarity = feature_similarity_matrix(X)
            % 计算特征相似度矩阵
            % 输入:
            %   X - 特征矩阵
            % 输出:
            %   similarity - 相似度矩阵
            
            % 计算相关系数矩阵
            correlation = corr(X, 'rows', 'pairwise');
            
            % 使用绝对值作为相似度度量
            similarity = abs(correlation);
        end
        
        function result = vif_factors(X)
            % 计算方差膨胀因子（VIF）
            % 输入:
            %   X - 特征矩阵
            % 输出:
            %   result - VIF值
            
            % 获取特征数量
            [n, p] = size(X);
            
            % 检查样本数是否足够
            if n <= p
                logger.log_message('warning', '样本数小于变量数，VIF计算可能不准确');
            end
            
            % 初始化VIF数组
            vif = zeros(p, 1);
            
            % 对每个特征计算VIF
            for i = 1:p
                % 当前特征作为因变量
                y = X(:, i);
                
                % 其他特征作为自变量
                X_others = X;
                X_others(:, i) = [];
                
                % 添加常数项
                X_reg = [ones(n, 1), X_others];
                
                % 使用稳健算法计算
                try
                    % QR分解求解最小二乘问题
                    [Q, R] = qr(X_reg, 0);
                    b = R \ (Q' * y);
                    
                    % 计算R方
                    y_pred = X_reg * b;
                    RSS = sum((y - y_pred).^2);
                    TSS = sum((y - mean(y)).^2);
                    r_squared = 1 - RSS/TSS;
                    
                    % 计算VIF
                    if r_squared < 1
                        vif(i) = 1 / (1 - r_squared);
                    else
                        vif(i) = Inf;
                    end
                catch
                    vif(i) = NaN;
                    logger.log_message('warning', sprintf('计算特征%d的VIF时出错', i));
                end
            end
            
            % 返回结果
            result = vif;
        end
        
        function [X_imputed, imputation_stats] = impute_missing_values(X, method)
            % 填补缺失值
            % 输入:
            %   X - 数据矩阵
            %   method - 填补方法: 'mean', 'median', 'mode', 'knn'
            % 输出:
            %   X_imputed - 填补后的数据
            %   imputation_stats - 填补统计信息
            
            % 设置默认方法
            if nargin < 2
                method = 'mean';
            end
            
            % 获取数据维度
            [n, p] = size(X);
            
            % 初始化结果
            X_imputed = X;
            imputation_stats = struct();
            imputation_stats.missing_count = sum(isnan(X), 1)';
            imputation_stats.missing_percent = imputation_stats.missing_count / n * 100;
            imputation_stats.method = method;
            imputation_stats.imputed_values = cell(p, 1);
            
            % 对每列进行填补
            for j = 1:p
                % 找出当前列的缺失值
                missing_idx = isnan(X(:, j));
                n_missing = sum(missing_idx);
                
                % 如果有缺失值
                if n_missing > 0
                    % 根据方法选择填补值
                    switch lower(method)
                        case 'mean'
                            fill_value = mean(X(~missing_idx, j), 'omitnan');
                        case 'median'
                            fill_value = median(X(~missing_idx, j), 'omitnan');
                        case 'mode'
                            % 计算众数
                            non_missing_vals = X(~missing_idx, j);
                            [counts, unique_vals] = histcounts(non_missing_vals);
                            [~, max_idx] = max(counts);
                            fill_value = unique_vals(max_idx);
                        case 'knn'
                            % KNN填补（简化版）
                            fill_value = mean(X(~missing_idx, j), 'omitnan');
                            logger.log_message('warning', 'KNN填补尚未完全实现，使用均值替代');
                        otherwise
                            fill_value = mean(X(~missing_idx, j), 'omitnan');
                            logger.log_message('warning', sprintf('未知填补方法: %s，使用均值替代', method));
                    end
                    
                    % 应用填补值
                    X_imputed(missing_idx, j) = fill_value;
                    
                    % 记录统计信息
                    imputation_stats.imputed_values{j} = fill_value;
                else
                    imputation_stats.imputed_values{j} = NaN;
                end
            end
        end
        
        function y_pred = sigmoid_predict(X, beta)
            % 使用Sigmoid函数预测二分类结果
            % 输入:
            %   X - 特征矩阵
            %   beta - 系数向量
            % 输出:
            %   y_pred - 预测概率
            
            % 添加截距项
            X_with_intercept = [ones(size(X, 1), 1), X];
            
            % 计算线性组合
            linear_pred = X_with_intercept * beta;
            
            % 应用Sigmoid函数
            y_pred = 1 ./ (1 + exp(-linear_pred));
        end
    end
end