%% utils.m - 工具函数模块
classdef utils
    methods(Static)
                function check_toolboxes()
            % 检查必要的工具箱是否已安装
            if ~license('test', 'statistics_toolbox')
                error('需要安装 Statistics and Machine Learning Toolbox');
            end
        end
        
        function data_gpu = toGPU(data)
            % 将数据转移到GPU(如果支持) - 为AMD 5500M GPU优化
            % 输入:
            %   data - 输入数据
            % 输出:
            %   data_gpu - GPU上的数据或原始数据
                
            persistent gpuAvailable gpuMemLimit;
                
            % 只检查一次GPU可用性
            if isempty(gpuAvailable)
                gpuAvailable = (exist('gpuArray', 'file') == 2) && gpuDeviceCount > 0;
                
                if gpuAvailable
                    gpu = gpuDevice();
                    % 为AMD GPU设置更保守的内存限制（总显存的60%）
                    gpuMemLimit = 0.6 * gpu.AvailableMemory;
                    
                    logger.log_message('info', sprintf('GPU可用: %s, 总内存: %.2f GB, 可用内存: %.2f GB', ...
                        gpu.Name, gpu.TotalMemory/1e9, gpu.AvailableMemory/1e9));
                else
                    gpuMemLimit = 0;
                    logger.log_message('info', 'GPU不可用，使用CPU计算');
                end
            end
                
            % 智能决策：根据数据大小和传输开销决定是否使用GPU
            if gpuAvailable
                try
                    % 计算数据大小(字节)
                    dataSize = numel(data) * 8; % 假设是双精度数据
                    
                    % 最小阈值：太小的数据传输开销大于收益(5MB)
                    minThreshold = 5 * 1024 * 1024;
                    
                    % 如果数据太大或太小，不使用GPU
                    if dataSize > gpuMemLimit || dataSize < minThreshold
                        data_gpu = data;
                        return;
                    end
                    
                    % 使用GPU
                    data_gpu = gpuArray(data);
                catch ME
                    logger.log_message('warning', sprintf('GPU转换失败: %s，使用CPU计算', ME.message));
                    data_gpu = data;
                end
            else
                data_gpu = data;
            end
        end
        
        function save_figure(fig, output_dir, filename_base, varargin)
            % 保存图形为多种格式
            % 输入:
            %   fig - 图形句柄
            %   output_dir - 输出目录
            %   filename_base - 文件名基础
            %   varargin - 可选参数
            
            % 解析输入参数
            p = inputParser;
            addParameter(p, 'Formats', {'svg'}, @(x) iscell(x) || ischar(x));
            addParameter(p, 'DPI', 300, @isnumeric);
            addParameter(p, 'MethodName', '', @ischar);
            parse(p, varargin{:});
            
            formats = p.Results.Formats;
            dpi = p.Results.DPI;
            method_name = p.Results.MethodName;
            
            % 如果formats是字符串，转换为cell数组
            if ischar(formats)
                formats = {formats};
            end
            
            % 确保输出目录存在
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            % 处理文件名中的方法名称替换
            if contains(filename_base, '%s')
                if ~isempty(method_name)
                    actual_filename = sprintf(filename_base, method_name);
                else
                    actual_filename = strrep(filename_base, '%s', 'unknown');
                    logger.log_message('warning', '文件名中包含%s但未提供method_name');
                end
            else
                actual_filename = filename_base;
            end
            
            % 准备图形
            set(fig, 'Color', 'white');
            
            % 检查是否存在exportgraphics函数(R2020a或更高版本)
            has_exportgraphics = exist('exportgraphics', 'file') == 2;
            
            % 初始化成功保存的格式列表
            successful_formats = {};
            
            % 保存每种格式
            for i = 1:length(formats)
                format = lower(formats{i});
                filename = fullfile(output_dir, [actual_filename '.' format]);
                
                try
                    if has_exportgraphics
                        % 使用exportgraphics函数(适用于R2020a或更高版本)
                        switch format
                            case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
                                % 位图格式
                                exportgraphics(fig, filename, 'Resolution', dpi);
                            case {'pdf', 'eps', 'svg'}
                                % 矢量格式
                                exportgraphics(fig, filename, 'ContentType', 'vector');
                            otherwise
                                % 其他格式回退到saveas
                                saveas(fig, filename);
                        end
                    else
                        % 回退到传统方法
                        % 禁用工具栏和菜单栏
                        set(fig, 'Toolbar', 'none');
                        set(fig, 'MenuBar', 'none');
                        
                        % 根据不同的格式选择不同的保存方法
                        switch format
                            case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
                                print(fig, filename, ['-d' format], ['-r' num2str(dpi)]);
                            case 'pdf'
                                print(fig, filename, '-dpdf', '-bestfit');
                            case 'eps'
                                print(fig, filename, '-depsc2');
                            case 'svg'
                                print(fig, filename, '-dsvg');
                            otherwise
                                saveas(fig, filename);
                        end
                    end
                    
                    successful_formats{end+1} = upper(format);
                    
                catch ME
                    % 保存失败，记录错误并尝试备用方法
                    logger.log_message('debug', ['保存' upper(format) '格式时出错: ' ME.message]);
                    
                    % 尝试备用方法
                    try
                        saveas(fig, filename);
                        successful_formats{end+1} = upper(format);
                    catch ME2
                        logger.log_message('warning', ['备用保存方法也失败: ' ME2.message]);
                    end
                end
            end
            
            % 输出汇总日志消息
            if ~isempty(successful_formats)
                formats_str = strjoin(successful_formats, ', ');
                logger.log_message('info', sprintf('图形已保存: %s (%s)', actual_filename, formats_str));
            else
                logger.log_message('error', ['图形保存失败: ' actual_filename]);
            end
        end
    end
end

%         function save_figure(fig, dir_path, file_name, varargin)
%             % 保存图形为多种格式
%             % 输入:
%             %   fig - 图形句柄
%             %   dir_path - 保存目录
%             %   file_name - 文件名模板（可包含%s占位符用于方法名）
%             %   varargin - 可选参数:
%             %       'MethodName' - 方法名称（用于格式化文件名）
%             %       'Formats' - 保存格式数组，例如{'png', 'svg'}
% 
%             % 提取可选参数
%             p = inputParser;
%             addParameter(p, 'MethodName', '');
%             addParameter(p, 'Formats', {'png', 'svg'});
%             parse(p, varargin{:});
% 
%             method_name = p.Results.MethodName;
%             formats = p.Results.Formats;
% 
%             % 格式化文件名
%             if ~isempty(method_name)
%                 formatted_name = sprintf(file_name, method_name);
%             else
%                 formatted_name = file_name;
%             end
% 
%             % 获取完整路径
%             path_temp = fullfile(dir_path, formatted_name);
% 
%             try
%                 % 保存为每种格式
%                 for i = 1:length(formats)
%                     format = formats{i};
%                     file_path = [path_temp, '.', format];
% 
%                     % 根据格式类型保存
%                     switch lower(format)
%                         case 'png'
%                             print(fig, file_path, '-dpng', '-r300');
%                         case 'svg'
%                             print(fig, file_path, '-dsvg', '-vector');
%                         case 'pdf'
%                             print(fig, file_path, '-dpdf', '-bestfit');
%                         case 'eps'
%                             print(fig, file_path, '-depsc', '-tiff');
%                         case 'fig'
%                             savefig(fig, file_path);
%                         otherwise
%                             logger.log_message('warning', sprintf('未知图形格式: %s', format));
%                     end
%                 end
% 
%                 logger.log_message('info', sprintf('图形已保存为 %s', formatted_name));
%             catch ME
%                 logger.log_message('error', sprintf('保存图形时出错: %s', ME.message));
%             end
%         end
% 
%         function result = normalize_values(values, min_val, max_val)
%             % 归一化数值到指定范围
%             % 输入:
%             %   values - 输入值
%             %   min_val - 最小输出值
%             %   max_val - 最大输出值
%             % 输出:
%             %   result - 归一化后的值
% 
%             % 检查输入值范围
%             if max(values) == min(values)
%                 result = ones(size(values)) * (min_val + max_val) / 2;
%                 return;
%             end
% 
%             % 执行归一化
%             result = (values - min(values)) / (max(values) - min(values)) * (max_val - min_val) + min_val;
%         end
% 
%         function data = load_data_safe(file_path, sheet_name)
%             % 安全加载数据文件（Excel或CSV）
%             % 输入:
%             %   file_path - 文件路径
%             %   sheet_name - 工作表名称（可选，仅用于Excel）
%             % 输出:
%             %   data - 加载的数据
% 
%             % 检查文件存在
%             if ~exist(file_path, 'file')
%                 error('文件不存在: %s', file_path);
%             end
% 
%             % 获取文件扩展名
%             [~, ~, ext] = fileparts(file_path);
% 
%             try
%                 % 根据文件类型加载数据
%                 switch lower(ext)
%                     case '.csv'
%                         data = readtable(file_path, 'ReadVariableNames', true, 'TreatAsEmpty', {'NA', 'N/A', ''});
% 
%                     case {'.xls', '.xlsx'}
%                         % 检查是否提供了工作表名称
%                         if nargin < 2 || isempty(sheet_name)
%                             data = readtable(file_path, 'ReadVariableNames', true, 'TreatAsEmpty', {'NA', 'N/A', ''});
%                         else
%                             data = readtable(file_path, 'Sheet', sheet_name, 'ReadVariableNames', true, 'TreatAsEmpty', {'NA', 'N/A', ''});
%                         end
% 
%                     otherwise
%                         error('不支持的文件格式: %s', ext);
%                 end
% 
%                 % 替换可能的空值
%                 varnames = data.Properties.VariableNames;
%                 for i = 1:length(varnames)
%                     if iscell(data.(varnames{i}))
%                         empty_idx = cellfun(@isempty, data.(varnames{i}));
%                         data.(varnames{i})(empty_idx) = {NaN};
%                     end
%                 end
% 
%                 logger.log_message('info', sprintf('成功加载数据文件: %s', file_path));
% 
%             catch ME
%                 % 处理错误
%                 logger.log_message('error', sprintf('加载数据文件时出错: %s', ME.message));
%                 rethrow(ME);
%             end
%         end
% 
%         function result = get_optimal_partition(data_size, k_fold)
%             % 获取最优数据划分方案
%             % 输入:
%             %   data_size - 数据集大小
%             %   k_fold - 折数
%             % 输出:
%             %   result - 包含索引的划分方案
% 
%             % 创建随机索引
%             indices = randperm(data_size);
% 
%             % 计算每一折的大小
%             fold_size = floor(data_size / k_fold);
%             remainder = mod(data_size, k_fold);
% 
%             % 初始化结果
%             result = cell(k_fold, 1);
% 
%             % 分配索引
%             start_idx = 1;
%             for i = 1:k_fold
%                 if i <= remainder
%                     end_idx = start_idx + fold_size;
%                 else
%                     end_idx = start_idx + fold_size - 1;
%                 end
% 
%                 result{i} = indices(start_idx:end_idx);
%                 start_idx = end_idx + 1;
%             end
%         end
% 
%         function table_out = standardize_table(table_in)
%             % 标准化表格中的数值列
%             % 输入:
%             %   table_in - 输入表格
%             % 输出:
%             %   table_out - 标准化后的表格
% 
%             % 获取变量名称
%             var_names = table_in.Properties.VariableNames;
% 
%             % 复制表格
%             table_out = table_in;
% 
%             % 标准化每列
%             for i = 1:length(var_names)
%                 var_name = var_names{i};
% 
%                 % 检查是否为数值列
%                 if isnumeric(table_in.(var_name)) && ~islogical(table_in.(var_name))
%                     % 计算标准差和均值
%                     col_std = std(table_in.(var_name), 'omitnan');
%                     col_mean = mean(table_in.(var_name), 'omitnan');
% 
%                     % 防止除以零
%                     if col_std > 0
%                         table_out.(var_name) = (table_in.(var_name) - col_mean) / col_std;
%                     end
%                 end
%             end
%         end
% 
%         function result = calculate_combination_stats(combinations, reference)
%             % 计算变量组合与参考组合的相似度
%             % 输入:
%             %   combinations - 变量组合的cell数组
%             %   reference - 参考组合
%             % 输出:
%             %   result - 相似度统计
% 
%             n_combos = length(combinations);
%             similarity = zeros(n_combos, 1);
%             jaccard = zeros(n_combos, 1);
% 
%             for i = 1:n_combos
%                 combo = combinations{i};
% 
%                 % 计算相似度（相同元素的百分比）
%                 n_common = length(intersect(combo, reference));
%                 similarity(i) = n_common / length(reference);
% 
%                 % 计算Jaccard相似度
%                 jaccard(i) = n_common / length(union(combo, reference));
%             end
% 
%             % 返回结果
%             result = struct();
%             result.average_similarity = mean(similarity);
%             result.max_similarity = max(similarity);
%             result.average_jaccard = mean(jaccard);
%             result.max_jaccard = max(jaccard);
%         end
% 
%         function confidence_interval = bootstrap_confidence_interval(data, n_bootstrap, alpha)
%             % 使用Bootstrap方法计算置信区间
%             % 输入:
%             %   data - 输入数据
%             %   n_bootstrap - Bootstrap样本数
%             %   alpha - 显著性水平（默认0.05，对应95%置信区间）
%             % 输出:
%             %   confidence_interval - [下限, 上限]
% 
%             % 设置默认参数
%             if nargin < 3
%                 alpha = 0.05;
%             end
% 
%             % 移除NaN值
%             data = data(~isnan(data));
% 
%             % 检查数据是否足够
%             if length(data) < 2
%                 confidence_interval = [NaN, NaN];
%                 return;
%             end
% 
%             % 执行Bootstrap
%             n_data = length(data);
%             bootstrap_samples = zeros(n_bootstrap, 1);
% 
%             for i = 1:n_bootstrap
%                 % 有放回抽样
%                 sample_idx = randi(n_data, n_data, 1);
%                 bootstrap_samples(i) = mean(data(sample_idx));
%             end
% 
%             % 计算置信区间
%             sorted_samples = sort(bootstrap_samples);
%             lower_idx = round(alpha/2 * n_bootstrap);
%             upper_idx = round((1-alpha/2) * n_bootstrap);
% 
%             % 防止索引越界
%             lower_idx = max(1, lower_idx);
%             upper_idx = min(n_bootstrap, upper_idx);
% 
%             confidence_interval = [sorted_samples(lower_idx), sorted_samples(upper_idx)];
%         end
% 
%         function similarity = feature_similarity_matrix(X)
%             % 计算特征相似度矩阵
%             % 输入:
%             %   X - 特征矩阵
%             % 输出:
%             %   similarity - 相似度矩阵
% 
%             % 计算相关系数矩阵
%             correlation = corr(X, 'rows', 'pairwise');
% 
%             % 使用绝对值作为相似度度量
%             similarity = abs(correlation);
%         end
% 
%         function result = vif_factors(X)
%             % 计算方差膨胀因子（VIF）
%             % 输入:
%             %   X - 特征矩阵
%             % 输出:
%             %   result - VIF值
% 
%             % 获取特征数量
%             [n, p] = size(X);
% 
%             % 检查样本数是否足够
%             if n <= p
%                 logger.log_message('warning', '样本数小于变量数，VIF计算可能不准确');
%             end
% 
%             % 初始化VIF数组
%             vif = zeros(p, 1);
% 
%             % 对每个特征计算VIF
%             for i = 1:p
%                 % 当前特征作为因变量
%                 y = X(:, i);
% 
%                 % 其他特征作为自变量
%                 X_others = X;
%                 X_others(:, i) = [];
% 
%                 % 添加常数项
%                 X_reg = [ones(n, 1), X_others];
% 
%                 % 使用稳健算法计算
%                 try
%                     % QR分解求解最小二乘问题
%                     [Q, R] = qr(X_reg, 0);
%                     b = R \ (Q' * y);
% 
%                     % 计算R方
%                     y_pred = X_reg * b;
%                     RSS = sum((y - y_pred).^2);
%                     TSS = sum((y - mean(y)).^2);
%                     r_squared = 1 - RSS/TSS;
% 
%                     % 计算VIF
%                     if r_squared < 1
%                         vif(i) = 1 / (1 - r_squared);
%                     else
%                         vif(i) = Inf;
%                     end
%                 catch
%                     vif(i) = NaN;
%                     logger.log_message('warning', sprintf('计算特征%d的VIF时出错', i));
%                 end
%             end
% 
%             % 返回结果
%             result = vif;
%         end
% 
%         function [X_imputed, imputation_stats] = impute_missing_values(X, method)
%             % 填补缺失值
%             % 输入:
%             %   X - 数据矩阵
%             %   method - 填补方法: 'mean', 'median', 'mode', 'knn'
%             % 输出:
%             %   X_imputed - 填补后的数据
%             %   imputation_stats - 填补统计信息
% 
%             % 设置默认方法
%             if nargin < 2
%                 method = 'mean';
%             end
% 
%             % 获取数据维度
%             [n, p] = size(X);
% 
%             % 初始化结果
%             X_imputed = X;
%             imputation_stats = struct();
%             imputation_stats.missing_count = sum(isnan(X), 1)';
%             imputation_stats.missing_percent = imputation_stats.missing_count / n * 100;
%             imputation_stats.method = method;
%             imputation_stats.imputed_values = cell(p, 1);
% 
%             % 对每列进行填补
%             for j = 1:p
%                 % 找出当前列的缺失值
%                 missing_idx = isnan(X(:, j));
%                 n_missing = sum(missing_idx);
% 
%                 % 如果有缺失值
%                 if n_missing > 0
%                     % 根据方法选择填补值
%                     switch lower(method)
%                         case 'mean'
%                             fill_value = mean(X(~missing_idx, j), 'omitnan');
%                         case 'median'
%                             fill_value = median(X(~missing_idx, j), 'omitnan');
%                         case 'mode'
%                             % 计算众数
%                             non_missing_vals = X(~missing_idx, j);
%                             [counts, unique_vals] = histcounts(non_missing_vals);
%                             [~, max_idx] = max(counts);
%                             fill_value = unique_vals(max_idx);
%                         case 'knn'
%                             % KNN填补（简化版）
%                             fill_value = mean(X(~missing_idx, j), 'omitnan');
%                             logger.log_message('warning', 'KNN填补尚未完全实现，使用均值替代');
%                         otherwise
%                             fill_value = mean(X(~missing_idx, j), 'omitnan');
%                             logger.log_message('warning', sprintf('未知填补方法: %s，使用均值替代', method));
%                     end
% 
%                     % 应用填补值
%                     X_imputed(missing_idx, j) = fill_value;
% 
%                     % 记录统计信息
%                     imputation_stats.imputed_values{j} = fill_value;
%                 else
%                     imputation_stats.imputed_values{j} = NaN;
%                 end
%             end
%         end
% 
%         function y_pred = sigmoid_predict(X, beta)
%             % 使用Sigmoid函数预测二分类结果
%             % 输入:
%             %   X - 特征矩阵
%             %   beta - 系数向量
%             % 输出:
%             %   y_pred - 预测概率
% 
%             % 添加截距项
%             X_with_intercept = [ones(size(X, 1), 1), X];
% 
%             % 计算线性组合
%             linear_pred = X_with_intercept * beta;
% 
%             % 应用Sigmoid函数
%             y_pred = 1 ./ (1 + exp(-linear_pred));
%         end
%     end
% end