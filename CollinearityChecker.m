classdef CollinearityChecker < handle
    % 多重共线性检查器类：检测和处理变量间的多重共线性
    % 使用多种方法：VIF、条件数、相关系数等
    
    properties (Access = private)
        Config
        Logger
        Results
    end
    
    methods (Access = public)
        function obj = CollinearityChecker(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.Results = struct();
        end
        
        function [X_cleaned, vif_values, removed_vars] = Check(obj, X, var_names)
            % 执行多重共线性检查
            % 输入:
            %   X - 自变量矩阵
            %   var_names - 变量名称
            % 输出:
            %   X_cleaned - 清理后的自变量矩阵
            %   vif_values - VIF值
            %   removed_vars - 被移除的变量标记
            
            obj.Logger.Log('info', '开始多重共线性检查');
            
            try
                % 初始化结果
                obj.Results.original_vars = var_names;
                obj.Results.original_dim = size(X);
                
                % 1. 计算基本统计信息
                obj.CalculateBasicStats(X, var_names);
                
                % 2. 计算相关矩阵
                obj.CalculateCorrelationMatrix(X);
                
                % 3. 检查矩阵条件
                obj.CheckMatrixCondition();
                
                % 4. 计算VIF值
                vif_values = obj.CalculateVIF(X);
                
                % 5. 检测高相关变量对
                obj.DetectHighCorrelationPairs();
                
                % 6. 处理多重共线性
                [X_cleaned, removed_vars] = obj.HandleMulticollinearity(X, var_names, vif_values);
                
                % 7. 验证清理后的矩阵
                obj.ValidateCleanedMatrix(X_cleaned);
                
                % 8. 生成报告
                obj.GenerateCollinearityReport(var_names, removed_vars);
                
                obj.Logger.Log('info', sprintf('多重共线性检查完成，最终变量数: %d', size(X_cleaned, 2)));
                
            catch ME
                obj.Logger.LogException(ME, 'CollinearityChecker.Check');
                rethrow(ME);
            end
        end
        
        function results = GetResults(obj)
            % 获取检查结果
            results = obj.Results;
        end
        
        function SaveResults(obj, output_dir)
            % 保存检查结果
            try
                % 保存结果到mat文件
                result_file = fullfile(output_dir, 'collinearity_results.mat');
                results = obj.Results;
                save(result_file, 'results', '-v7.3');
                
                % 保存VIF结果到CSV
                if isfield(obj.Results, 'vif_table')
                    csv_file = fullfile(output_dir, 'vif_values.csv');
                    writetable(obj.Results.vif_table, csv_file);
                end
                
                % 保存相关矩阵图
                obj.SaveCorrelationHeatmap(output_dir);
                
                obj.Logger.Log('info', '多重共线性检查结果已保存');
                
            catch ME
                obj.Logger.LogException(ME, 'CollinearityChecker.SaveResults');
            end
        end
    end
    
    methods (Access = private)
        function CalculateBasicStats(obj, X, var_names)
            % 计算基本统计信息
            obj.Results.basic_stats = struct();
            obj.Results.basic_stats.mean = mean(X);
            obj.Results.basic_stats.std = std(X);
            obj.Results.basic_stats.variance = var(X);
            obj.Results.basic_stats.range = range(X);
            
            % 检查零方差变量
            zero_var_idx = find(var(X) < 1e-10);
            if ~isempty(zero_var_idx)
                obj.Logger.Log('warning', sprintf('发现 %d 个零方差变量:', length(zero_var_idx)));
                for i = 1:length(zero_var_idx)
                    obj.Logger.Log('warning', sprintf('  - %s', var_names{zero_var_idx(i)}));
                end
            end
            
            obj.Results.basic_stats.zero_variance_vars = zero_var_idx;
        end
        
        function CalculateCorrelationMatrix(obj, X)
            % 计算相关矩阵
            try
                obj.Results.correlation_matrix = corr(X, 'Type', 'Pearson');
                
                % 处理可能出现的NaN
                if any(isnan(obj.Results.correlation_matrix(:)))
                    obj.Logger.Log('warning', '相关矩阵包含NaN值，使用Spearman相关系数重新计算');
                    obj.Results.correlation_matrix = corr(X, 'Type', 'Spearman');
                end
                
                % 计算平均相关系数
                n_vars = size(X, 2);
                upper_tri_idx = triu(true(n_vars), 1);
                correlations = obj.Results.correlation_matrix(upper_tri_idx);
                
                obj.Results.avg_correlation = mean(abs(correlations));
                obj.Results.max_correlation = max(abs(correlations));
                
                obj.Logger.Log('debug', sprintf('平均相关系数绝对值: %.3f', obj.Results.avg_correlation));
                obj.Logger.Log('debug', sprintf('最大相关系数绝对值: %.3f', obj.Results.max_correlation));
                
            catch ME
                obj.Logger.LogException(ME, 'CalculateCorrelationMatrix');
                % 使用备用方法
                obj.Results.correlation_matrix = eye(size(X, 2));
            end
        end
        
        function CheckMatrixCondition(obj)
            % 检查矩阵条件数
            try
                R = obj.Results.correlation_matrix;
                
                % 计算条件数
                obj.Results.condition_number = cond(R);
                
                % 计算特征值
                eigenvalues = eig(R);
                obj.Results.eigenvalues = eigenvalues;
                obj.Results.min_eigenvalue = min(eigenvalues);
                obj.Results.max_eigenvalue = max(eigenvalues);
                
                % 检查矩阵是否接近奇异
                if obj.Results.condition_number > obj.Config.ConditionNumberThreshold
                    obj.Logger.Log('warning', sprintf('相关矩阵条件数过高 (%.2f)，可能存在严重多重共线性', obj.Results.condition_number));
                end
                
                if obj.Results.min_eigenvalue < 1e-10
                    obj.Logger.Log('warning', '相关矩阵接近奇异，最小特征值接近零');
                end
                
            catch ME
                obj.Logger.LogException(ME, 'CheckMatrixCondition');
            end
        end
        
        function vif_values = CalculateVIF(obj, X)
            % 计算方差膨胀因子（VIF）
            n_vars = size(X, 2);
            vif_values = zeros(n_vars, 1);
            
            try
                % 使用并行计算VIF
                parfor i = 1:n_vars
                    vif_values(i) = obj.CalculateSingleVIF(X, i);
                end
                
                % 保存VIF结果
                obj.Results.vif_values = vif_values;
                
                % 创建VIF表格
                obj.Results.vif_table = table((1:n_vars)', vif_values, obj.Results.original_vars', ...
                    'VariableNames', {'VariableIndex', 'VIF', 'VariableName'});
                
                % 按VIF值排序
                obj.Results.vif_table = sortrows(obj.Results.vif_table, 'VIF', 'descend');
                
                % 记录VIF统计
                obj.Logger.Log('info', sprintf('平均VIF: %.2f', mean(vif_values)));
                obj.Logger.Log('info', sprintf('最大VIF: %.2f', max(vif_values)));
                
                % 记录高VIF变量
                high_vif_idx = find(vif_values > obj.Config.VifThreshold);
                if ~isempty(high_vif_idx)
                    obj.Logger.Log('warning', sprintf('发现 %d 个高VIF变量:', length(high_vif_idx)));
                    for i = 1:length(high_vif_idx)
                        idx = high_vif_idx(i);
                        obj.Logger.Log('warning', sprintf('  - %s: VIF = %.2f', ...
                            obj.Results.original_vars{idx}, vif_values(idx)));
                    end
                end
                
            catch ME
                obj.Logger.LogException(ME, 'CalculateVIF');
                % 使用备用方法
                vif_values = ones(n_vars, 1);
            end
        end
        
        function vif = CalculateSingleVIF(~, X, var_index)
            % 计算单个变量的VIF
            try
                % 选择其他变量作为自变量
                other_idx = setdiff(1:size(X, 2), var_index);
                X_other = X(:, other_idx);
                y_target = X(:, var_index);
                
                % 执行线性回归
                mdl = fitlm(X_other, y_target, 'Intercept', true);
                
                % 计算VIF
                r_squared = mdl.Rsquared.Ordinary;
                vif = 1 / (1 - r_squared);
                
                % 处理特殊情况
                if isinf(vif) || isnan(vif)
                    vif = 1000;  % 设置一个大值表示严重共线性
                end
                
            catch
                % 计算失败，返回默认值
                vif = 1;
            end
        end
        
        function DetectHighCorrelationPairs(obj)
            % 检测高相关变量对
            try
                R = obj.Results.correlation_matrix;
                n_vars = size(R, 1);
                
                % 找出高相关变量对
                high_corr_pairs = [];
                threshold = 0.8;  % 相关系数阈值
                
                for i = 1:n_vars-1
                    for j = i+1:n_vars
                        if abs(R(i, j)) > threshold
                            high_corr_pairs = [high_corr_pairs; i, j, R(i, j)];
                        end
                    end
                end
                
                obj.Results.high_correlation_pairs = high_corr_pairs;
                
                if ~isempty(high_corr_pairs)
                    obj.Logger.Log('warning', sprintf('发现 %d 个高相关变量对 (|r| > %.2f):', ...
                        size(high_corr_pairs, 1), threshold));
                    
                    for i = 1:size(high_corr_pairs, 1)
                        var1_idx = high_corr_pairs(i, 1);
                        var2_idx = high_corr_pairs(i, 2);
                        corr_val = high_corr_pairs(i, 3);
                        
                        obj.Logger.Log('warning', sprintf('  - %s 与 %s: r = %.3f', ...
                            obj.Results.original_vars{var1_idx}, ...
                            obj.Results.original_vars{var2_idx}, ...
                            corr_val));
                    end
                end
                
            catch ME
                obj.Logger.LogException(ME, 'DetectHighCorrelationPairs');
            end
        end
        
        function [X_cleaned, removed_vars] = HandleMulticollinearity(obj, X, var_names, vif_values)
            % 处理多重共线性
            removed_vars = false(size(X, 2), 1);
            X_cleaned = X;
            
            try
                % 检查是否需要使用PCA
                if obj.Results.condition_number > obj.Config.ConditionNumberThreshold * 2
                    obj.Logger.Log('warning', '检测到严重多重共线性，使用PCA处理');
                    [X_cleaned, removed_vars] = obj.ApplyPCA(X);
                    return;
                end
                
                % 逐步移除高VIF变量
                [X_cleaned, removed_vars] = obj.RemoveHighVIFVariables(X, var_names, vif_values);
                
                % 递归检查剩余变量
                if sum(removed_vars) > 0 && sum(~removed_vars) > 1
                    obj.Logger.Log('info', '递归检查剩余变量的VIF值');
                    [X_cleaned_rec, removed_vars_rec] = obj.RecursiveCheck(X_cleaned, var_names(~removed_vars));
                    
                    % 更新removed_vars
                    still_removed = false(size(X, 2), 1);
                    still_removed(~removed_vars) = removed_vars_rec;
                    removed_vars = removed_vars | still_removed;
                    X_cleaned = X_cleaned_rec;
                end
                
            catch ME
                obj.Logger.LogException(ME, 'HandleMulticollinearity');
                % 如果失败，使用原始数据
                X_cleaned = X;
                removed_vars = false(size(X, 2), 1);
            end
        end
        
        function [X_cleaned, removed_vars] = RemoveHighVIFVariables(obj, X, var_names, vif_values)
            % 移除高VIF变量
            removed_vars = false(size(X, 2), 1);
            X_cleaned = X;
            
            while true
                % 找出最高VIF值
                [max_vif, max_idx] = max(vif_values);
                
                if max_vif <= obj.Config.VifThreshold
                    break;  % 所有变量VIF都在阈值内
                end
                
                % 标记移除该变量
                removed_vars(max_idx) = true;
                
                obj.Logger.Log('info', sprintf('移除变量: %s (VIF = %.2f)', var_names{max_idx}, max_vif));
                
                % 更新数据
                X_cleaned = X_cleaned(:, ~removed_vars);
                
                if size(X_cleaned, 2) < 2
                    obj.Logger.Log('warning', '移除变量导致剩余变量过少，停止移除');
                    break;
                end
                
                % 重新计算VIF
                vif_values_new = obj.CalculateVIF(X_cleaned);
                
                % 更新原始VIF向量
                temp_vif = zeros(size(vif_values));
                temp_vif(~removed_vars) = vif_values_new;
                vif_values = temp_vif;
            end
        end
        
        function [X_pca, removed_vars] = ApplyPCA(obj, X)
            % 使用主成分分析处理多重共线性
            try
                obj.Logger.Log('info', '开始PCA降维处理');
                
                % 执行PCA
                [coeff, score, ~, ~, explained, mu] = pca(X, 'Algorithm', 'svd');
                
                % 确定保留的主成分数量
                cum_var = cumsum(explained);
                n_components = find(cum_var >= obj.Config.PcaVarianceThreshold, 1, 'first');
                
                if isempty(n_components)
                    n_components = size(X, 2);  % 保留所有成分
                end
                
                % 选择主成分
                X_pca = score(:, 1:n_components);
                
                % 记录PCA结果
                obj.Results.pca_results = struct();
                obj.Results.pca_results.coefficients = coeff;
                obj.Results.pca_results.explained_variance = explained;
                obj.Results.pca_results.cumulative_variance = cum_var;
                obj.Results.pca_results.n_components = n_components;
                obj.Results.pca_results.variance_threshold = obj.Config.PcaVarianceThreshold;
                
                obj.Logger.Log('info', sprintf('PCA降维完成：从 %d 个变量降至 %d 个主成分', ...
                    size(X, 2), n_components));
                obj.Logger.Log('info', sprintf('保留 %.1f%% 的方差', cum_var(n_components)));
                
                % 所有原始变量都被"移除"（转换为主成分）
                removed_vars = true(size(X, 2), 1);
                
            catch ME
                obj.Logger.LogException(ME, 'ApplyPCA');
                % 如果PCA失败，返回原始数据
                X_pca = X;
                removed_vars = false(size(X, 2), 1);
            end
        end
        
        function [X_cleaned, removed_vars] = RecursiveCheck(obj, X, var_names)
            % 递归检查多重共线性
            n_iterations = 0;
            max_iterations = 10;  % 防止无限循环
            
            removed_vars = false(size(X, 2), 1);
            X_cleaned = X;
            
            while n_iterations < max_iterations
                n_iterations = n_iterations + 1;
                
                % 计算VIF
                vif_values = obj.CalculateVIF(X_cleaned);
                
                % 检查是否还有高VIF变量
                high_vif_idx = find(vif_values > obj.Config.VifThreshold);
                
                if isempty(high_vif_idx)
                    break;  % 没有高VIF变量了
                end
                
                % 移除VIF最高的变量
                [~, remove_idx] = max(vif_values);
                
                obj.Logger.Log('debug', sprintf('递归检查第 %d 轮：移除 %s (VIF = %.2f)', ...
                    n_iterations, var_names{remove_idx}, vif_values(remove_idx)));
                
                % 更新标记
                removed_vars(remove_idx) = true;
                
                % 更新数据
                X_cleaned = X_cleaned(:, ~removed_vars);
                var_names = var_names(~removed_vars);
                
                if size(X_cleaned, 2) < 2
                    obj.Logger.Log('warning', '递归检查导致剩余变量过少，停止检查');
                    break;
                end
            end
            
            if n_iterations >= max_iterations
                obj.Logger.Log('warning', '递归检查达到最大迭代次数，可能仍存在共线性');
            end
        end
        
        function ValidateCleanedMatrix(obj, X_cleaned)
            % 验证清理后的矩阵
            try
                % 检查维度
                if size(X_cleaned, 2) < 1
                    obj.Logger.Log('error', '清理后没有剩余变量');
                    return;
                end
                
                % 检查相关矩阵条件数
                try
                    R_cleaned = corr(X_cleaned);
                    cond_num_cleaned = cond(R_cleaned);
                    
                    obj.Logger.Log('info', sprintf('清理后相关矩阵条件数: %.2f', cond_num_cleaned));
                    
                    if cond_num_cleaned > obj.Config.ConditionNumberThreshold
                        obj.Logger.Log('warning', '清理后仍存在多重共线性问题');
                    end
                catch
                    obj.Logger.Log('warning', '无法计算清理后矩阵的条件数');
                end
                
                % 检查方差
                variances = var(X_cleaned);
                if any(variances < 1e-10)
                    obj.Logger.Log('warning', '清理后存在零方差变量');
                end
                
                % 保存验证结果
                obj.Results.validation = struct();
                obj.Results.validation.final_variables = size(X_cleaned, 2);
                obj.Results.validation.condition_number_final = cond_num_cleaned;
                obj.Results.validation.min_variance = min(variances);
                
            catch ME
                obj.Logger.LogException(ME, 'ValidateCleanedMatrix');
            end
        end
        
        function GenerateCollinearityReport(obj, var_names, removed_vars)
            % 生成多重共线性报告
            try
                % 创建报告结构
                report = struct();
                report.original_variables = length(var_names);
                report.removed_variables = sum(removed_vars);
                report.final_variables = sum(~removed_vars);
                report.removal_rate = report.removed_variables / report.original_variables * 100;
                
                % 添加详细信息
                if ~isempty(removed_vars)
                    report.removed_variable_names = var_names(removed_vars);
                    report.kept_variable_names = var_names(~removed_vars);
                end
                
                % 添加统计信息
                report.original_condition_number = obj.Results.condition_number;
                if isfield(obj.Results, 'validation')
                    report.final_condition_number = obj.Results.validation.condition_number_final;
                end
                
                % 添加VIF信息
                if isfield(obj.Results, 'vif_values')
                    report.max_original_vif = max(obj.Results.vif_values);
                    report.avg_original_vif = mean(obj.Results.vif_values);
                end
                
                % 保存报告
                obj.Results.report = report;
                
                % 记录摘要信息
                obj.Logger.Log('info', '=== 多重共线性检查报告 ===');
                obj.Logger.Log('info', sprintf('原始变量数: %d', report.original_variables));
                obj.Logger.Log('info', sprintf('移除变量数: %d', report.removed_variables));
                obj.Logger.Log('info', sprintf('最终变量数: %d', report.final_variables));
                obj.Logger.Log('info', sprintf('移除率: %.1f%%', report.removal_rate));
                
                if isfield(report, 'removed_variable_names') && ~isempty(report.removed_variable_names)
                    obj.Logger.Log('info', '移除的变量:');
                    for i = 1:length(report.removed_variable_names)
                        obj.Logger.Log('info', sprintf('  - %s', report.removed_variable_names{i}));
                    end
                end
                
            catch ME
                obj.Logger.LogException(ME, 'GenerateCollinearityReport');
            end
        end
        
        function SaveCorrelationHeatmap(obj, output_dir)
            % 保存相关矩阵热图
            try
                if isfield(obj.Results, 'correlation_matrix')
                    fig = figure('Visible', 'off');
                    imagesc(obj.Results.correlation_matrix, [-1, 1]);
                    colorbar;
                    colormap('RdBu_r');
                    
                    title('变量相关性矩阵');
                    xlabel('变量');
                    ylabel('变量');
                    
                    % 添加变量名标签
                    if length(obj.Results.original_vars) <= 20
                        set(gca, 'XTick', 1:length(obj.Results.original_vars), ...
                            'XTickLabel', obj.Results.original_vars, ...
                            'XTickLabelRotation', 45);
                        set(gca, 'YTick', 1:length(obj.Results.original_vars), ...
                            'YTickLabel', obj.Results.original_vars);
                    end
                    
                    % 保存图形
                    saveas(fig, fullfile(output_dir, 'correlation_heatmap.png'), 'png');
                    saveas(fig, fullfile(output_dir, 'correlation_heatmap.svg'), 'svg');
                    
                    close(fig);
                    obj.Logger.Log('info', '相关矩阵热图已保存');
                end
                
            catch ME
                obj.Logger.LogException(ME, 'SaveCorrelationHeatmap');
            end
        end
    end
end