%% reporting_module.m - 报告和结果输出模块
classdef reporting_module
    methods(Static)
        function save_enhanced_results(results, var_names, group_means, cv_results, coef_stability, param_stats, var_contribution)
            % 保存分析结果，包括变量组合信息、交叉验证结果、系数稳定性、参数统计和变量贡献
            % 输入:
            %   results - 结果结构
            %   var_names - 变量名称
            %   group_means - 分组均值
            %   cv_results - 交叉验证结果
            %   coef_stability - 系数稳定性
            %   param_stats - 参数统计
            %   var_contribution - 变量贡献
            
            t_start = toc;
            
            % 从results结构中获取方法名称
            methods = fieldnames(results);
            
            % 创建高级结果目录
            result_dir = fullfile('results');
            if ~exist(result_dir, 'dir')
                mkdir(result_dir);
            end
            
            % 保存时间戳
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            result_mat = fullfile(result_dir, sprintf('enhanced_analysis_%s.mat', timestamp));
            
            % 保存原始结果 - 使用-v7.3支持大文件
            try
                save(result_mat, 'results', 'var_names', 'group_means', 'cv_results', ...
                     'coef_stability', 'param_stats', 'var_contribution', '-v7.3');
                logger.log_message('info', sprintf('增强分析结果已保存至 %s', result_mat));
            catch ME
                logger.log_message('error', sprintf('保存MAT文件时出错: %s', ME.message));
            end
            
            % 创建CSV结果目录
            csv_dir = fullfile(result_dir, 'csv');
            if ~exist(csv_dir, 'dir')
                mkdir(csv_dir);
            end
            
            % 创建图形结果目录
            figure_dir = fullfile(result_dir, 'figures');
            if ~exist(figure_dir, 'dir')
                mkdir(figure_dir);
            end
            
            % 创建报告目录
            report_dir = fullfile(result_dir, 'reports');
            if ~exist(report_dir, 'dir')
                mkdir(report_dir);
            end
            
            % 创建CSV统计表
            reporting_module.save_csv_tables(results, methods, var_names, cv_results, coef_stability, param_stats, var_contribution, csv_dir);
            
            % 创建可视化图表
            reporting_module.create_visualizations(results, methods, var_names, cv_results, coef_stability, param_stats, var_contribution, figure_dir);
            
            % 创建综合比较报告
            reporting_module.create_enhanced_summary_report(results, methods, var_names, cv_results, coef_stability, param_stats, var_contribution, report_dir);
            
            t_end = toc;
            logger.log_message('info', sprintf('结果保存完成，耗时：%.2f秒', t_end - t_start));
        end
        
        function save_csv_tables(results, methods, var_names, cv_results, coef_stability, param_stats, var_contribution, csv_dir)
            % 保存CSV格式的统计表格
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            %   cv_results - 交叉验证结果
            %   coef_stability - 系数稳定性
            %   param_stats - 参数统计
            %   var_contribution - 变量贡献
            %   csv_dir - CSV保存目录
            
            % 创建主要统计表格
            try
                % AIC和BIC比较表
                file_path = fullfile(csv_dir, 'aic_bic_comparison.csv');
                if ~exist(file_path, 'file')
                    aic_bic_table = reporting_module.create_aic_bic_table(results, methods);
                    writetable(aic_bic_table, file_path);
                    logger.log_message('info', 'AIC和BIC比较表已保存');
                end
                
                % 变量选择频率表
                file_path = fullfile(csv_dir, 'variable_selection_frequency.csv');
                if ~exist(file_path, 'file')
                    var_freq_table = reporting_module.create_var_freq_table(results, methods, var_names);
                    writetable(var_freq_table, file_path);
                    logger.log_message('info', '变量选择频率表已保存');
                end
                
                % 详细模型性能表
                file_path = fullfile(csv_dir, 'model_performance_detailed.csv');
                if ~exist(file_path, 'file')
                    perf_detail_table = reporting_module.create_performance_detail_table(results, methods);
                    writetable(perf_detail_table, file_path);
                    logger.log_message('info', '详细模型性能表已保存');
                end
                
                % 变量组合性能表
                file_path = fullfile(csv_dir, 'variable_group_performance.csv');
                if ~exist(file_path, 'file')
                    var_group_table = reporting_module.create_variable_group_table(results, methods);
                    writetable(var_group_table, file_path);
                    logger.log_message('info', '变量组合性能表已保存');
                end
                
                % K折交叉验证结果表
                file_path = fullfile(csv_dir, 'cv_results.csv');
                if ~exist(file_path, 'file')
                    cv_table = reporting_module.create_cv_results_table(cv_results);
                    writetable(cv_table, file_path);
                    logger.log_message('info', 'K折交叉验证结果表已保存');
                end
                
                % 模型参数表
                file_path = fullfile(csv_dir, 'model_parameters.csv');
                if ~exist(file_path, 'file')
                    param_table = reporting_module.create_parameter_table(results, methods);
                    writetable(param_table, file_path);
                    logger.log_message('info', '模型参数表已保存');
                end
            catch ME
                logger.log_message('error', sprintf('创建基本统计表时出错: %s', ME.message));
            end
            
            % 保存系数稳定性表和参数统计表
            try
                % 各方法系数稳定性表
                for m = 1:length(methods)
                    method = methods{m};
                    file_path = fullfile(csv_dir, sprintf('%s_coefficient_stability.csv', method));
                    if ~exist(file_path, 'file')
                        if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
                            writetable(coef_stability.(method).table, file_path);
                            logger.log_message('info', sprintf('%s方法的系数稳定性表已保存', method));
                        end
                    end
                end
                
                % 各方法参数统计表
                for m = 1:length(methods)
                    method = methods{m};
                    file_path = fullfile(csv_dir, sprintf('%s_parameter_statistics.csv', method));
                    if ~exist(file_path, 'file')
                        if isfield(param_stats, method) && isfield(param_stats.(method), 'table')
                            writetable(param_stats.(method).table, file_path);
                            logger.log_message('info', sprintf('%s方法的参数统计表已保存', method));
                        end
                    end
                end
            catch ME
                logger.log_message('error', sprintf('创建系数和参数统计表时出错: %s', ME.message));
            end
            
            % 保存变量贡献相关表
            try
                % 全局重要性表
                if isfield(var_contribution, 'correlation')
                    file_path = fullfile(csv_dir, 'correlation_importance.csv');
                    if ~exist(file_path, 'file')
                        writetable(var_contribution.correlation, file_path);
                    end
                end
                
                if isfield(var_contribution, 'logistic')
                    file_path = fullfile(csv_dir, 'logistic_importance.csv');
                    if ~exist(file_path, 'file')
                        writetable(var_contribution.logistic, file_path);
                    end
                end
                
                if isfield(var_contribution, 'randomforest')
                    file_path = fullfile(csv_dir, 'randomforest_importance.csv');
                    if ~exist(file_path, 'file')
                        writetable(var_contribution.randomforest, file_path);
                    end
                end
                
                if isfield(var_contribution, 'overall_importance')
                    file_path = fullfile(csv_dir, 'overall_importance.csv');
                    if ~exist(file_path, 'file')
                        writetable(var_contribution.overall_importance, file_path);
                    end
                end
                
                % 方法特定贡献表
                for m = 1:length(methods)
                    method = methods{m};
                    file_path = fullfile(csv_dir, sprintf('%s_variable_contribution.csv', method));
                    if ~exist(file_path, 'file')
                        if isfield(var_contribution, 'methods') && isfield(var_contribution.methods, method) && ...
                           isfield(var_contribution.methods.(method), 'contribution_table')
                            writetable(var_contribution.methods.(method).contribution_table, file_path);
                            logger.log_message('info', sprintf('%s方法的变量贡献表已保存', method));
                        end
                    end
                end
                
                logger.log_message('info', '变量贡献表已保存');
            catch ME
                logger.log_message('error', sprintf('创建变量贡献表时出错: %s', ME.message));
            end
        end
        
        function create_visualizations(results, methods, var_names, cv_results, coef_stability, param_stats, var_contribution, figure_dir)
            % 创建所有可视化图表
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            %   cv_results - 交叉验证结果
            %   coef_stability - 系数稳定性
            %   param_stats - 参数统计
            %   var_contribution - 变量贡献
            %   figure_dir - 图形保存目录
            
            % ROC曲线图
            try
                reporting_module.create_roc_curves(results, methods, figure_dir);
                logger.log_message('info', 'ROC曲线图已保存');
            catch ME
                logger.log_message('error', sprintf('创建ROC曲线图时出错: %s', ME.message));
            end
            
            % 变量重要性图
            try
                reporting_module.create_variable_importance_plot(results, methods, var_names, figure_dir);
                logger.log_message('info', '变量重要性图已保存');
            catch ME
                logger.log_message('error', sprintf('创建变量重要性图时出错: %s', ME.message));
            end
            
            % 变量组合可视化图
            try
                reporting_module.create_variable_group_plot(results, methods, var_names, figure_dir);
                logger.log_message('info', '变量组合可视化图已保存');
            catch ME
                logger.log_message('error', sprintf('创建变量组合可视化图时出错: %s', ME.message));
            end
            
            % K折交叉验证性能图
            try
                reporting_module.create_cv_performance_plot(cv_results, figure_dir);
                logger.log_message('info', 'K折交叉验证性能图已保存');
            catch ME
                logger.log_message('error', sprintf('创建K折交叉验证性能图时出错: %s', ME.message));
            end
            
            % 系数稳定性图
            try
                reporting_module.create_coefficient_stability_plot(coef_stability, methods, figure_dir);
                logger.log_message('info', '系数稳定性图已保存');
            catch ME
                logger.log_message('error', sprintf('创建系数稳定性图时出错: %s', ME.message));
            end
            
            % 变量贡献图
            try
                reporting_module.create_variable_contribution_plot(var_contribution, figure_dir);
                logger.log_message('info', '变量贡献图已保存');
            catch ME
                logger.log_message('error', sprintf('创建变量贡献图时出错: %s', ME.message));
            end
            
            % 箱线图可视化
            try
                reporting_module.create_boxplot_visualization(results, methods, figure_dir);
                logger.log_message('info', '箱线图可视化已保存');
            catch ME
                logger.log_message('error', sprintf('创建箱线图可视化时出错: %s', ME.message));
            end
            
            % PR曲线
            try
                reporting_module.create_pr_curves(results, methods, figure_dir);
                logger.log_message('info', 'PR曲线图已保存');
            catch ME
                logger.log_message('error', sprintf('创建PR曲线图时出错: %s', ME.message));
            end
            
            % 校准曲线
            try
                reporting_module.create_calibration_curves(results, methods, figure_dir);
                logger.log_message('info', '校准曲线图已保存');
            catch ME
                logger.log_message('error', sprintf('创建校准曲线图时出错: %s', ME.message));
            end
            
            % 混淆矩阵
            try
                reporting_module.create_confusion_matrices(results, methods, figure_dir);
                logger.log_message('info', '混淆矩阵图已保存');
            catch ME
                logger.log_message('error', sprintf('创建混淆矩阵图时出错: %s', ME.message));
            end
        end
        
        %% reporting_module.m - 报告和结果输出模块（续）
        function aic_bic_table = create_aic_bic_table(results, methods)
            % 创建AIC和BIC比较表
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            % 输出:
            %   aic_bic_table - AIC和BIC比较表
            
            % 初始化变量
            methods_cell = cell(length(methods), 1);
            aic_values = zeros(length(methods), 1);
            aic_std_values = zeros(length(methods), 1);
            bic_values = zeros(length(methods), 1);
            bic_std_values = zeros(length(methods), 1);
            n_params = zeros(length(methods), 1);
            
            % 提取每种方法的AIC和BIC
            for i = 1:length(methods)
                method = methods{i};
                methods_cell{i} = method;
                
                % 检查该方法是否有AIC和BIC值
                if isfield(results.(method).performance, 'avg_aic') && ...
                   isfield(results.(method).performance, 'avg_bic')
                    aic_values(i) = results.(method).performance.avg_aic;
                    aic_std_values(i) = results.(method).performance.std_aic;
                    bic_values(i) = results.(method).performance.avg_bic;
                    bic_std_values(i) = results.(method).performance.std_bic;
                    
                    % 获取参数数量（基于最常见的变量组合）
                    selected_vars = find(results.(method).selected_vars);
                    n_params(i) = length(selected_vars) + 1; % +1是因为有截距项
                else
                    aic_values(i) = NaN;
                    aic_std_values(i) = NaN;
                    bic_values(i) = NaN;
                    bic_std_values(i) = NaN;
                    n_params(i) = 0;
                end
            end
            
            % 创建表格
            aic_bic_table = table(methods_cell, n_params, aic_values, aic_std_values, bic_values, bic_std_values, ...
                'VariableNames', {'Method', 'NumParams', 'AIC', 'AIC_StdDev', 'BIC', 'BIC_StdDev'});
            
            % 按AIC排序
            aic_bic_table = sortrows(aic_bic_table, 'AIC', 'ascend');
        end
        
        function perf_detail_table = create_performance_detail_table(results, methods)
            % 创建详细模型性能表，包括均值和标准差
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            % 输出:
            %   perf_detail_table - 详细性能表
            
            % 初始化变量
            methods_cell = cell(length(methods), 1);
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
            metric_names = {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'AUC'};
            
            % 初始化数组
            mean_values = zeros(length(methods), length(metrics));
            std_values = zeros(length(methods), length(metrics));
            cv_values = zeros(length(methods), length(metrics));
            
            % 提取每种方法的性能指标
            for i = 1:length(methods)
                method = methods{i};
                methods_cell{i} = method;
                
                for j = 1:length(metrics)
                    metric = metrics{j};
                    
                    % 均值
                    if isfield(results.(method).performance, ['avg_' metric])
                        mean_values(i, j) = results.(method).performance.(['avg_' metric]);
                    else
                        mean_values(i, j) = NaN;
                    end
                    
                    % 标准差
                    if isfield(results.(method).performance, ['std_' metric])
                        std_values(i, j) = results.(method).performance.(['std_' metric]);
                    else
                        std_values(i, j) = NaN;
                    end
                    
                    % 变异系数
                    if mean_values(i, j) > 0
                        cv_values(i, j) = std_values(i, j) / mean_values(i, j);
                    else
                        cv_values(i, j) = NaN;
                    end
                end
            end
            
            % 创建表格
            table_vars = {'Method'};
            data_vars = {methods_cell};
            
            % 添加各指标的均值、标准差和变异系数
            for j = 1:length(metrics)
                table_vars{end+1} = [metric_names{j} '_Mean'];
                data_vars{end+1} = mean_values(:, j);
                
                table_vars{end+1} = [metric_names{j} '_StdDev'];
                data_vars{end+1} = std_values(:, j);
                
                table_vars{end+1} = [metric_names{j} '_CV'];
                data_vars{end+1} = cv_values(:, j);
            end
            
            % 创建表格
            perf_detail_table = table(data_vars{:}, 'VariableNames', table_vars);
            
            % 按F1分数均值排序
            f1_col = find(strcmp(table_vars, 'F1_Score_Mean'));
            perf_detail_table = sortrows(perf_detail_table, f1_col, 'descend');
        end
        
        function var_freq_table = create_var_freq_table(results, methods, var_names)
            % 创建变量选择频率表
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   var_freq_table - 变量频率表
            
            % 初始化变量名称列
            var_names_cell = cell(length(var_names), 1);
            for i = 1:length(var_names)
                var_names_cell{i} = var_names{i};
            end
            
            % 初始化表格
            var_freq_table = table(var_names_cell, 'VariableNames', {'VariableName'});
            
            % 添加各方法的频率
            for i = 1:length(methods)
                method = methods{i};
                var_freq = results.(method).var_freq;
                
                % 确保长度一致
                if length(var_freq) ~= length(var_names)
                    logger.log_message('warning', sprintf('%s方法的变量频率长度(%d)与变量名长度(%d)不匹配，进行调整', method, length(var_freq), length(var_names)));
                    
                    % 如果var_freq较短，扩展它
                    if length(var_freq) < length(var_names)
                        var_freq_extended = zeros(length(var_names), 1);
                        var_freq_extended(1:length(var_freq)) = var_freq;
                        var_freq = var_freq_extended;
                    % 如果var_freq较长，截断它
                    else
                        var_freq = var_freq(1:length(var_names));
                    end
                end
                
                % 确保var_freq是列向量
                if size(var_freq, 2) > 1
                    var_freq = var_freq';
                end
                
                % 添加到表格中
                var_freq_table.(method) = var_freq;
            end
            
            % 添加平均频率
            avg_vals = zeros(height(var_freq_table), 1);
            for i = 1:length(methods)
                method = methods{i};
                avg_vals = avg_vals + var_freq_table.(method);
            end
            avg_vals = avg_vals / length(methods);
            var_freq_table.Average = avg_vals;
            
            % 对表格进行排序
            var_freq_table = sortrows(var_freq_table, 'Average', 'descend');
        end
        
        function var_group_table = create_variable_group_table(results, methods)
            % 创建变量组合性能表
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            % 输出:
            %   var_group_table - 变量组合性能表
            
            % 初始化表格行
            rows = [];
            
            for i = 1:length(methods)
                method = methods{i};
                group_perf = results.(method).group_performance;
                
                % 对于每个变量组合创建一行
                for j = 1:length(group_perf)
                    combo = group_perf(j);
                    var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
                    row = {method, var_str, combo.count, combo.accuracy, combo.sensitivity, combo.specificity, combo.precision, combo.f1_score, combo.auc};
                    rows = [rows; row];
                end
            end
            
            % 创建表格 - 将"Variables"改为"VarCombination"并增加新指标
            var_group_table = cell2table(rows, 'VariableNames', {'Method', 'VarCombination', 'Count', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC'});
            
            % 对表格进行排序（先按方法，再按F1分数）
            var_group_table = sortrows(var_group_table, {'Method', 'F1_Score'}, {'ascend', 'descend'});
        end
        
        function cv_table = create_cv_results_table(cv_results)
            % 创建K折交叉验证结果表
            % 输入:
            %   cv_results - 交叉验证结果
            % 输出:
            %   cv_table - 交叉验证结果表
            
            % 提取每个折的性能
            n_folds = length(cv_results.accuracy);
            fold_numbers = (1:n_folds)';
            
            % 初始化表格
            cv_table = table(fold_numbers, 'VariableNames', {'Fold'});
            
            % 添加各性能指标
            cv_table.Accuracy = cv_results.accuracy;
            cv_table.Precision = cv_results.precision;
            cv_table.Recall = cv_results.recall;
            cv_table.Specificity = cv_results.specificity;
            cv_table.F1_Score = cv_results.f1_score;
            cv_table.AUC = cv_results.auc;
            
            % 添加均值和标准差行
            mean_row = table('Size', [1 size(cv_table,2)], 'VariableTypes', repmat({'double'}, 1, size(cv_table,2)), 'VariableNames', cv_table.Properties.VariableNames);
            std_row = mean_row;
            
            mean_row.Fold = 0; % 用0表示均值行
            std_row.Fold = -1; % 用-1表示标准差行
            
            % 填充数据
            metrics = {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'AUC'};
            for i = 1:length(metrics)
                metric = metrics{i};
                mean_row.(metric) = mean(cv_table.(metric), 'omitnan');
                std_row.(metric) = std(cv_table.(metric), 'omitnan');
            end
            
            % 合并表格
            cv_table = [cv_table; mean_row; std_row];
        end
        
        function param_table = create_parameter_table(results, methods)
            % 创建模型参数表
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            % 输出:
            %   param_table - 参数表
            
            % 初始化表格行
            rows = [];
            
            for i = 1:length(methods)
                method = methods{i};
                params = results.(method).params;
                
                % 对于每个模型
                for j = 1:length(params.coef_cell)
                    coef = params.coef_cell{j};
                    pval = params.pval_cell{j};
                    vars = params.var_cell{j};
                    
                    % 对于每个变量
                    for k = 1:length(coef)
                        if k <= length(vars)
                            var_name = vars{k};
                            row = {method, j, var_name, coef(k), pval(k)};
                            rows = [rows; row];
                        end
                    end
                end
            end
            
            % 创建表格
            param_table = cell2table(rows, 'VariableNames', {'Method', 'Model_Index', 'Variable', 'Coefficient', 'P_Value'});
        end
        
        function create_roc_curves(results, methods, figure_dir)
            % 创建ROC曲线图
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
            
            fig = figure('Name', 'ROC Curves', 'Position', [100, 100, 1000, 800]);
            
            % 禁用工具栏
            set(gcf, 'Toolbar', 'none');
        
            colors = lines(length(methods));
            markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
            hold on;
        
            legend_entries = cell(length(methods), 1);
        
            for i = 1:length(methods)
                method = methods{i};
                auc = results.(method).performance.avg_auc;
                sensitivity = results.(method).performance.avg_sensitivity;
                specificity = results.(method).performance.avg_specificity;
                precision = results.(method).performance.avg_precision;
                f1_score = results.(method).performance.avg_f1_score;
        
                fpr = 1 - specificity;
                color_idx = mod(i-1, size(colors, 1)) + 1;
                marker_idx = mod(i-1, length(markers)) + 1;
        
                plot(fpr, sensitivity, [markers{marker_idx}], 'Color', colors(color_idx,:), ...
                    'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
                plot([0, fpr, 1], [0, sensitivity, 1], '-', 'Color', colors(color_idx,:), 'LineWidth', 1.5);
        
                legend_entries{i} = sprintf('%s (AUC=%.3f, F1=%.3f)', method, auc, f1_score);
            end
        
            plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);
        
            xlim([0, 1]);
            ylim([0, 1]);
            xlabel('假阳性率 (1 - 特异性)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('真阳性率 (敏感性)', 'FontSize', 12, 'FontWeight', 'bold');
            title('不同变量选择方法的ROC曲线比较', 'FontSize', 14, 'FontWeight', 'bold');
            legend([legend_entries; {'随机猜测'}], 'Location', 'southeast', 'FontSize', 10);
            grid on;
            set(gca, 'FontSize', 11);
            box on;
        
            text(0.05, 0.95, '注: 点表示在测试集上的平均性能', 'FontSize', 9);
        
            set(gcf, 'Color', 'white');
            set(gca, 'TickDir', 'out');
        
            % 设置纸张属性
            set(gcf, 'PaperPositionMode', 'manual');
            set(gcf, 'PaperUnits', 'inches');
            set(gcf, 'PaperSize', [12 8]); % 设置为 12x8 英寸
            set(gcf, 'PaperPosition', [0 0 12 8]);
        
            % 保存矢量图
            utils.save_figure(fig, figure_dir, 'roc_curves', 'Formats', {'svg'});
            close(fig);
        end
        
        function create_variable_importance_plot(results, methods, var_names, figure_dir)
            % 创建变量重要性图
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            %   figure_dir - 图形保存目录
            
            % 计算每个变量的平均频率
            var_freq = zeros(length(var_names), 1);
            for i = 1:length(methods)
                method = methods{i};
                var_freq = var_freq + results.(method).var_freq;
            end
            var_freq = var_freq / length(methods);
        
            [sorted_freq, idx] = sort(var_freq, 'descend');
            sorted_names = var_names(idx);
        
            fig = figure('Name', 'Variable Importance', 'Position', [100, 100, 900, 700]);
            
            % 禁用工具栏
            set(gcf, 'Toolbar', 'none');
        
            h = barh(sorted_freq);
            set(h, 'FaceColor', 'flat');
        
            colormap(autumn);
            for i = 1:length(sorted_freq)
                h.CData(i,:) = [sorted_freq(i), 0.5, 1-sorted_freq(i)];
            end
        
            set(gca, 'YTick', 1:length(sorted_names), 'YTickLabel', sorted_names, 'FontSize', 10);
            xlabel('选择频率', 'FontSize', 12, 'FontWeight', 'bold');
            title('不同方法中变量重要性比较', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            box on;
        
            for i = 1:length(sorted_freq)
                text(sorted_freq(i) + 0.03, i, sprintf('%.2f', sorted_freq(i)), ...
                    'VerticalAlignment', 'middle', 'FontSize', 9);
            end
        
            text(0.5, length(sorted_names) + 1.5, ...
                ['注: 此图显示每个变量在', num2str(length(methods)), '种方法中的平均选择频率'], ...
                'FontSize', 9, 'HorizontalAlignment', 'center');
        
            set(gcf, 'Color', 'white');
            set(gca, 'TickDir', 'out');
            set(gcf, 'Position', [100, 100, 900, max(500, 150 + 30*length(sorted_names))]);
        
            % 设置纸张属性
            set(gcf, 'PaperPositionMode', 'manual');
            set(gcf, 'PaperUnits', 'inches');
            set(gcf, 'PaperSize', [10 8]); % 设置为 10x8 英寸
            set(gcf, 'PaperPosition', [0 0 10 8]);
        
            % 保存矢量图
            utils.save_figure(fig, figure_dir, 'variable_importance', 'Formats', {'svg'});
            close(fig);
        end
        
        function create_variable_group_plot(results, methods, var_names, figure_dir)
            % 创建变量组合可视化图
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            %   figure_dir - 图形保存目录
        
            % 对于每种方法创建图形
            for i = 1:length(methods)
                method = methods{i};
                group_perf = results.(method).group_performance;
                
                % 如果有至少3个不同的组合，则创建图
                if length(group_perf) >= 2
                    % 取前10个最常见的组合
                    top_n = min(10, length(group_perf));
                    
                    % 提取数据
                    combo_labels = cell(top_n, 1);
                    combo_counts = zeros(top_n, 1);
                    combo_aucs = zeros(top_n, 1);
                    combo_acc = zeros(top_n, 1);
                    combo_sens = zeros(top_n, 1);
                    combo_spec = zeros(top_n, 1);
                    combo_prec = zeros(top_n, 1);    
                    combo_f1 = zeros(top_n, 1);
                    
                    for j = 1:top_n
                        combo = group_perf(j);
                        var_str = sprintf('组合 %d', j);
                        combo_labels{j} = var_str;
                        combo_counts(j) = combo.count;
                        combo_aucs(j) = combo.auc;
                        combo_acc(j) = combo.accuracy;
                        combo_sens(j) = combo.sensitivity;
                        combo_spec(j) = combo.specificity;
                        combo_prec(j) = combo.precision;
                        combo_f1(j) = combo.f1_score;
                    end
                    
                    % 创建组合性能图
                    fig1 = figure('Name', sprintf('%s Variable Combinations', method), 'Position', [100, 100, 1200, 800]);
                    
                    % 创建子图1：组合计数
                    subplot(2, 2, 1);
                    bar(combo_counts, 'FaceColor', [0.3 0.6 0.9]);
                    set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
                    title([method, ': 变量组合出现频率'], 'FontSize', 12, 'FontWeight', 'bold');
                    ylabel('频率', 'FontSize', 10);
                    grid on;
                    
                    % 创建子图2：组合AUC和F1分数
                    subplot(2, 2, 2);
                    metrics_2 = [combo_aucs, combo_f1];
                    h2 = bar(metrics_2);
                    set(h2(1), 'FaceColor', [0.9 0.4 0.3]);
                    set(h2(2), 'FaceColor', [0.3 0.8 0.8]);
                    set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
                    title([method, ': 变量组合AUC和F1值'], 'FontSize', 12, 'FontWeight', 'bold');
                    ylabel('值', 'FontSize', 10);
                    legend({'AUC', 'F1分数'}, 'Location', 'southwest', 'FontSize', 8);
                    grid on;
                    
                    % 创建子图3：准确率、敏感性和特异性
                    subplot(2, 2, 3);
                    metrics_3 = [combo_acc, combo_sens, combo_spec];
                    h3 = bar(metrics_3);
                    set(h3(1), 'FaceColor', [0.3 0.8 0.3]);
                    set(h3(2), 'FaceColor', [0.9 0.6 0.1]);
                    set(h3(3), 'FaceColor', [0.5 0.5 0.8]);
                    set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
                    title([method, ': 组合性能指标1'], 'FontSize', 12, 'FontWeight', 'bold');
                    ylabel('值', 'FontSize', 10);
                    legend({'准确率', '敏感性', '特异性'}, 'Location', 'southeast', 'FontSize', 8);
                    grid on;
                    
                    % 创建子图4：精确率（新增）
                    subplot(2, 2, 4);
                    bar(combo_prec, 'FaceColor', [0.6 0.3 0.8]);
                    set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
                    title([method, ': 变量组合精确率'], 'FontSize', 12, 'FontWeight', 'bold');
                    ylabel('精确率', 'FontSize', 10);
                    grid on;
                    
                    % 调整整体间距
                    set(gcf, 'Color', 'white');
                    set(fig1, 'Position', [100, 100, 1200, 800]);
                    
                    % 保存矢量图 - 注意这里传入方法名
                    utils.save_figure(fig1, figure_dir, '%s_variable_combinations', 'MethodName', method, 'Formats', {'svg'});
                    close(fig1);
                    
                    % 创建组合详情图
                    fig2 = figure('Name', sprintf('%s Combination Details', method), 'Position', [100, 100, 1200, 800]);
                    
                    % 创建矩阵显示每个组合包含哪些变量
                    combo_matrix = zeros(top_n, length(var_names));
                    
                    for j = 1:top_n
                        combo = group_perf(j);
                        for k = 1:length(combo.variables)
                            var_name = combo.variables{k};
                            var_idx = find(strcmp(var_names, var_name));
                            if ~isempty(var_idx)
                                combo_matrix(j, var_idx) = 1;
                            end
                        end
                    end
                    
                    % 绘制热图
                    h = heatmap(combo_matrix, 'XDisplayLabels', var_names, 'YDisplayLabels', combo_labels);
                    
                    % 自定义颜色映射
                    colormap([1 1 1; 0.2 0.6 0.8]); % 白色和蓝色
                    
                    % 设置标题和标签
                    h.Title = sprintf('%s: 前%d个组合的变量构成', method, top_n);
                    h.XLabel = '变量';
                    h.YLabel = '组合';
                    h.FontSize = 10;
                    
                    % 保存矢量图 - 注意这里传入方法名
                    utils.save_figure(fig2, figure_dir, '%s_combination_details', 'MethodName', method, 'Formats', {'svg'});
                    close(fig2);
                else
                    logger.log_message('info', sprintf('%s方法的变量组合少于3个，跳过可视化', method));
                end
            end
        
            % 创建所有方法的综合比较图
            try
                % 从所有方法中收集组合信息
                all_combos = struct('method', {}, 'vars', {}, 'auc', {}, 'f1', {}, 'count', {});
                for i = 1:length(methods)
                    method = methods{i};
                    group_perf = results.(method).group_performance;
                    
                    for j = 1:min(3, length(group_perf))  % 每个方法取前3个
                        if group_perf(j).count >= 5 % 只考虑出现至少5次的组合
                            combo = group_perf(j);
                            var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
                            new_combo = struct('method', method, 'vars', var_str, 'auc', combo.auc, 'f1', combo.f1_score, 'count', combo.count);
                            all_combos(end+1) = new_combo;
                        end
                    end
                end
                
                % 如果有足够的组合，创建比较图
                if length(all_combos) >= 3
                    fig3 = figure('Name', 'Top Combinations Across Methods', 'Position', [100, 100, 1000, 600]);
                    
                    % 提取数据
                    methods_list = {all_combos.method};
                    aucs = [all_combos.auc];
                    f1s = [all_combos.f1];
                    counts = [all_combos.count];
                    vars_list = {all_combos.vars};
                    
                    % 创建气泡图（使用F1分数作为Y轴）
                    scatter(1:length(all_combos), f1s, counts*10, aucs*50, 'filled', 'MarkerFaceAlpha', 0.7);
                    colormap(jet);
                    colorbar;
                    
                    % 添加方法标签
                    text(1:length(all_combos), f1s, methods_list, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
                    
                    % 设置图形属性
                    set(gca, 'XTick', 1:length(all_combos));
                    set(gca, 'XTickLabel', vars_list, 'XTickLabelRotation', 45, 'FontSize', 9);
                    xlabel('变量组合', 'FontSize', 12);
                    ylabel('F1分数', 'FontSize', 12);
                    title('各方法中顶级变量组合的性能比较', 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加说明
                    text(length(all_combos)/2, min(f1s)-0.1, '注: 气泡大小表示组合出现频率，颜色表示AUC值', ...
                        'HorizontalAlignment', 'center', 'FontSize', 10);
                    
                    % 保存矢量图
                    utils.save_figure(fig3, figure_dir, 'top_combinations_comparison', 'Formats', {'svg'});
                    close(fig3);
                end
            catch ME
                logger.log_message('warning', sprintf('创建综合比较图时出错: %s', ME.message));
            end
        end
        
        function create_cv_performance_plot(cv_results, figure_dir)
            % 创建K折交叉验证性能图
            % 输入:
            %   cv_results - 交叉验证结果
            %   figure_dir - 图形保存目录
            
            % 创建图形
            fig = figure('Name', 'Cross-Validation Performance', 'Position', [100, 100, 1200, 800]);
            
            % 获取折数
            k = length(cv_results.accuracy);
            
            % 准备数据
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
            metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
            metric_colors = lines(length(metrics));
            
            % 创建子图1：各折性能
            subplot(2, 2, 1);
            hold on;
            
            % 为每个指标绘制折线
            for i = 1:length(metrics)
                metric = metrics{i};
                values = cv_results.(metric);
                
                % 绘制折线
                plot(1:k, values, 'o-', 'LineWidth', 1.5, 'Color', metric_colors(i,:), 'DisplayName', metric_labels{i});
            end
            
            % 设置图形属性
            xlabel('折数', 'FontSize', 12);
            ylabel('性能值', 'FontSize', 12);
            title('K折交叉验证中各折的性能表现', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            legend('Location', 'best');
            xlim([0.5, k+0.5]);
            ylim([0, 1.05]);
            set(gca, 'XTick', 1:k);
            
            % 创建子图2：各指标均值和标准差
            subplot(2, 2, 2);
            
            % 计算均值和标准差
            metric_means = zeros(length(metrics), 1);
            metric_stds = zeros(length(metrics), 1);
            
            for i = 1:length(metrics)
                metric = metrics{i};
                metric_means(i) = mean(cv_results.(metric), 'omitnan');
                metric_stds(i) = std(cv_results.(metric), 'omitnan');
            end
            
            % 创建条形图
            bar_h = bar(metric_means);
            set(bar_h, 'FaceColor', 'flat');
            for i = 1:length(metrics)
                bar_h.CData(i,:) = metric_colors(i,:);
            end
            
            % 添加误差线
            hold on;
            errorbar(1:length(metrics), metric_means, metric_stds, '.k');
            
            % 设置图形属性
            set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metric_labels);
            ylabel('平均性能', 'FontSize', 12);
            title('各评估指标的均值和标准差', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 1.05]);
            
            % 添加数值标签
            for i = 1:length(metrics)
                text(i, metric_means(i) + 0.03, sprintf('%.3f±%.3f', metric_means(i), metric_stds(i)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 9);
            end
            
            % 创建子图3：系数稳定性
            subplot(2, 2, 3);
            
            % 提取系数变异系数
            coef_cv = cv_results.coef_cv;
            var_list = ['Intercept'; cv_results.variables(2:end)]; % 排除截距
            
            % 创建条形图
            bar_h = barh(coef_cv);
            set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);
            
            % 设置图形属性
            set(gca, 'YTick', 1:length(coef_cv), 'YTickLabel', var_list);
            xlabel('变异系数 (CV)', 'FontSize', 12);
            title('模型系数稳定性 (变异系数)', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            % 创建子图4：学习曲线或ROC曲线
            subplot(2, 2, 4);
            
            try
                % 计算平均ROC曲线（如果可能）
                x_points = linspace(0, 1, 100);
                y_points = zeros(length(x_points), 1);
                
                % 绘制ROC曲线
                plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5); % 对角线
                hold on;
                
                for i = 1:k
                    if ~isnan(cv_results.auc(i)) && cv_results.auc(i) > 0.5
                        % 绘制简化的ROC曲线
                        x0 = 0;
                        y0 = 0;
                        x1 = 1 - cv_results.specificity(i);
                        y1 = cv_results.recall(i);
                        x2 = 1;
                        y2 = 1;
                        
                        % 绘制折线
                        plot([x0, x1, x2], [y0, y1, y2], '-', 'Color', [0.7, 0.7, 0.7, 0.3], 'LineWidth', 0.5);
                        
                        % 计算近似曲线下面积
                        for j = 1:length(x_points)
                            x = x_points(j);
                            if x <= x1
                                y = y1 * x / x1;
                            else
                                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
                            end
                            y_points(j) = y_points(j) + y / k;
                        end
                    end
                end
                
                % 绘制平均ROC曲线
                plot(x_points, y_points, '-', 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2], 'DisplayName', '平均ROC曲线');
                
                % 添加AUC值
                mean_auc = mean(cv_results.auc, 'omitnan');
                text(0.6, 0.2, sprintf('平均AUC = %.3f ± %.3f', mean_auc, std(cv_results.auc, 'omitnan')), ...
                    'FontSize', 10, 'FontWeight', 'bold');
                
                % 设置图形属性
                xlabel('1 - 特异性', 'FontSize', 12);
                ylabel('敏感性', 'FontSize', 12);
                title('平均ROC曲线', 'FontSize', 14, 'FontWeight', 'bold');
                grid on;
                legend('无信息线', '平均ROC曲线', 'Location', 'southeast');
                
            catch ME
                % 如果ROC曲线绘制失败，显示错误消息
                text(0.5, 0.5, '无法生成ROC曲线', 'HorizontalAlignment', 'center', 'FontSize', 12);
                logger.log_message('warning', sprintf('绘制ROC曲线失败: %s', ME.message));
            end
            
            % 调整整体布局
            sgtitle('K折交叉验证性能分析', 'FontSize', 16, 'FontWeight', 'bold');
            set(gcf, 'Color', 'white');
            
            % 保存矢量图
            utils.save_figure(fig, figure_dir, 'cv_performance', 'Formats', {'svg'});
            
            % 关闭图形
            close(fig);
        end
        
        function create_coefficient_stability_plot(coef_stability, methods, figure_dir)
            % 创建系数稳定性图
            % 输入:
            %   coef_stability - 系数稳定性结果
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
            
            % 对每种支持的方法创建图形
            for m = 1:length(methods)
                method = methods{m};
                
                % 检查该方法是否有系数稳定性结果
                if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
                    
                    % 提取数据
                    table_data = coef_stability.(method).table;
                    var_list = table_data.Variable;
                    coef_mean = table_data.Mean;
                    coef_std = table_data.StdDev;
                    coef_cv = table_data.CV;
                    
                    % 创建图形
                    fig = figure('Name', sprintf('%s Coefficient Stability', method), 'Position', [100, 100, 1200, 800]);
                    
                    % 创建子图1：系数均值和标准差
                    subplot(2, 1, 1);
                    
                    % 创建条形图
                    bar_h = bar(coef_mean);
                    hold on;
                    
                    % 添加误差线
                    errorbar(1:length(coef_mean), coef_mean, coef_std, '.k');
                    
                    % 设置图形属性
                    set(gca, 'XTick', 1:length(var_list), 'XTickLabel', var_list, 'XTickLabelRotation', 45);
                    ylabel('系数值', 'FontSize', 12);
                    title(sprintf('%s方法的系数均值和标准差', method), 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加零线
                    line([0, length(var_list)+1], [0, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
                    
                    % 创建子图2：系数变异系数
                    subplot(2, 1, 2);
                    
                    % 按变异系数大小排序
                    [sorted_cv, idx] = sort(coef_cv, 'descend');
                    sorted_vars = var_list(idx);
                    
                    % 创建条形图
                    bar_h = barh(sorted_cv);
                    set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);
                    
                    % 设置图形属性
                    set(gca, 'YTick', 1:length(sorted_vars), 'YTickLabel', sorted_vars);
                    xlabel('变异系数 (CV)', 'FontSize', 12);
                    title(sprintf('%s方法的系数稳定性 (变异系数)', method), 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加阈值线
                    line([0.5, 0.5], [0, length(sorted_vars)+1], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1);
                    text(0.52, length(sorted_vars)-1, '不稳定阈值 (CV > 0.5)', 'Color', 'r', 'FontSize', 10);
                    
                    % 调整整体布局
                    sgtitle(sprintf('%s方法的系数稳定性分析', method), 'FontSize', 16, 'FontWeight', 'bold');
                    set(gcf, 'Color', 'white');
                            
                    % 保存矢量图
                    utils.save_figure(fig, figure_dir, sprintf('%s_coefficient_stability', method), 'Formats', {'svg'});
                    
                    % 关闭图形
                    close(fig);
                end
            end
            
            % 创建所有方法的综合比较图
            try
                % 收集所有方法的变异系数
                all_methods = {};
                all_vars = {};
                all_cvs = [];
                
                for m = 1:length(methods)
                    method = methods{m};
                    if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
                        table_data = coef_stability.(method).table;
                        
                        % 只考虑截距项
                        for i = 1:height(table_data)
                            all_methods{end+1} = method;
                            all_vars{end+1} = table_data.Variable{i};
                            all_cvs(end+1) = table_data.CV(i);
                        end
                    end
                end
                
                % 如果有足够的数据，创建比较图
                if length(all_cvs) >= 3
                    fig = figure('Name', 'Coefficient Stability Comparison', 'Position', [100, 100, 1200, 600]);
                    
                    % 创建散点图
                    scatter(1:length(all_cvs), all_cvs, 50, 'filled', 'MarkerFaceAlpha', 0.7);
                    
                    % 添加方法和变量标签
                    for i = 1:length(all_cvs)
                        text(i, all_cvs(i) + 0.03, sprintf('%s\n%s', all_methods{i}, all_vars{i}), ...
                            'HorizontalAlignment', 'center', 'FontSize', 8, 'Rotation', 45);
                    end
                    
                    % 设置图形属性
                    set(gca, 'XTick', []);
                    ylabel('变异系数 (CV)', 'FontSize', 12);
                    title('各方法系数稳定性比较', 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加阈值线
                    line([0, length(all_cvs)+1], [0.5, 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1);
                    text(length(all_cvs)*0.9, 0.52, '不稳定阈值 (CV > 0.5)', 'Color', 'r', 'FontSize', 10);
                    
                    % 调整图形
                    set(gcf, 'Color', 'white');
                            
                    % 保存矢量图
                    utils.save_figure(fig, figure_dir, 'coefficient_stability_comparison', 'Formats', {'svg'});
                    
                    % 关闭图形
                    close(fig);
                end
            catch ME
                logger.log_message('warning', sprintf('创建系数稳定性比较图失败: %s', ME.message));
            end
        end
        
        function create_variable_contribution_plot(var_contribution, figure_dir)
            % 创建变量贡献图
            % 输入:
            %   var_contribution - 变量贡献分析结果
            %   figure_dir - 图形保存目录
            
            % 创建综合变量重要性图
            if isfield(var_contribution, 'overall_importance')
                try
                    % 提取数据
                    importance_table = var_contribution.overall_importance;
                    vars = importance_table.Variable;
                    importance = importance_table.Normalized_Importance;
                    
                    % 取前10个变量
                    top_n = min(10, length(vars));
                    top_vars = vars(1:top_n);
                    top_importance = importance(1:top_n);
                    
                    % 创建图形
                    fig = figure('Name', 'Overall Variable Importance', 'Position', [100, 100, 1000, 600]);
                    
                    % 创建条形图
                    barh_h = barh(top_importance);
                    set(barh_h, 'FaceColor', [0.3, 0.6, 0.8]);
                    
                    % 设置图形属性
                    set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
                    xlabel('归一化重要性 (%)', 'FontSize', 12);
                    title('综合变量重要性排名 (前10个变量)', 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加数值标签
                    for i = 1:top_n
                        text(top_importance(i) + 0.5, i, sprintf('%.2f%%', top_importance(i)), ...
                            'VerticalAlignment', 'middle', 'FontSize', 9);
                    end
                    
                    % 调整图形
                    set(gcf, 'Color', 'white');
                    
                    % 保存矢量图
                    utils.save_figure(fig, figure_dir, 'overall_variable_importance', 'Formats', {'svg'});
                    
                    % 关闭图形
                    close(fig);
                catch ME
                    logger.log_message('warning', sprintf('创建综合变量重要性图失败: %s', ME.message));
                end
            end
            
            % 创建相关性分析图
            if isfield(var_contribution, 'correlation')
                try
                    % 提取数据
                    corr_table = var_contribution.correlation;
                    vars = corr_table.Variable;
                    corr_values = corr_table.Correlation;
                    partial_corr = corr_table.PartialCorr;
                    
                    % 按偏相关系数绝对值排序
                    [~, idx] = sort(abs(partial_corr), 'descend');
                    sorted_vars = vars(idx);
                    sorted_corr = corr_values(idx);
                    sorted_partial = partial_corr(idx);
                    
                    % 取前10个变量
                    top_n = min(10, length(sorted_vars));
                    top_vars = sorted_vars(1:top_n);
                    top_corr = sorted_corr(1:top_n);
                    top_partial = sorted_partial(1:top_n);
                    
                    % 创建图形
                    fig = figure('Name', 'Correlation Analysis', 'Position', [100, 100, 1000, 600]);
                    
                    % 创建分组条形图
                    bar_data = [top_corr, top_partial];
                    bar_h = barh(bar_data);
                    set(bar_h(1), 'FaceColor', [0.3, 0.6, 0.8]);
                    set(bar_h(2), 'FaceColor', [0.8, 0.3, 0.3]);
                    
                    % 设置图形属性
                    set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
                    xlabel('相关系数', 'FontSize', 12);
                    title('变量相关性分析 (前10个变量)', 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加图例
                    legend({'普通相关', '偏相关'}, 'Location', 'southeast');
                    
                    % 添加参考线
                    line([0, 0], [0, top_n+1], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
                    
                    % 调整图形
                    set(gcf, 'Color', 'white');
                            
                    % 保存矢量图
                    utils.save_figure(fig, figure_dir, 'correlation_analysis', 'Formats', {'svg'});
            
                    % 关闭图形
                    close(fig);
                catch ME
                    logger.log_message('warning', sprintf('创建相关性分析图失败: %s', ME.message));
                end
            end
            
            % 为各个方法创建变量贡献图
            if isfield(var_contribution, 'methods')
                method_names = fieldnames(var_contribution.methods);
                
                for m = 1:length(method_names)
                    method = method_names{m};
                    
                    % 检查该方法是否有贡献表
                    if isfield(var_contribution.methods.(method), 'contribution_table')
                        try
                            % 提取数据
                            contrib_table = var_contribution.methods.(method).contribution_table;
                            
                            % 创建图形
                            fig = figure('Name', sprintf('%s Variable Contribution', method), 'Position', [100, 100, 1000, 600]);
                            
                            % 提取变量和贡献
                            vars = contrib_table.Variable;
                            rel_contrib = contrib_table.Relative_Contribution;
                            
                            % 取前10个变量
                            top_n = min(10, height(contrib_table));
                            top_vars = vars(1:top_n);
                            top_contrib = rel_contrib(1:top_n);
                            
                            % 为不同方法设置不同的表现形式
                            if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                                % 回归类方法显示系数和相对贡献
                                
                                % 提取系数和方向
                                coeffs = contrib_table.Std_Coefficient(1:top_n);
                                directions = contrib_table.Effect_Direction(1:top_n);
                                
                                % 创建颜色映射
                                colors = zeros(top_n, 3);
                                for i = 1:top_n
                                    if strcmp(directions{i}, '正向')
                                        colors(i,:) = [0.2, 0.6, 0.8]; % 蓝色表示正向影响
                                    else
                                        colors(i,:) = [0.8, 0.3, 0.3]; % 红色表示负向影响
                                    end
                                end
                                
                                % 创建水平条形图
                                bar_h = barh(top_contrib);
                                set(bar_h, 'FaceColor', 'flat');
                                bar_h.CData = colors;
                                
                                % 设置图形属性
                                set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
                                xlabel('相对贡献 (%)', 'FontSize', 12);
                                title(sprintf('%s方法的变量贡献分析 (前%d个变量)', method, top_n), 'FontSize', 14, 'FontWeight', 'bold');
                                grid on;
                                
                                % 添加方向指示
                                for i = 1:top_n
                                    if strcmp(directions{i}, '正向')
                                        text(top_contrib(i) + 0.5, i, '(+)', 'VerticalAlignment', 'middle', 'FontSize', 9, 'Color', [0, 0.5, 0]);
                                    else
                                        text(top_contrib(i) + 0.5, i, '(-)', 'VerticalAlignment', 'middle', 'FontSize', 9, 'Color', [0.8, 0, 0]);
                                    end
                                end
                                
                            else
                                % 非回归类方法只显示重要性
                                bar_h = barh(top_contrib);
                                set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);
                                
                                % 设置图形属性
                                set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
                                xlabel('相对贡献 (%)', 'FontSize', 12);
                                title(sprintf('%s方法的变量重要性 (前%d个变量)', method, top_n), 'FontSize', 14, 'FontWeight', 'bold');
                                grid on;
                            end
                            
                            % 添加数值标签
                            for i = 1:top_n
                                text(top_contrib(i) + 0.5, i, sprintf('%.2f%%', top_contrib(i)), ...
                                    'VerticalAlignment', 'middle', 'FontSize', 9);
                            end
                            
                            % 调整图形
                            set(gcf, 'Color', 'white');
                                            
                            % 保存矢量图
                            utils.save_figure(fig, figure_dir, [method, '_variable_contribution'], 'Formats', {'svg'});
                            
                            % 关闭图形
                            close(fig);
                        catch ME
                            logger.log_message('warning', sprintf('创建%s方法的变量贡献图失败: %s', method, ME.message));
                        end
                    end
                end
            end
        end
        
        function create_boxplot_visualization(results, methods, figure_dir)
            % 创建箱线图可视化
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
        
            % 定义性能指标名称
            metric_names = {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC'};
            n_metrics = length(metric_names);
            n_methods = length(methods);
        
            % 初始化数据矩阵
            data = cell(n_methods, n_metrics);
            for i = 1:n_methods
                method = methods{i};
                if isfield(results, method) && isfield(results.(method), 'performance')
                    perf = results.(method).performance;
                    
                    % 检查并填充每个指标的数据
                    data{i, 1} = reporting_module.check_data(perf, 'accuracy', method, 'Accuracy');
                    data{i, 2} = reporting_module.check_data(perf, 'sensitivity', method, 'Sensitivity');
                    data{i, 3} = reporting_module.check_data(perf, 'specificity', method, 'Specificity');
                    data{i, 4} = reporting_module.check_data(perf, 'precision', method, 'Precision');
                    data{i, 5} = reporting_module.check_data(perf, 'f1_score', method, 'F1_Score');
                    data{i, 6} = reporting_module.check_data(perf, 'auc', method, 'AUC');
                else
                    % 如果方法或性能数据缺失，填充 NaN
                    logger.log_message('warning', sprintf('Method %s is missing in results or performance data', method));
                    for j = 1:n_metrics
                        data{i, j} = NaN;
                    end
                end
            end
        
            % 为每个指标创建箱线图
            for i = 1:n_metrics
                metric = metric_names{i};
                
                % 提取当前指标的数据
                metric_data_cell = data(:, i);
                
                % 确定最大数据长度并填充 NaN
                max_len = 0;
                for j = 1:n_methods
                    if ~isempty(metric_data_cell{j}) && ~all(isnan(metric_data_cell{j}))
                        max_len = max(max_len, length(metric_data_cell{j}));
                    end
                end
                
                % 如果没有有效数据，跳过
                if max_len == 0
                    logger.log_message('warning', sprintf('No valid data available for metric %s', metric));
                    continue;
                end
                
                % 创建矩阵，填充 NaN 以对齐维度
                metric_data = NaN(max_len, n_methods);
                for j = 1:n_methods
                    current_data = metric_data_cell{j};
                    if ~isempty(current_data) && ~all(isnan(current_data))
                        len = length(current_data);
                        metric_data(1:len, j) = current_data;
                    end
                end
                
                % 验证列数与方法数一致
                if size(metric_data, 2) ~= n_methods
                    logger.log_message('error', sprintf('metric_data has %d columns, but there are %d methods for %s', ...
                        size(metric_data, 2), n_methods, metric));
                    continue;
                end
                
                % 创建图形
                fig = figure('Name', sprintf('%s Boxplot', metric), 'Position', [100, 100, 1000, 600]);
                
                % 创建箱线图
                boxplot(metric_data, 'Labels', methods, 'Notch', 'on', 'Symbol', 'r+');
                
                % 设置图形属性
                title(sprintf('%s分布箱线图 - 各方法比较', metric), 'FontSize', 14, 'FontWeight', 'bold');
                ylabel(metric, 'FontSize', 12, 'FontWeight', 'bold');
                grid on;
                
                % 添加均值点
                hold on;
                means = nanmean(metric_data);
                scatter(1:n_methods, means, 100, 'filled', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
                
                % 添加均值和标准差标签
                for j = 1:n_methods
                    method_data = metric_data(:, j);
                    mean_val = nanmean(method_data);
                    std_val = nanstd(method_data);
                    text(j, max(method_data, [], 'omitnan') + 0.02, sprintf('均值: %.3f\n标准差: %.3f', mean_val, std_val), ...
                        'HorizontalAlignment', 'center', 'FontSize', 9);
                end
                
                % 添加整体均值线
                overall_mean = nanmean(metric_data(:));
                plot([0.5, n_methods+0.5], [overall_mean, overall_mean], 'k--', 'LineWidth', 1.5);
                text(n_methods+0.5, overall_mean, sprintf(' 总体均值: %.3f', overall_mean), ...
                    'VerticalAlignment', 'middle', 'FontSize', 9);
                
                % 调整 Y 轴范围
                ylim_current = ylim;
                ylim([ylim_current(1), ylim_current(2) + 0.1]);
                
                % 保存图形
                utils.save_figure(fig, figure_dir, sprintf('boxplot_%s', lower(metric)), 'Formats', {'svg'});
                close(fig);
            end
        end
        
        function data_out = check_data(perf, field, method, metric_name)
            % 检查和处理数据
            % 输入:
            %   perf - 性能结构
            %   field - 字段名
            %   method - 方法名
            %   metric_name - 指标名
            % 输出:
            %   data_out - 处理后的数据
            
            if isfield(perf, field) && ~isempty(perf.(field))
                data_out = perf.(field);
            else
                logger.log_message('warning', sprintf('Method %s has missing or empty %s data', method, metric_name));
                data_out = NaN;
            end
        end
        
        function create_pr_curves(results, methods, figure_dir)
            % 创建精确率-召回率曲线图
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
            
            fig = figure('Name', 'Precision-Recall Curves', 'Position', [100, 100, 1000, 800]);
            
            colors = lines(length(methods));
            markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
            hold on;
            
            legend_entries = cell(length(methods), 1);
            
            % 创建标准PR曲线点
            std_recall = linspace(0, 1, 100)';
            avg_precision = zeros(100, length(methods));
            
            for i = 1:length(methods)
                method = methods{i};
                
                % 计算平均精确率-召回率曲线
                all_recall = [];
                all_precision = [];
                
                % 如果存在性能结构中的预测概率，使用它们绘制PR曲线
                if isfield(results.(method).performance, 'y_pred_prob') && ...
                   isfield(results.(method).performance, 'y_test')
                    
                    y_pred_prob = results.(method).performance.y_pred_prob;
                    y_test = results.(method).performance.y_test;
                    
                    % 对每个Bootstrap样本计算PR曲线
                    for j = 1:length(y_pred_prob)
                        if ~isempty(y_pred_prob{j}) && ~isempty(y_test{j})
                            [precision, recall, ~] = reporting_module.precision_recall_curve(y_test{j}, y_pred_prob{j});
                            
                            % 存储所有召回率和精确率值
                            all_recall = [all_recall; recall];
                            all_precision = [all_precision; precision];
                        end
                    end
                    
                    % 如果收集到足够的点，绘制平均PR曲线
                    if ~isempty(all_recall) && ~isempty(all_precision)
                        % 对所有样本的召回率进行排序
                        [sorted_recall, idx] = sort(all_recall);
                        sorted_precision = all_precision(idx);
                        
                        % 在标准召回率点计算平均精确率
                        for k = 1:length(std_recall)
                            r = std_recall(k);
                            % 找出大于或等于r的最接近点
                            idx = find(sorted_recall >= r, 1, 'first');
                            if isempty(idx)
                                avg_precision(k, i) = 0;
                            else
                                avg_precision(k, i) = sorted_precision(idx);
                            end
                        end
                        
                        % 计算平均精确率 (AP)
                        ap = trapz(std_recall, avg_precision(:, i));
                        
                        % 绘制PR曲线
                        color_idx = mod(i-1, size(colors, 1)) + 1;
                        plot(std_recall, avg_precision(:, i), '-', 'Color', colors(color_idx,:), 'LineWidth', 2);
                        
                        % 在曲线上标记数据点
                        marker_idx = mod(i-1, length(markers)) + 1;
                        num_points = 10; % 均匀分布的点数
                        point_idx = round(linspace(1, length(std_recall), num_points));
                        plot(std_recall(point_idx), avg_precision(point_idx, i), markers{marker_idx}, ...
                            'Color', colors(color_idx,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(color_idx,:));
                        
                        legend_entries{i} = sprintf('%s (AP=%.3f)', method, ap);
                    else
                        % 如果没有足够的数据点，使用性能指标中的平均值绘制单个点
                        recall = results.(method).performance.avg_sensitivity;
                        precision = results.(method).performance.avg_precision;
                        
                        color_idx = mod(i-1, size(colors, 1)) + 1;
                        marker_idx = mod(i-1, length(markers)) + 1;
                        
                        plot(recall, precision, markers{marker_idx}, 'Color', colors(color_idx,:), ...
                            'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
                        
                        legend_entries{i} = sprintf('%s (单点)', method);
                    end
                else
                    % 使用单点性能指标
                    recall = results.(method).performance.avg_sensitivity;
                    precision = results.(method).performance.avg_precision;
                    
                    color_idx = mod(i-1, size(colors, 1)) + 1;
                    marker_idx = mod(i-1, length(markers)) + 1;
                    
                    plot(recall, precision, markers{marker_idx}, 'Color', colors(color_idx,:), ...
                        'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
                    
                    legend_entries{i} = sprintf('%s (单点)', method);
                end
            end
            
            % 添加随机分类器的基准线
            random_precision = sum([results.(methods{1}).performance.avg_sensitivity]) / length(methods);
            plot([0, 1], [random_precision, random_precision], 'k--', 'LineWidth', 1.5);
            legend_entries{end+1} = sprintf('随机 (Precision=%.3f)', random_precision);
            
            % 设置图形属性
            xlim([0, 1]);
            ylim([0, 1]);
            xlabel('召回率 (Recall)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('精确率 (Precision)', 'FontSize', 12, 'FontWeight', 'bold');
            title('不同变量选择方法的精确率-召回率曲线比较', 'FontSize', 14, 'FontWeight', 'bold');
            legend(legend_entries, 'Location', 'southwest', 'FontSize', 10);
            grid on;
            set(gca, 'FontSize', 11);
            box on;
            
            % 保存图形
            utils.save_figure(fig, figure_dir, 'precision_recall_curves', 'Formats', {'svg'});
            close(fig);
        end
        
        function [precision, recall, thresholds] = precision_recall_curve(y_true, y_score)
            % 计算精确率-召回率曲线
            % 输入:
            %   y_true - 真实标签
            %   y_score - 预测分数或概率
            % 输出:
            %   precision - 精确率
            %   recall - 召回率
            %   thresholds - 阈值
            
            % 获取唯一的阈值，以降序排列
            thresholds = sort(unique(y_score), 'descend');
            
            % 添加0作为最后一个阈值
            thresholds = [thresholds; -Inf];
            
            n_thresholds = length(thresholds);
            precision = zeros(n_thresholds, 1);
            recall = zeros(n_thresholds, 1);
            
            for i = 1:n_thresholds
                threshold = thresholds(i);
                
                % 在当前阈值下的预测
                y_pred = y_score >= threshold;
                
                % 计算混淆矩阵元素
                TP = sum(y_pred == 1 & y_true == 1);
                FP = sum(y_pred == 1 & y_true == 0);
                FN = sum(y_pred == 0 & y_true == 1);
                
                % 计算精确率和召回率
                if TP + FP == 0
                    precision(i) = 1;  % 如果没有阳性预测，精确率为1
                else
                    precision(i) = TP / (TP + FP);
                end
                
                if TP + FN == 0
                    recall(i) = 0;  % 如果没有真阳性，召回率为0
                else
                    recall(i) = TP / (TP + FN);
                end
            end
            
            % 确保精确率-召回率曲线是单调递减的
            for i = n_thresholds-1:-1:1
                precision(i) = max(precision(i), precision(i+1));
            end
        end
        
        function create_calibration_curves(results, methods, figure_dir)
            % 创建校准曲线图
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
            
            fig = figure('Name', 'Calibration Curves', 'Position', [100, 100, 1000, 800]);
            
            colors = lines(length(methods));
            markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
            hold on;
            
            legend_entries = cell(length(methods), 1);
            
            % 绘制理想校准曲线
            plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);
            
            n_bins = 10;  % 分箱数量
            bin_edges = linspace(0, 1, n_bins+1);
            bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
            
            for i = 1:length(methods)
                method = methods{i};
                
                % 检查是否存在预测概率
                if isfield(results.(method).performance, 'y_pred_prob') && ...
                   isfield(results.(method).performance, 'y_test')
                    
                    y_pred_prob = results.(method).performance.y_pred_prob;
                    y_test = results.(method).performance.y_test;
                    
                    % 合并所有Bootstrap样本的数据
                    all_probs = [];
                    all_labels = [];
                    
                    for j = 1:length(y_pred_prob)
                        if ~isempty(y_pred_prob{j}) && ~isempty(y_test{j})
                            all_probs = [all_probs; y_pred_prob{j}];
                            all_labels = [all_labels; y_test{j}];
                        end
                    end
                    
                    if ~isempty(all_probs) && ~isempty(all_labels)
                        % 计算校准曲线
                        [fraction_of_positives, mean_predicted_value] = reporting_module.calibration_curve(all_labels, all_probs, n_bins);
                        
                        % 绘制校准曲线
                        color_idx = mod(i-1, size(colors, 1)) + 1;
                        marker_idx = mod(i-1, length(markers)) + 1;
                        
                        plot(mean_predicted_value, fraction_of_positives, ['-' markers{marker_idx}], ...
                            'Color', colors(color_idx,:), 'LineWidth', 2, 'MarkerSize', 8, ...
                            'MarkerFaceColor', colors(color_idx,:));
                        
                        % 计算Brier分数（校准误差）
                        brier_score = mean((all_probs - all_labels).^2);
                        
                        legend_entries{i} = sprintf('%s (Brier=%.3f)', method, brier_score);
                    else
                        legend_entries{i} = method;
                    end
                else
                    % 使用单点性能指标
                    y_pred = results.(method).performance.avg_sensitivity;
                    y_true = results.(method).performance.avg_precision;
                    
                    color_idx = mod(i-1, size(colors, 1)) + 1;
                    marker_idx = mod(i-1, length(markers)) + 1;
                    
                    plot(y_pred, y_true, markers{marker_idx}, 'Color', colors(color_idx,:), ...
                        'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
                    
                    legend_entries{i} = sprintf('%s (单点)', method);
                end
            end
            
            % 设置图形属性
            xlim([0, 1]);
            ylim([0, 1]);
            xlabel('预测概率', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('实际阳性比例', 'FontSize', 12, 'FontWeight', 'bold');
            title('不同变量选择方法的校准曲线比较', 'FontSize', 14, 'FontWeight', 'bold');
            legend([legend_entries; {'完美校准'}], 'Location', 'southeast', 'FontSize', 10);
            grid on;
            set(gca, 'FontSize', 11);
            box on;
            
            % 保存图形
            utils.save_figure(fig, figure_dir, 'calibration_curves', 'Formats', {'svg'});
            close(fig);
        end
        
        function [fraction_of_positives, mean_predicted_value] = calibration_curve(y_true, y_prob, n_bins)
            % 计算校准曲线
            % 输入:
            %   y_true - 真实标签
            %   y_prob - 预测概率
            %   n_bins - 分箱数量
            % 输出:
            %   fraction_of_positives - 每个分箱中的实际阳性比例
            %   mean_predicted_value - 每个分箱的平均预测概率
            
            % 计算分箱边界
            bin_edges = linspace(0, 1, n_bins+1);
            
            % 初始化结果数组
            fraction_of_positives = zeros(n_bins, 1);
            mean_predicted_value = zeros(n_bins, 1);
            
            % 对每个分箱计算
            for i = 1:n_bins
                % 找出落入当前分箱的样本
                bin_mask = (y_prob >= bin_edges(i)) & (y_prob < bin_edges(i+1));
                
                % 如果分箱为空，使用默认值
                if sum(bin_mask) == 0
                    fraction_of_positives(i) = 0;
                    mean_predicted_value(i) = (bin_edges(i) + bin_edges(i+1)) / 2;
                else
                    % 计算实际阳性比例
                    fraction_of_positives(i) = mean(y_true(bin_mask));
                    
                    % 计算平均预测概率
                    mean_predicted_value(i) = mean(y_prob(bin_mask));
                end
            end
        end
        
        function create_confusion_matrices(results, methods, figure_dir)
            % 创建混淆矩阵可视化
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
        
            % 为每种方法创建混淆矩阵
            for i = 1:length(methods)
                method = methods{i};
                
                % 获取性能指标
                performance = results.(method).performance;
                
                % 计算平均混淆矩阵
                if isfield(performance, 'y_pred') && isfield(performance, 'y_test')
                    y_pred = performance.y_pred;
                    y_test = performance.y_test;
                    
                    % 初始化混淆矩阵
                    conf_matrix = zeros(2, 2);
                    count = 0;
                    
                    % 合并所有Bootstrap样本的混淆矩阵
                    for j = 1:length(y_pred)
                        if ~isempty(y_pred{j}) && ~isempty(y_test{j})
                            % 计算当前样本的混淆矩阵
                            pred = y_pred{j};
                            test = y_test{j};
                            
                            TP = sum(pred == 1 & test == 1);
                            FP = sum(pred == 1 & test == 0);
                            FN = sum(pred == 0 & test == 1);
                            TN = sum(pred == 0 & test == 0);
                            
                            conf_matrix = conf_matrix + [TN, FP; FN, TP];
                            count = count + 1;
                        end
                    end
                    
                    if count > 0
                        conf_matrix = conf_matrix / count;
                    end
                else
                    % 使用性能指标估算混淆矩阵
                    sensitivity = performance.avg_sensitivity;
                    specificity = performance.avg_specificity;
                    precision = performance.avg_precision;
                    
                    % 假设测试集中正负样本比例为1:1
                    TP = sensitivity * 50;
                    FN = 50 - TP;
                    FP = TP / precision - TP;
                    TN = specificity * 50;
                    
                    conf_matrix = [TN, FP; FN, TP];
                end
                
                % 计算归一化混淆矩阵（按行归一化）
                conf_matrix_norm = zeros(2, 2);
                for j = 1:2
                    if sum(conf_matrix(j, :)) > 0
                        conf_matrix_norm(j, :) = conf_matrix(j, :) / sum(conf_matrix(j, :));
                    end
                end
                
                % 创建混淆矩阵图
                fig = figure('Name', sprintf('%s Confusion Matrix', method), 'Position', [100, 100, 800, 600]);
                
                % 绘制混淆矩阵热图
                subplot(1, 2, 1);
                h = heatmap(conf_matrix, 'XLabel', '预测', 'YLabel', '实际', ...
                    'XDisplayLabels', {'负类 (0)', '正类 (1)'}, 'YDisplayLabels', {'负类 (0)', '正类 (1)'});
                h.Title = sprintf('%s: 原始混淆矩阵', method);
                h.FontSize = 12;
                colormap(jet);
                
                % 绘制归一化混淆矩阵热图
                subplot(1, 2, 2);
                h_norm = heatmap(conf_matrix_norm, 'XLabel', '预测', 'YLabel', '实际', ...
                    'XDisplayLabels', {'负类 (0)', '正类 (1)'}, 'YDisplayLabels', {'负类 (0)', '正类 (1)'});
                h_norm.Title = sprintf('%s: 归一化混淆矩阵', method);
                h_norm.FontSize = 12;
                colormap(jet);
                
                % 计算性能指标
                TN = conf_matrix(1, 1);
                FP = conf_matrix(1, 2);
                FN = conf_matrix(2, 1);
                TP = conf_matrix(2, 2);
                
                accuracy = (TP + TN) / sum(conf_matrix(:));
                sensitivity = TP / (TP + FN);
                specificity = TN / (TN + FP);
                precision = TP / (TP + FP);
                f1_score = 2 * precision * sensitivity / (precision + sensitivity);
                
                % 创建标题包含性能指标
                sgtitle(sprintf('%s 混淆矩阵分析\n准确率=%.3f, 灵敏度=%.3f, 特异性=%.3f, 精确率=%.3f, F1=%.3f', ...
                    method, accuracy, sensitivity, specificity, precision, f1_score), ...
                    'FontSize', 14, 'FontWeight', 'bold');
                
                % 保存图形
                utils.save_figure(fig, figure_dir, sprintf('%s_confusion_matrix', method), 'Formats', {'svg'});
                close(fig);
            end
        
            % 创建所有方法的混淆矩阵比较图
            try
                % 计算每种方法的归一化混淆矩阵
                all_conf_matrices = cell(length(methods), 1);
                all_performance = zeros(length(methods), 4); % 准确率、灵敏度、特异性、精确率
                
                for i = 1:length(methods)
                    method = methods{i};
                    performance = results.(method).performance;
                    
                    if isfield(performance, 'y_pred') && isfield(performance, 'y_test')
                        y_pred = performance.y_pred;
                        y_test = performance.y_test;
                        
                        % 初始化混淆矩阵
                        conf_matrix = zeros(2, 2);
                        count = 0;
                        
                        % 合并所有Bootstrap样本的混淆矩阵
                        for j = 1:length(y_pred)
                            if ~isempty(y_pred{j}) && ~isempty(y_test{j})
                                % 计算当前样本的混淆矩阵
                                pred = y_pred{j};
                                test = y_test{j};
                                
                                TP = sum(pred == 1 & test == 1);
                                FP = sum(pred == 1 & test == 0);
                                FN = sum(pred == 0 & test == 1);
                                TN = sum(pred == 0 & test == 0);
                                
                                conf_matrix = conf_matrix + [TN, FP; FN, TP];
                                count = count + 1;
                            end
                        end
                        
                        % 计算平均混淆矩阵
                        if count > 0
                            conf_matrix = conf_matrix / count;
                        end
                    else
                        % 使用性能指标估算混淆矩阵
                        sensitivity = performance.avg_sensitivity;
                        specificity = performance.avg_specificity;
                        precision = performance.avg_precision;
                        
                        % 假设测试集中正负样本比例为1:1
                        TP = sensitivity * 50;
                        FN = 50 - TP;
                        FP = TP / precision - TP;
                        TN = specificity * 50;
                        
                        conf_matrix = [TN, FP; FN, TP];
                    end
                    
                    % 计算归一化混淆矩阵（按行归一化）
                    conf_matrix_norm = zeros(2, 2);
                    for j = 1:2
                        if sum(conf_matrix(j, :)) > 0
                            conf_matrix_norm(j, :) = conf_matrix(j, :) / sum(conf_matrix(j, :));
                        end
                    end
                    
                    all_conf_matrices{i} = conf_matrix_norm;
                    
                    % 计算性能指标
                    TN = conf_matrix(1, 1);
                    FP = conf_matrix(1, 2);
                    FN = conf_matrix(2, 1);
                    TP = conf_matrix(2, 2);
                    
                    accuracy = (TP + TN) / sum(conf_matrix(:));
                    sensitivity = TP / (TP + FN);
                    specificity = TN / (TN + FP);
                    precision = TP / (TP + FP);
                    
                    all_performance(i, :) = [accuracy, sensitivity, specificity, precision];
                end
                
                % 创建比较图
                fig = figure('Name', 'Confusion Matrix Comparison', 'Position', [100, 100, 1200, 900]);
                
                n_methods = length(methods);
                rows = ceil(sqrt(n_methods));
                cols = ceil(n_methods / rows);
                
                for i = 1:n_methods
                    subplot(rows, cols, i);
                    h = heatmap(all_conf_matrices{i}, 'XLabel', '预测', 'YLabel', '实际', ...
                        'XDisplayLabels', {'0', '1'}, 'YDisplayLabels', {'0', '1'});
                    h.Title = sprintf('%s\n准确率=%.3f', methods{i}, all_performance(i, 1));
                    h.FontSize = 9;
                    colormap(jet);
                end
                
                % 添加整体标题
                sgtitle('各方法混淆矩阵比较 (归一化)', 'FontSize', 14, 'FontWeight', 'bold');
        
                % 保存图形
                save_figure(fig, figure_dir, 'confusion_matrix_comparison', 'Formats', {'svg'});
                close(fig);
            catch ME
                log_message('warning', sprintf('创建混淆矩阵比较图失败: %s', ME.message));
            end
        end

        