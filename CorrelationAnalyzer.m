classdef CorrelationAnalyzer < handle
    % 相关性分析器类：执行变量间的相关性分析
    % 包括主成分分析、偏相关分析等
    
    properties (Access = private)
        Config
        Logger
        Results
    end
    
    methods (Access = public)
        function obj = CorrelationAnalyzer(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.Results = struct();
        end
        
        function pca_results = Analyze(obj, X, var_names)
            % 执行相关性分析
            % 输入:
            %   X - 自变量矩阵
            %   var_names - 变量名称
            % 输出:
            %   pca_results - PCA分析结果
            
            obj.Logger.Log('info', '开始相关性分析');
            
            try
                % 1. 计算基本相关性
                obj.CalculateBasicCorrelations(X, var_names);
                
                % 2. 执行主成分分析
                pca_results = obj.PerformPCA(X, var_names);
                
                % 3. 计算偏相关系数
                obj.CalculatePartialCorrelations(X, var_names);
                
                % 4. 因子分析
                obj.PerformFactorAnalysis(X, var_names);
                
                % 5. 聚类分析
                obj.PerformVariableClustering(X, var_names);
                
                % 6. 创建可视化
                obj.CreateVisualizations();
                
                % 7. 生成分析报告
                obj.GenerateAnalysisReport();
                
                obj.Logger.Log('info', '相关性分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'CorrelationAnalyzer.Analyze');
                rethrow(ME);
            end
        end
        
        function results = GetResults(obj)
            % 获取分析结果
            results = obj.Results;
        end
    end
    
    methods (Access = private)
        function CalculateBasicCorrelations(obj, X, var_names)
            % 计算基本相关性
            try
                obj.Logger.Log('debug', '计算基本相关性');
                
                % 计算Pearson相关系数
                [R_pearson, P_pearson] = corr(X, 'Type', 'Pearson');
                
                % 计算Spearman相关系数
                [R_spearman, P_spearman] = corr(X, 'Type', 'Spearman');
                
                % 计算Kendall相关系数
                try
                    [R_kendall, P_kendall] = corr(X, 'Type', 'Kendall');
                catch
                    obj.Logger.Log('warning', 'Kendall相关系数计算失败，跳过');
                    R_kendall = [];
                    P_kendall = [];
                end
                
                % 保存结果
                obj.Results.correlations = struct();
                obj.Results.correlations.pearson = R_pearson;
                obj.Results.correlations.pearson_pvalues = P_pearson;
                obj.Results.correlations.spearman = R_spearman;
                obj.Results.correlations.spearman_pvalues = P_spearman;
                obj.Results.correlations.kendall = R_kendall;
                obj.Results.correlations.kendall_pvalues = P_kendall;
                
                % 创建相关性汇总
                obj.CreateCorrelationSummary(var_names);
                
            catch ME
                obj.Logger.LogException(ME, 'CalculateBasicCorrelations');
            end
        end
        
        function CreateCorrelationSummary(obj, var_names)
            % 创建相关性汇总
            try
                R = obj.Results.correlations.pearson;
                n_vars = size(R, 1);
                
                % 创建上三角矩阵索引
                upper_tri = triu(true(n_vars), 1);
                
                % 提取相关系数
                correlations = R(upper_tri);
                
                % 计算统计信息
                summary = struct();
                summary.mean_correlation = mean(abs(correlations));
                summary.median_correlation = median(abs(correlations));
                summary.max_correlation = max(abs(correlations));
                summary.min_correlation = min(abs(correlations));
                summary.std_correlation = std(abs(correlations));
                
                % 找出高相关变量对
                threshold = 0.7;
                high_corr_pairs = [];
                
                for i = 1:n_vars-1
                    for j = i+1:n_vars
                        if abs(R(i, j)) > threshold
                            high_corr_pairs = [high_corr_pairs; ...
                                {var_names{i}, var_names{j}, R(i, j), abs(R(i, j))}];
                        end
                    end
                end
                
                % 保存高相关对
                if ~isempty(high_corr_pairs)
                    summary.high_correlation_pairs = cell2table(high_corr_pairs, ...
                        'VariableNames', {'Variable1', 'Variable2', 'Correlation', 'AbsCorrelation'});
                    summary.high_correlation_pairs = sortrows(summary.high_correlation_pairs, ...
                        'AbsCorrelation', 'descend');
                else
                    summary.high_correlation_pairs = table();
                end
                
                obj.Results.correlation_summary = summary;
                
                % 记录统计信息
                obj.Logger.Log('info', sprintf('相关性统计：平均=%.3f, 最大=%.3f, 标准差=%.3f', ...
                    summary.mean_correlation, summary.max_correlation, summary.std_correlation));
                
                if height(summary.high_correlation_pairs) > 0
                    obj.Logger.Log('info', sprintf('发现 %d 对高相关变量 (|r| > %.2f)', ...
                        height(summary.high_correlation_pairs), threshold));
                end
                
            catch ME
                obj.Logger.LogException(ME, 'CreateCorrelationSummary');
            end
        end
        
        function pca_results = PerformPCA(obj, X, var_names)
            % 执行主成分分析
            try
                obj.Logger.Log('debug', '开始主成分分析');
                
                % 执行PCA
                [coeff, score, latent, tsquared, explained, mu] = pca(X, ...
                    'Algorithm', 'svd', 'VariableWeights', 'variance');
                
                % 保存PCA结果
                pca_results = struct();
                pca_results.coefficients = coeff;
                pca_results.scores = score;
                pca_results.latent = latent;
                pca_results.tsquared = tsquared;
                pca_results.explained = explained;
                pca_results.mu = mu;
                pca_results.cumulative_variance = cumsum(explained);
                
                % 确定保留的主成分数量
                pca_results.n_components = struct();
                pca_results.n_components.kaiser = sum(latent > 1);  % Kaiser准则
                pca_results.n_components.variance_80 = find(pca_results.cumulative_variance >= 80, 1, 'first');
                pca_results.n_components.variance_90 = find(pca_results.cumulative_variance >= 90, 1, 'first');
                pca_results.n_components.variance_95 = find(pca_results.cumulative_variance >= 95, 1, 'first');
                
                % 计算变量载荷
                loadings = coeff .* sqrt(latent)';
                pca_results.loadings = loadings;
                
                % 创建PCA解释表
                pca_table = table((1:length(explained))', explained, pca_results.cumulative_variance, ...
                    'VariableNames', {'Component', 'ExplainedVariance', 'CumulativeVariance'});
                pca_results.variance_table = pca_table;
                
                % 创建变量贡献表
                contribution_table = table(var_names, abs(loadings(:, 1)), abs(loadings(:, 2)), ...
                    'VariableNames', {'Variable', 'PC1_Loading', 'PC2_Loading'});
                contribution_table = sortrows(contribution_table, 'PC1_Loading', 'descend');
                pca_results.contribution_table = contribution_table;
                
                % 保存到主结果
                obj.Results.pca = pca_results;
                
                % 记录PCA统计信息
                obj.Logger.Log('info', sprintf('PCA分析完成：'));
                obj.Logger.Log('info', sprintf('  - Kaiser准则建议保留 %d 个主成分', pca_results.n_components.kaiser));
                obj.Logger.Log('info', sprintf('  - 前 %d 个主成分解释了 80%% 的方差', pca_results.n_components.variance_80));
                obj.Logger.Log('info', sprintf('  - 前 %d 个主成分解释了 90%% 的方差', pca_results.n_components.variance_90));
                obj.Logger.Log('info', sprintf('  - 前 %d 个主成分解释了 95%% 的方差', pca_results.n_components.variance_95));
                
            catch ME
                obj.Logger.LogException(ME, 'PerformPCA');
                pca_results = struct();
            end
        end
        
        function CalculatePartialCorrelations(obj, X, var_names)
            % 计算偏相关系数
            try
                obj.Logger.Log('debug', '计算偏相关系数');
                
                n_vars = size(X, 2);
                partial_corr = zeros(n_vars, n_vars);
                partial_pval = ones(n_vars, n_vars);
                
                % 计算每对变量的偏相关
                for i = 1:n_vars
                    for j = i+1:n_vars
                        % 选择控制变量
                        control_vars = setdiff(1:n_vars, [i, j]);
                        
                        if ~isempty(control_vars)
                            % 计算偏相关
                            [rho, pval] = obj.CalculatePartialCorr(X(:, i), X(:, j), X(:, control_vars));
                            partial_corr(i, j) = rho;
                            partial_corr(j, i) = rho;
                            partial_pval(i, j) = pval;
                            partial_pval(j, i) = pval;
                        else
                            % 如果没有控制变量，使用简单相关
                            [rho, pval] = corr(X(:, i), X(:, j));
                            partial_corr(i, j) = rho;
                            partial_corr(j, i) = rho;
                            partial_pval(i, j) = pval;
                            partial_pval(j, i) = pval;
                        end
                    end
                end
                
                % 对角线设为1
                partial_corr(eye(n_vars) == 1) = 1;
                partial_pval(eye(n_vars) == 1) = 0;
                
                % 保存结果
                obj.Results.partial_correlations = struct();
                obj.Results.partial_correlations.matrix = partial_corr;
                obj.Results.partial_correlations.pvalues = partial_pval;
                
                % 创建偏相关汇总
                obj.CreatePartialCorrelationSummary(partial_corr, var_names);
                
            catch ME
                obj.Logger.LogException(ME, 'CalculatePartialCorrelations');
            end
        end
        
        function [rho, pval] = CalculatePartialCorr(~, x, y, z)
            % 计算单个偏相关系数
            try
                % 回归残差法
                if size(z, 2) > 0
                    % 对x和y分别回归z
                    mdl_x = fitlm(z, x);
                    mdl_y = fitlm(z, y);
                    
                    % 获取残差
                    res_x = mdl_x.Residuals.Raw;
                    res_y = mdl_y.Residuals.Raw;
                    
                    % 计算残差间的相关
                    [rho, pval] = corr(res_x, res_y);
                else
                    % 如果没有控制变量，使用简单相关
                    [rho, pval] = corr(x, y);
                end
                
                % 处理特殊情况
                if isnan(rho)
                    rho = 0;
                    pval = 1;
                end
                
            catch
                rho = 0;
                pval = 1;
            end
        end
        
        function CreatePartialCorrelationSummary(obj, partial_corr, var_names)
            % 创建偏相关汇总
            try
                n_vars = size(partial_corr, 1);
                upper_tri = triu(true(n_vars), 1);
                correlations = partial_corr(upper_tri);
                
                % 计算统计信息
                summary = struct();
                summary.mean_partial_correlation = mean(abs(correlations));
                summary.median_partial_correlation = median(abs(correlations));
                summary.max_partial_correlation = max(abs(correlations));
                summary.min_partial_correlation = min(abs(correlations));
                
                % 对比简单相关和偏相关
                simple_corr = obj.Results.correlations.pearson;
                simple_correlations = simple_corr(upper_tri);
                
                summary.correlation_difference = mean(abs(simple_correlations) - abs(correlations));
                
                obj.Results.partial_correlation_summary = summary;
                
                obj.Logger.Log('info', sprintf('偏相关分析：平均=%.3f, 最大=%.3f', ...
                    summary.mean_partial_correlation, summary.max_partial_correlation));
                
            catch ME
                obj.Logger.LogException(ME, 'CreatePartialCorrelationSummary');
            end
        end
        
        function PerformFactorAnalysis(obj, X, var_names)
            % 执行因子分析
            try
                obj.Logger.Log('debug', '执行因子分析');
                
                % 确定因子数量
                n_factors = obj.DetermineFactorNumber(X);
                
                % 执行因子分析
                [lambda, psi, T, stats, F] = factoran(X, n_factors, 'rotate', 'varimax');
                
                % 保存结果
                factor_results = struct();
                factor_results.n_factors = n_factors;
                factor_results.loadings = lambda;
                factor_results.specific_var = psi;
                factor_results.rotation_matrix = T;
                factor_results.stats = stats;
                factor_results.scores = F;
                
                % 计算因子解释的方差
                factor_results.variance_explained = sum(lambda.^2, 1);
                factor_results.proportion_explained = factor_results.variance_explained / sum(factor_results.variance_explained);
                factor_results.cumulative_proportion = cumsum(factor_results.proportion_explained);
                
                % 创建因子载荷表
                loading_table = array2table([var_names, num2cell(lambda)]);
                loading_table.Properties.VariableNames = ['Variable', ...
                    cellfun(@(x) sprintf('Factor%d', x), num2cell(1:n_factors), 'UniformOutput', false)];
                factor_results.loading_table = loading_table;
                
                obj.Results.factor_analysis = factor_results;
                
                obj.Logger.Log('info', sprintf('因子分析完成：提取 %d 个因子', n_factors));
                
            catch ME
                obj.Logger.LogException(ME, 'PerformFactorAnalysis');
                obj.Results.factor_analysis = struct();
            end
        end
        
        function n_factors = DetermineFactorNumber(obj, X)
            % 确定因子数量
            try
                % 使用特征值准则
                R = corr(X);
                eigenvalues = eig(R);
                n_factors_eigen = sum(eigenvalues > 1);
                
                % 使用碎石图准则
                diff_eigenvalues = diff(eigenvalues);
                diff_diff = diff(diff_eigenvalues);
                [~, elbow_idx] = max(abs(diff_diff));
                n_factors_scree = length(eigenvalues) - elbow_idx - 1;
                
                % 使用解释方差准则
                explained_var = cumsum(eigenvalues) / sum(eigenvalues) * 100;
                n_factors_var80 = find(explained_var >= 80, 1, 'first');
                n_factors_var90 = find(explained_var >= 90, 1, 'first');
                
                % 综合决策
                n_factors = min([n_factors_eigen, n_factors_scree, n_factors_var80]);
                
                obj.Logger.Log('debug', sprintf('因子数量建议：特征值准则=%d, 碎石图准则=%d, 方差准则80%%=%d', ...
                    n_factors_eigen, n_factors_scree, n_factors_var80));
                
                % 确保最小和最大值
                n_factors = max(1, min(n_factors, size(X, 2) - 1));
                
            catch
                % 如果失败，使用默认值
                n_factors = min(3, size(X, 2) - 1);
            end
        end
        
        function PerformVariableClustering(obj, X, var_names)
            % 执行变量聚类分析
            try
                obj.Logger.Log('debug', '执行变量聚类分析');
                
                % 使用相关系数作为距离度量
                R = obj.Results.correlations.pearson;
                D = 1 - abs(R);  % 将相关系数转换为距离
                
                % 执行层次聚类
                Z = linkage(squareform(D), 'average');
                
                % 确定聚类数量
                n_clusters = obj.DetermineClusterNumber(Z, var_names);
                
                % 获取聚类结果
                clusters = cluster(Z, 'maxclust', n_clusters);
                
                % 保存结果
                clustering_results = struct();
                clustering_results.n_clusters = n_clusters;
                clustering_results.linkage_matrix = Z;
                clustering_results.cluster_assignments = clusters;
                clustering_results.distance_matrix = D;
                
                % 创建聚类结果表
                cluster_table = table(var_names, clusters, ...
                    'VariableNames', {'Variable', 'Cluster'});
                cluster_table = sortrows(cluster_table, 'Cluster');
                clustering_results.cluster_table = cluster_table;
                
                % 分析每个聚类
                cluster_stats = [];
                for i = 1:n_clusters
                    cluster_vars = var_names(clusters == i);
                    cluster_size = length(cluster_vars);
                    
                    % 计算聚类内相关性
                    if cluster_size > 1
                        cluster_idx = clusters == i;
                        cluster_corr = R(cluster_idx, cluster_idx);
                        avg_corr = mean(cluster_corr(triu(true(cluster_size), 1)));
                    else
                        avg_corr = 1;
                    end
                    
                    cluster_stats = [cluster_stats; i, cluster_size, avg_corr];
                end
                
                clustering_results.cluster_stats = array2table(cluster_stats, ...
                    'VariableNames', {'Cluster', 'Size', 'AvgIntraCorrelation'});
                
                obj.Results.variable_clustering = clustering_results;
                
                obj.Logger.Log('info', sprintf('变量聚类完成：识别出 %d 个聚类', n_clusters));
                
            catch ME
                obj.Logger.LogException(ME, 'PerformVariableClustering');
                obj.Results.variable_clustering = struct();
            end
        end
        
        function n_clusters = DetermineClusterNumber(obj, Z, var_names)
            % 确定聚类数量
            try
                n_vars = length(var_names);
                
                % 使用肘部法则
                inertia = [];
                for k = 1:min(10, n_vars-1)
                    T = cluster(Z, 'maxclust', k);
                    inertia = [inertia, sum(Z(end-k+2:end, 3))];
                end
                
                % 找到肘部点
                if length(inertia) > 2
                    diff1 = diff(inertia);
                    diff2 = diff(diff1);
                    [~, elbow_idx] = max(abs(diff2));
                    n_clusters = elbow_idx + 1;
                else
                    n_clusters = 2;
                end
                
                % 使用轮廓系数验证
                if n_clusters > 1 && n_clusters < n_vars
                    T = cluster(Z, 'maxclust', n_clusters);
                    try
                        silhouette_vals = silhouette(var_names, T);
                        avg_silhouette = mean(silhouette_vals);
                        
                        if avg_silhouette < 0.3
                            n_clusters = max(2, n_clusters - 1);
                        end
                    catch
                        % silhouette函数可能不适用于所有数据类型
                    end
                end
                
                % 确保合理范围
                n_clusters = max(2, min(n_clusters, floor(n_vars/2)));
                
            catch
                % 默认值
                n_clusters = min(3, length(var_names) - 1);
            end
        end
        
        function CreateVisualizations(obj)
            % 创建相关性分析可视化
            try
                obj.Logger.Log('debug', '创建相关性分析可视化');
                
                figure_dir = fullfile(obj.Config.OutputDirectory, 'figures');
                
                % 1. 相关性热图
                obj.CreateCorrelationHeatmap(figure_dir);
                
                % 2. PCA双标图
                obj.CreatePCABiplot(figure_dir);
                
                % 3. 主成分方差解释图
                obj.CreateVarianceExplainedPlot(figure_dir);
                
                % 4. 因子载荷图
                obj.CreateFactorLoadingPlot(figure_dir);
                
                % 5. 变量聚类树状图
                obj.CreateClusterDendrogram(figure_dir);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateVisualizations');
            end
        end
        
        function CreateCorrelationHeatmap(obj, figure_dir)
            % 创建相关性热图
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 900]);
                
                % 使用Pearson相关系数
                R = obj.Results.correlations.pearson;
                var_names = obj.Results.correlation_summary.high_correlation_pairs.Variable1;
                
                % 创建热图
                h = heatmap(R, 'XDisplayLabels', var_names, 'YDisplayLabels', var_names);
                h.Title = '变量间相关性热图';
                h.FontSize = 10;
                h.Colormap = colormap('RdBu_r');
                h.ColorLimits = [-1, 1];
                
                % 保存图形
                save_figure_util(fig, figure_dir, 'correlation_heatmap');
                
            catch ME
                obj.Logger.LogException(ME, 'CreateCorrelationHeatmap');
            end
        end
        
        function CreatePCABiplot(obj, figure_dir)
            % 创建PCA双标图
            try
                if ~isfield(obj.Results, 'pca') || isempty(obj.Results.pca)
                    return;
                end
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 900]);
                
                pca_results = obj.Results.pca;
                
                % 创建双标图
                biplot(pca_results.coefficients(:, 1:2), ...
                    'Scores', pca_results.scores(:, 1:2), ...
                    'VarLabels', obj.Results.correlation_summary.high_correlation_pairs.Variable1);
                
                xlabel(sprintf('PC1 (%.1f%% variance)', pca_results.explained(1)));
                ylabel(sprintf('PC2 (%.1f%% variance)', pca_results.explained(2)));
                title('PCA双标图');
                grid on;
                
                % 保存图形
                save_figure_util(fig, figure_dir, 'pca_biplot');
                
            catch ME
                obj.Logger.LogException(ME, 'CreatePCABiplot');
            end
        end
        
        function CreateVarianceExplainedPlot(obj, figure_dir)
            % 创建方差解释图
            try
                if ~isfield(obj.Results, 'pca') || isempty(obj.Results.pca)
                    return;
                end
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 900, 600]);
                
                pca_results = obj.Results.pca;
                
                % 创建两个子图
                subplot(2, 1, 1);
                bar(pca_results.explained);
                xlabel('主成分');
                ylabel('解释方差百分比');
                title('各主成分解释方差');
                grid on;
                
                subplot(2, 1, 2);
                plot(pca_results.cumulative_variance, 'o-', 'LineWidth', 2);
                xlabel('主成分数量');
                ylabel('累积解释方差百分比');
                title('累积解释方差');
                grid on;
                
                % 添加参考线
                yline(80, '--', '80%');
                yline(90, '--', '90%');
                yline(95, '--', '95%');
                
                % 保存图形
                save_figure_util(fig, figure_dir, 'variance_explained');
                
            catch ME
                obj.Logger.LogException(ME, 'CreateVarianceExplainedPlot');
            end
        end
        
        function CreateFactorLoadingPlot(obj, figure_dir)
            % 创建因子载荷图
            try
                if ~isfield(obj.Results, 'factor_analysis') || isempty(obj.Results.factor_analysis)
                    return;
                end
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 800]);
                
                factor_results = obj.Results.factor_analysis;
                loadings = factor_results.loadings;
                
                % 绘制前两个因子的载荷图
                if size(loadings, 2) >= 2
                    scatter(loadings(:, 1), loadings(:, 2), 100, 'filled');
                    
                    % 添加标签
                    var_names = obj.Results.correlation_summary.high_correlation_pairs.Variable1;
                    for i = 1:length(var_names)
                        text(loadings(i, 1), loadings(i, 2), var_names{i}, ...
                            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                    end
                    
                    % 添加坐标轴
                    axisLim = max(abs(loadings(:, 1:2)));
                    xlim([-axisLim, axisLim]);
                    ylim([-axisLim, axisLim]);
                    
                    % 添加网格和标签
                    grid on;
                    xlabel('因子1');
                    ylabel('因子2');
                    title('因子载荷图');
                    
                    % 添加单位圆
                    theta = linspace(0, 2*pi, 100);
                    plot(cos(theta), sin(theta), 'k--');
                end
                
                % 保存图形
                save_figure_util(fig, figure_dir, 'factor_loadings');
                
            catch ME
                obj.Logger.LogException(ME, 'CreateFactorLoadingPlot');
            end
        end
        
        function CreateClusterDendrogram(obj, figure_dir)
            % 创建聚类树状图
            try
                if ~isfield(obj.Results, 'variable_clustering') || isempty(obj.Results.variable_clustering)
                    return;
                end
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
                
                clustering_results = obj.Results.variable_clustering;
                
                % 创建树状图
                dendrogram(clustering_results.linkage_matrix, ...
                    'Labels', obj.Results.correlation_summary.high_correlation_pairs.Variable1);
                
                xlabel('变量');
                ylabel('距离');
                title('变量层次聚类树状图');
                
                % 添加聚类切割线
                n_clusters = clustering_results.n_clusters;
                if n_clusters > 1
                    cutoff = clustering_results.linkage_matrix(end-n_clusters+2, 3);
                    yline(cutoff, 'r--', sprintf('%d clusters', n_clusters));
                end
                
                % 保存图形
                save_figure_util(fig, figure_dir, 'cluster_dendrogram');
                
            catch ME
                obj.Logger.LogException(ME, 'CreateClusterDendrogram');
            end
        end
        
        function GenerateAnalysisReport(obj)
            % 生成相关性分析报告
            try
                obj.Logger.Log('debug', '生成相关性分析报告');
                
                report = struct();
                report.timestamp = datetime('now');
                
                % 基本统计
                if isfield(obj.Results, 'correlation_summary')
                    report.correlation_stats = obj.Results.correlation_summary;
                end
                
                % PCA结果
                if isfield(obj.Results, 'pca') && ~isempty(obj.Results.pca)
                    report.pca_summary = struct();
                    report.pca_summary.n_components_80pct = obj.Results.pca.n_components.variance_80;
                    report.pca_summary.n_components_90pct = obj.Results.pca.n_components.variance_90;
                    report.pca_summary.n_components_95pct = obj.Results.pca.n_components.variance_95;
                    report.pca_summary.kaiser_components = obj.Results.pca.n_components.kaiser;
                end
                
                % 因子分析结果
                if isfield(obj.Results, 'factor_analysis') && ~isempty(obj.Results.factor_analysis)
                    report.factor_summary = struct();
                    report.factor_summary.n_factors = obj.Results.factor_analysis.n_factors;
                    report.factor_summary.total_variance_explained = sum(obj.Results.factor_analysis.proportion_explained);
                end
                
                % 聚类结果
                if isfield(obj.Results, 'variable_clustering') && ~isempty(obj.Results.variable_clustering)
                    report.clustering_summary = struct();
                    report.clustering_summary.n_clusters = obj.Results.variable_clustering.n_clusters;
                    report.clustering_summary.avg_intra_correlation = ...
                        mean(obj.Results.variable_clustering.cluster_stats.AvgIntraCorrelation);
                end
                
                % 保存报告
                obj.Results.analysis_report = report;
                
                % 记录关键发现
                obj.LogKeyFindings(report);
                
            catch ME
                obj.Logger.LogException(ME, 'GenerateAnalysisReport');
            end
        end
        
        function LogKeyFindings(obj, report)
            % 记录关键发现
            obj.Logger.CreateSection('相关性分析关键发现');
            
            try
                % 相关性发现
                if isfield(report, 'correlation_stats')
                    stats = report.correlation_stats;
                    obj.Logger.Log('info', sprintf('变量间平均相关性: %.3f', stats.mean_correlation));
                    
                    if isfield(stats, 'high_correlation_pairs') && height(stats.high_correlation_pairs) > 0
                        top_pair = stats.high_correlation_pairs(1, :);
                        obj.Logger.Log('info', sprintf('最高相关性: %s 与 %s (r = %.3f)', ...
                            top_pair.Variable1{1}, top_pair.Variable2{1}, top_pair.Correlation));
                    end
                end
                
                % PCA发现
                if isfield(report, 'pca_summary')
                    pca = report.pca_summary;
                    obj.Logger.Log('info', sprintf('主成分分析：前 %d 个主成分解释了 90%% 的方差', ...
                        pca.n_components_90pct));
                end
                
                % 因子分析发现
                if isfield(report, 'factor_summary')
                    factor = report.factor_summary;
                    obj.Logger.Log('info', sprintf('因子分析：提取了 %d 个因子，解释了 %.1f%% 的总方差', ...
                        factor.n_factors, factor.total_variance_explained * 100));
                end
                
                % 聚类发现
                if isfield(report, 'clustering_summary')
                    cluster = report.clustering_summary;
                    obj.Logger.Log('info', sprintf('变量聚类：识别出 %d 个聚类，平均簇内相关性 %.3f', ...
                        cluster.n_clusters, cluster.avg_intra_correlation));
                end
                
            catch ME
                obj.Logger.LogException(ME, 'LogKeyFindings');
            end
        end
    end
end

% 辅助函数：保存图形
function save_figure_util(fig, output_dir, filename_base)
    % 统一保存图形的辅助函数
    formats = {'svg', 'png'};
    
    for i = 1:length(formats)
        format = formats{i};
        filepath = fullfile(output_dir, [filename_base '.' format]);
        
        try
            switch format
                case 'svg'
                    print(fig, filepath, '-dsvg');
                case 'png'
                    print(fig, filepath, '-dpng', '-r300');
                otherwise
                    saveas(fig, filepath);
            end
        catch
            % 忽略保存失败
        end
    end
    
    % 关闭图形
    close(fig);
end