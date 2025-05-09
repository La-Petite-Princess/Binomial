%% feature_module.m - 特征工程模块
classdef feature_module
    methods(Static)
        function [X_cleaned, vif_values, removed_vars] = check_collinearity(X, var_names)
            % 检查并处理多重共线性
            % 输入:
            %   X - 自变量矩阵
            %   var_names - 变量名称
            % 输出:
            %   X_cleaned - 处理后的自变量矩阵
            %   vif_values - VIF值
            %   removed_vars - 被移除的变量标记
            
            t_start = toc;
            
            % 计算相关矩阵 - 使用更高效的计算
            R = corr(X, 'Type', 'Pearson');
            
            % 检查相关矩阵
            if any(isnan(R(:))) || any(isinf(R(:)))
                error('相关矩阵 R 包含 NaN 或 Inf，请检查输入数据！');
            end
            
            % 初始化输出变量
            removed_vars = false(size(X, 2), 1);
            vif_values = zeros(size(X, 2), 1);
            
            % 优化：先检查矩阵的条件数而不是秩
            cond_num = cond(R);
            if cond_num > 30
                logger.log_message('warning', sprintf('相关矩阵条件数较高(%.2f)，可能存在多重共线性', cond_num));
                
                % 使用SVD代替直接求逆计算VIF，更数值稳定
                [U, S, V] = svd(R);
                s = diag(S);
                
                % 如果最小奇异值小于阈值，认为矩阵接近奇异
                if min(s) < 1e-10
                    warning('MatrixError:NearSingular', '相关矩阵接近奇异，使用PCA处理多重共线性');
                    
                    % 使用主成分分析处理多重共线性
                    [~, score, ~, ~, explained] = pca(X, 'Algorithm', 'svd');
                    cum_var = cumsum(explained);
                    k = find(cum_var >= 95, 1, 'first'); % 保留解释95%方差的成分，提高保留信息
                    X_cleaned = score(:, 1:k);
                    
                    logger.log_message('warning', sprintf('使用PCA降维，从%d个变量降至%d个主成分', size(X, 2), k));
                    
                    % 所有原始变量都被"移除"
                    removed_vars = true(size(X, 2), 1);
                    vif_values = ones(size(X, 2), 1) * Inf;
                    return;
                else
                    % 使用SVD计算VIF
                    vif_values = zeros(size(X, 2), 1);
                    for i = 1:size(X, 2)
                        % 选择第i列作为因变量
                        y_i = X(:, i);
                        % 选择其他列作为自变量
                        X_i = X(:, setdiff(1:size(X, 2), i));
                        % 计算R²
                        b = X_i \ y_i;
                        y_hat = X_i * b;
                        SS_total = sum((y_i - mean(y_i)).^2);
                        SS_residual = sum((y_i - y_hat).^2);
                        R_squared = 1 - SS_residual/SS_total;
                        % 计算VIF
                        vif_values(i) = 1 / (1 - R_squared);
                    end
                end
            else
                % 条件良好，直接计算VIF
                try
                    % 使用更高效的计算方法
                    vif_values = zeros(size(X, 2), 1);
                    for i = 1:size(X, 2)
                        % 使用线性回归而不是直接求逆计算VIF，数值更稳定
                        idx = setdiff(1:size(X, 2), i);
                        mdl = fitlm(X(:, idx), X(:, i));
                        vif_values(i) = 1 / (1 - mdl.Rsquared.Ordinary);
                    end
                catch ME
                    logger.log_message('warning', sprintf('VIF计算失败，使用SVD方法: %s', ME.message));
                    % 使用SVD方法作为备选
                    vif_values = ones(size(X, 2), 1) ./ diag(pinv(R));
                end
            end
            
            % 输出VIF值
            logger.log_message('info', '自变量的VIF值：');
            for i = 1:length(vif_values)
                logger.log_message('info', sprintf('%s: %.2f', var_names{i}, vif_values(i)));
            end
            
            % 找出高VIF值的变量 - 使用阈值为10
            high_vif = find(vif_values > 10);
            if ~isempty(high_vif)
                logger.log_message('warning', '移除高VIF变量索引：');
                for i = 1:length(high_vif)
                    logger.log_message('warning', sprintf('%s (VIF = %.2f)', var_names{high_vif(i)}, vif_values(high_vif(i))));
                end
                
                % 移除高VIF值的变量
                removed_vars(high_vif) = true;
                X_cleaned = X(:, ~removed_vars);
                
                % 递归检查剩余变量的VIF
                if sum(~removed_vars) > 1
                    logger.log_message('info', '递归检查剩余变量的VIF值');
                    [X_cleaned_rec, vif_values_rec, removed_vars_rec] = feature_module.check_collinearity(X_cleaned, var_names(~removed_vars));
                    
                    % 更新removed_vars以反映递归结果
                    still_removed = false(size(X, 2), 1);
                    still_removed(~removed_vars) = removed_vars_rec;
                    removed_vars = removed_vars | still_removed;
                    
                    X_cleaned = X_cleaned_rec;
                end
            else
                X_cleaned = X;
            end
            
            t_end = toc;
            logger.log_message('info', sprintf('多重共线性检查完成，最终自变量数：%d，耗时：%.2f秒', ...
                size(X_cleaned, 2), t_end - t_start));
        end
        
        function [pca_results] = analyze_variable_correlations(X, var_names)
            % 分析变量之间的相关性
            % 输入:
            %   X - 自变量矩阵
            %   var_names - 变量名称
            % 输出:
            %   pca_results - PCA分析结果
            
            t_start = toc;
            
            % 计算变量之间的相关性
            R = corr(X);
            
            % 创建更高分辨率的热图
            fig = figure('Name', '变量相关性矩阵', 'Position', [100, 100, 1000, 900]);
            
            % 使用更美观的热图
            h = heatmap(R, 'XDisplayLabels', var_names, 'YDisplayLabels', var_names);
            h.Title = '变量相关性矩阵';
            h.FontSize = 10;
            h.Colormap = parula;
            
            % 调整colorbar
            caxis([-1, 1]);
            colorbar;
            
            % 保存矢量图
            utils.save_figure(fig, 'results', 'variable_correlation', 'Formats', {'svg'});
            close(fig);
            
            % 识别高度相关的变量对
            [rows, cols] = find(triu(abs(R) > 0.8, 1));
            if ~isempty(rows)
                logger.log_message('warning', '发现高度相关的变量对 (|r| > 0.8):');
                for i = 1:length(rows)
                    logger.log_message('warning', sprintf('%s 与 %s: r = %.2f', var_names{rows(i)}, var_names{cols(i)}, R(rows(i), cols(i))));
                end
            else
                logger.log_message('info', '未发现高度相关的变量对 (|r| > 0.8)');
            end
            
            % 增加主成分分析可视化 - 仅在变量较多时使用
            pca_results = struct(); % 初始化PCA结果结构
            
            if length(var_names) > 3
                try
                    % 执行PCA
                    [coeff, score, ~, ~, explained, mu] = pca(X);
                    
                    % 存储PCA结果
                    pca_results.coeff = coeff;
                    pca_results.score = score;
                    pca_results.explained = explained;
                    pca_results.mu = mu;
                    pca_results.cum_explained = cumsum(explained);
                    
                    % 创建PCA双线图
                    fig2 = figure('Name', '主成分分析', 'Position', [100, 100, 1200, 900]);
                    
                    % 绘制变量在前两个主成分上的投影
                    subplot(2, 2, 1);
                    biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', var_names);
                    title('变量在主成分1-2上的投影');
                    grid on;
                    
                    % 绘制解释方差比例
                    subplot(2, 2, 2);
                    bar(explained);
                    xlabel('主成分');
                    ylabel('解释方差百分比');
                    title('各主成分解释方差比例');
                    grid on;
                    
                    % 绘制累积解释方差
                    subplot(2, 2, 3);
                    plot(cumsum(explained), 'o-', 'LineWidth', 2);
                    xlabel('主成分数量');
                    ylabel('累积解释方差百分比');
                    title('累积解释方差');
                    grid on;
                    
                    % 增加百分比标注
                    cum_explained = cumsum(explained);
                    hold on;
                    
                    % 选取几个关键点标注
                    key_components = [1, min(3, length(cum_explained)), min(5, length(cum_explained))];
                    for i = 1:length(key_components)
                        idx = key_components(i);
                        plot(idx, cum_explained(idx), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
                        text(idx, cum_explained(idx) + 2, sprintf('%.1f%%', cum_explained(idx)), ...
                            'HorizontalAlignment', 'center', 'FontSize', 9);
                    end
                    
                    % 绘制95%方差的线
                    idx_95 = find(cum_explained >= 95, 1, 'first');
                    if ~isempty(idx_95)
                        plot([0, idx_95, idx_95], [95, 95, 0], 'k--');
                        text(idx_95-0.5, 96, sprintf('95%%方差需要%d个主成分', idx_95), ...
                            'HorizontalAlignment', 'right', 'FontSize', 9);
                    end
                    
                    % 绘制变量对主成分的贡献
                    subplot(2, 2, 4);
                    imagesc(abs(coeff(:, 1:min(5, size(coeff, 2)))));
                    colorbar;
                    xlabel('主成分');
                    set(gca, 'YTick', 1:length(var_names), 'YTickLabel', var_names);
                    title('变量对前5个主成分的贡献(绝对值)');
                    
                    % 保存PCA图
                    utils.save_figure(fig2, 'results', 'pca_analysis', 'Formats', {'svg'});
                    close(fig2);
                    
                    % 创建主成分累计方差表
                    cum_var_table = table((1:length(explained))', explained, cum_explained, ...
                        'VariableNames', {'Component', 'ExplainedVariance', 'CumulativeVariance'});
                    
                    % 输出累计方差
                    logger.log_message('info', '主成分累计方差:');
                    key_idx = [1, 2, 3, min(5, length(cum_explained)), min(10, length(cum_explained))];
                    key_idx = unique(key_idx);
                    for i = 1:length(key_idx)
                        idx = key_idx(i);
                        logger.log_message('info', sprintf('  前%d个主成分解释了%.2f%%的总方差', idx, cum_explained(idx)));
                    end
                    
                    % 创建新的累计方差图
                    fig3 = figure('Name', '主成分累计方差', 'Position', [100, 100, 900, 600]);
                    
                    % 绘制阶梯图
                    stairs(cum_explained, 'LineWidth', 2);
                    hold on;
                    
                    % 标记95%方差
                    if ~isempty(idx_95)
                        plot([0, idx_95], [95, 95], 'r--');
                        plot([idx_95, idx_95], [0, 95], 'r--');
                        text(idx_95 + 0.1, 60, sprintf('95%%方差需要%d个主成分', idx_95), ...
                            'FontSize', 10, 'Color', 'r');
                    end
                    
                    % 标记80%方差
                    idx_80 = find(cum_explained >= 80, 1, 'first');
                    if ~isempty(idx_80)
                        plot([0, idx_80], [80, 80], 'g--');
                        plot([idx_80, idx_80], [0, 80], 'g--');
                        text(idx_80 + 0.1, 40, sprintf('80%%方差需要%d个主成分', idx_80), ...
                            'FontSize', 10, 'Color', 'g');
                    end
                    
                    % 设置图形属性
                    xlabel('主成分数量', 'FontSize', 12, 'FontWeight', 'bold');
                    ylabel('累计解释方差百分比', 'FontSize', 12, 'FontWeight', 'bold');
                    title('主成分分析累计解释方差', 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    xlim([0, min(15, length(explained))]);
                    ylim([0, 100]);
                    
                    % 保存图形
                    utils.save_figure(fig3, 'results', 'cumulative_variance', 'Formats', {'svg'});
                    close(fig3);
                    
                    % 创建主成分载荷可视化
                    fig4 = figure('Name', '主成分载荷', 'Position', [100, 100, 1200, 600]);
                    
                    % 显示前3个主成分的载荷
                    num_pc = min(3, size(coeff, 2));
                    for i = 1:num_pc
                        subplot(1, num_pc, i);
                        bar(coeff(:, i));
                        xlabel('变量');
                        ylabel('载荷系数');
                        title(sprintf('主成分%d载荷', i));
                        set(gca, 'XTick', 1:length(var_names), 'XTickLabel', var_names, 'XTickLabelRotation', 45);
                        grid on;
                    end
                    
                    % 保存图形
                    utils.save_figure(fig4, 'results', 'principal_component_loadings', 'Formats', {'svg'});
                    close(fig4);
                catch ME
                    logger.log_message('warning', sprintf('PCA可视化失败: %s', ME.message));
                end
            end
            
            t_end = toc;
            logger.log_message('info', sprintf('变量相关性分析完成，耗时：%.2f秒', t_end - t_start));
        end
    end
end
