%% visualization_module.m - 可视化模块
classdef visualization_module
    methods(Static)
        function create_kfold_performance_visualization(cv_results, figure_dir)
            % 创建K折交叉验证各折性能可视化
            % 输入:
            %   cv_results - 交叉验证结果
            %   figure_dir - 图形保存目录
            
            % 获取折数
            k = length(cv_results.accuracy);
            
            % 创建图形
            fig = figure('Name', 'K-Fold Performance by Fold', 'Position', [100, 100, 1200, 800]);
            
            % 准备数据
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
            metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
            n_metrics = length(metrics);
            
            % 创建子图布局
            rows = 2;
            cols = 3;
            
            % 绘制每个指标的折线图
            for i = 1:n_metrics
                metric = metrics{i};
                metric_label = metric_labels{i};
                
                % 创建子图
                subplot(rows, cols, i);
                
                % 获取数据
                values = cv_results.(metric);
                mean_val = cv_results.(['avg_' metric]);
                std_val = cv_results.(['std_' metric]);
                
                % 绘制折线图
                plot(1:k, values, 'o-', 'LineWidth', 1.5, 'Color', [0.3, 0.6, 0.8], 'MarkerFaceColor', [0.3, 0.6, 0.8]);
                hold on;
                
                % 绘制均值线
                plot([0.5, k+0.5], [mean_val, mean_val], 'r--', 'LineWidth', 1.5);
                
                % 绘制标准差区间
                fill([1:k, fliplr(1:k)], [mean_val + std_val * ones(1, k), fliplr(mean_val - std_val * ones(1, k))], ...
                    [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
                
                % 设置图形属性
                xlabel('折数', 'FontSize', 10);
                ylabel(metric_label, 'FontSize', 10);
                title(sprintf('%s (均值=%.3f, 标准差=%.3f)', metric_label, mean_val, std_val), 'FontSize', 12);
                grid on;
                xlim([0.5, k+0.5]);
                
                % 调整Y轴范围
                if strcmp(metric, 'auc') || strcmp(metric, 'f1_score')
                    ylim([0.5, 1]);
                else
                    ylim([0, 1]);
                end
                
                % 添加数据点标签
                for j = 1:k
                    text(j, values(j) + 0.02, sprintf('%.3f', values(j)), ...
                        'HorizontalAlignment', 'center', 'FontSize', 7);
                end
                
                % 添加图例
                legend({'各折值', '均值', '标准差区间'}, 'Location', 'best', 'FontSize', 8);
            end
            
            % 添加总标题
            sgtitle(sprintf('K折交叉验证各折性能指标 (K=%d)', k), 'FontSize', 16, 'FontWeight', 'bold');
            set(gcf, 'Color', 'white');
            
            % 保存图形
            utils.save_figure(fig, figure_dir, 'kfold_performance_by_fold', 'Formats', {'svg'});
            close(fig);
        end
        
        function create_residual_analysis_plot(method, all_probs, residuals, deviance_residuals, all_labels, figure_dir)
            % 创建残差分析图
            % 输入:
            %   method - 方法名称
            %   all_probs - 预测概率
            %   residuals - 皮尔森残差
            %   deviance_residuals - 偏差残差
            %   all_labels - 实际标签
            %   figure_dir - 图形保存目录
            
            % 创建残差分析图
            fig = figure('Name', sprintf('%s Residual Analysis', method), 'Position', [100, 100, 1200, 900]);
            
            % 创建2x3的子图布局
            % 创建子图1：皮尔森残差 vs 预测概率
            subplot(2, 3, 1);
            scatter(all_probs, residuals, 30, 'filled', 'MarkerFaceAlpha', 0.6);
            hold on;
            plot([0, 1], [0, 0], 'k--');
            xlabel('预测概率', 'FontSize', 10);
            ylabel('皮尔森残差', 'FontSize', 10);
            title('皮尔森残差 vs 预测概率', 'FontSize', 12);
            grid on;
            
            % 添加平滑曲线
            try
                [xData, yData] = prepareCurveData(all_probs, residuals);
                smoothed = smooth(xData, yData, 0.2, 'loess');
                plot(xData, smoothed, 'r-', 'LineWidth', 2);
            catch
                % 如果平滑失败，忽略
            end
            
            % 创建子图2：Deviance残差 vs 预测概率
            subplot(2, 3, 2);
            scatter(all_probs, deviance_residuals, 30, 'filled', 'MarkerFaceAlpha', 0.6);
            hold on;
            plot([0, 1], [0, 0], 'k--');
            xlabel('预测概率', 'FontSize', 10);
            ylabel('Deviance残差', 'FontSize', 10);
            title('Deviance残差 vs 预测概率', 'FontSize', 12);
            grid on;
            
            % 添加平滑曲线
            try
                [xData, yData] = prepareCurveData(all_probs, deviance_residuals);
                smoothed = smooth(xData, yData, 0.2, 'loess');
                plot(xData, smoothed, 'r-', 'LineWidth', 2);
            catch
                % 如果平滑失败，忽略
            end
            
            % 创建子图3：皮尔森残差箱线图
            subplot(2, 3, 3);
            boxplot(residuals, all_labels, 'Labels', {'0', '1'});
            ylabel('皮尔森残差', 'FontSize', 10);
            title('按实际类别分组的皮尔森残差', 'FontSize', 12);
            grid on;
            
            % 创建子图4：Deviance残差箱线图 (新增)
            subplot(2, 3, 4);
            boxplot(deviance_residuals, all_labels, 'Labels', {'0', '1'});
            ylabel('Deviance残差', 'FontSize', 10);
            title('按实际类别分组的Deviance残差', 'FontSize', 12);
            grid on;
            
            % 创建子图5：皮尔森残差QQ图
            subplot(2, 3, 5);
            qqplot(residuals);
            title('皮尔森残差QQ图', 'FontSize', 12);
            grid on;
            
            % 创建子图6：Deviance残差QQ图 (新增)
            subplot(2, 3, 6);
            qqplot(deviance_residuals);
            title('Deviance残差QQ图', 'FontSize', 12);
            grid on;
            
            % 添加总标题
            sgtitle(sprintf('%s方法的残差分析', method), 'FontSize', 14, 'FontWeight', 'bold');
            
            % 保存图形
            utils.save_figure(fig, figure_dir, sprintf('%s_residual_analysis', method), 'Formats', {'svg'});
            close(fig);
        end
        
        function create_residual_comparison_plots(methods_with_residuals, all_pearson_residuals, all_deviance_residuals, figure_dir)
            % 创建残差比较图
            % 输入:
            %   methods_with_residuals - 包含残差的方法
            %   all_pearson_residuals - 所有皮尔森残差
            %   all_deviance_residuals - 所有偏差残差
            %   figure_dir - 图形保存目录
            
            % 创建残差比较图 - 皮尔森残差
            fig1 = figure('Name', 'Pearson Residuals Comparison', 'Position', [100, 100, 1000, 600]);
            
            % 创建箱线图比较
            boxplot(cell2mat(all_pearson_residuals), repelem(1:length(methods_with_residuals), cellfun(@length, all_pearson_residuals)), ...
                'Labels', methods_with_residuals, 'Notch', 'on');
            
            % 设置图形属性
            ylabel('皮尔森残差', 'FontSize', 12, 'FontWeight', 'bold');
            title('各方法皮尔森残差分布比较', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            % 添加零线
            hold on;
            plot(xlim, [0 0], 'k--');
            
            % 保存图形
            utils.save_figure(fig1, figure_dir, 'pearson_residuals_comparison', 'Formats', {'svg'});
            close(fig1);
            
            % 创建残差比较图 - Deviance残差 (新增)
            fig2 = figure('Name', 'Deviance Residuals Comparison', 'Position', [100, 100, 1000, 600]);
            
            % 创建箱线图比较
            boxplot(cell2mat(all_deviance_residuals), repelem(1:length(methods_with_residuals), cellfun(@length, all_deviance_residuals)), ...
                'Labels', methods_with_residuals, 'Notch', 'on');
            
            % 设置图形属性
            ylabel('Deviance残差', 'FontSize', 12, 'FontWeight', 'bold');
            title('各方法Deviance残差分布比较', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            % 添加零线
            hold on;
            plot(xlim, [0 0], 'k--');
            
            % 保存图形
            utils.save_figure(fig2, figure_dir, 'deviance_residuals_comparison', 'Formats', {'svg'});
            close(fig2);
        end
    end
end