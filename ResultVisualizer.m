classdef ResultVisualizer < handle    
    % ResultVisualizer - 模型结果可视化
    %
    % 该类负责将模型的各种结果以图形化方式展示，包括系数图、
    % 拟合优度图、残差诊断图、变量重要性图等。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   modelResults - 模型结果结构体
    %   figureHandles - 生成的图形句柄
    %   figureCounter - 图形计数器
    %   figuresInfo - 图形信息记录
    %   colorScheme - 颜色方案
    %   savePath - 图形保存路径
    %   highDPI - 是否使用高DPI输出
    
    properties
        logger            % 日志记录器
        modelResults      % 模型结果结构体
        figureHandles     % 生成的图形句柄
        figureCounter     % 图形计数器
        figuresInfo       % 图形信息记录
        colorScheme       % 颜色方案
        savePath          % 图形保存路径
        highDPI           % 是否使用高DPI输出
    end
    
    methods
        function obj = ResultVisualizer(logger, colorScheme)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            %   colorScheme - 颜色方案（可选）
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('ResultVisualizer');
            else
                obj.logger = logger;
            end
            
            if nargin < 2 || isempty(colorScheme)
                % 默认颜色方案
                obj.colorScheme = struct(...
                    'primary', [0.2, 0.4, 0.6], ...
                    'secondary', [0.8, 0.3, 0.3], ...
                    'tertiary', [0.3, 0.7, 0.4], ...
                    'light', [0.8, 0.8, 0.8], ...
                    'dark', [0.2, 0.2, 0.2], ...
                    'highlight', [1.0, 0.5, 0.0], ...
                    'gradient', {{'#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f'}});
            else
                obj.colorScheme = colorScheme;
            end
            
            obj.figureHandles = {};
            obj.figureCounter = 0;
            obj.figuresInfo = struct('title', {}, 'description', {}, 'filename', {});
            obj.savePath = pwd;
            obj.highDPI = true;
            
            obj.logger.info('结果可视化模块已初始化');
        end
        
        function setModelResults(obj, modelResults)
            % 设置模型结果
            %
            % 参数:
            %   modelResults - 包含模型分析结果的结构体
            
            obj.modelResults = modelResults;
            obj.logger.debug('模型结果已设置');
        end
        
        function setSavePath(obj, path)
            % 设置图形保存路径
            %
            % 参数:
            %   path - 图形保存路径
            
            if exist(path, 'dir') || mkdir(path)
                obj.savePath = path;
                obj.logger.info('图形保存路径已设置为: %s', path);
            else
                obj.logger.error('无法创建或访问路径: %s', path);
            end
        end
        
        function plotCoefficientEstimates(obj, coefficients, stdErrors, names, figTitle)
            % 绘制系数估计及置信区间
            %
            % 参数:
            %   coefficients - 系数向量
            %   stdErrors - 标准误向量
            %   names - 变量名称元胞数组（可选）
            %   figTitle - 图形标题（可选）
            
            if nargin < 5 || isempty(figTitle)
                figTitle = '系数估计与置信区间';
            end
            
            if nargin < 4 || isempty(names)
                names = arrayfun(@(i) sprintf('Var %d', i), ...
                    1:length(coefficients), 'UniformOutput', false);
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 800, 600]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 计算95%置信区间
            ci95 = 1.96 * stdErrors;
            
            % 按系数绝对值排序
            [~, idx] = sort(abs(coefficients), 'descend');
            sortedCoefs = coefficients(idx);
            sortedErrors = ci95(idx);
            sortedNames = names(idx);
            
            % 创建水平条形图
            barh(sortedCoefs, 'FaceColor', obj.colorScheme.primary);
            
            % 添加误差线
            hold on;
            errorbar(sortedCoefs, 1:length(sortedCoefs), sortedErrors, sortedErrors, '.', ...
                'Color', obj.colorScheme.secondary, 'LineWidth', 1.5, 'CapSize', 10, ...
                'Marker', 'none', 'LineStyle', 'none');
            
            % 添加零线
            plot([0, 0], [0, length(sortedCoefs)+1], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            hold off;
            
            % 设置坐标轴和标签
            yticks(1:length(sortedCoefs));
            yticklabels(sortedNames);
            grid on;
            xlabel('系数值');
            ylabel('变量');
            title(figTitle);
            
            % 美化图形
            set(gca, 'FontSize', 11, 'FontWeight', 'bold', 'Box', 'on');
            set(gca, 'YGrid', 'off', 'XGrid', 'on');
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '系数估计值及95%置信区间';
            obj.figuresInfo(obj.figureCounter).filename = 'coefficient_estimates';
            
            obj.logger.info('系数估计图已生成');
        end
        
        function plotVariableImportance(obj, importance, names, figTitle)
            % 绘制变量重要性图
            %
            % 参数:
            %   importance - 变量重要性向量
            %   names - 变量名称元胞数组（可选）
            %   figTitle - 图形标题（可选）
            
            if nargin < 4 || isempty(figTitle)
                figTitle = '变量重要性';
            end
            
            if nargin < 3 || isempty(names)
                names = arrayfun(@(i) sprintf('Var %d', i), ...
                    1:length(importance), 'UniformOutput', false);
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 800, 600]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 按重要性排序
            [sortedImportance, idx] = sort(importance, 'descend');
            sortedNames = names(idx);
            
            % 规格化重要性为0-100%
            if max(sortedImportance) > 0
                normImportance = 100 * sortedImportance / max(sortedImportance);
            else
                normImportance = sortedImportance;
            end
            
            % 创建条形图
            barh(normImportance, 'FaceColor', obj.colorScheme.tertiary);
            
            % 设置坐标轴和标签
            yticks(1:length(normImportance));
            yticklabels(sortedNames);
            grid on;
            xlabel('相对重要性 (%)');
            ylabel('变量');
            title(figTitle);
            
            % 美化图形
            set(gca, 'FontSize', 11, 'FontWeight', 'bold', 'Box', 'on');
            set(gca, 'YGrid', 'off', 'XGrid', 'on');
            
            % 添加数值标签
            for i = 1:length(normImportance)
                if normImportance(i) > 5  % 只在重要性超过5%时显示标签
                    text(normImportance(i) + 1, i, sprintf('%.1f%%', normImportance(i)), ...
                        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
                        'FontWeight', 'bold');
                end
            end
            
            % 设置x轴范围
            if max(normImportance) > 0
                xlim([0, max(normImportance) * 1.1]);
            end
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '变量相对重要性排名';
            obj.figuresInfo(obj.figureCounter).filename = 'variable_importance';
            
            obj.logger.info('变量重要性图已生成');
        end
        
        function plotModelFit(obj, observed, predicted, figTitle)
            % 绘制模型拟合优度图
            %
            % 参数:
            %   observed - 观测值向量
            %   predicted - 预测值向量
            %   figTitle - 图形标题（可选）
            
            if nargin < 4 || isempty(figTitle)
                figTitle = '模型拟合优度';
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 800, 700]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 创建散点图
            subplot(2, 2, 1);
            scatter(observed, predicted, 50, obj.colorScheme.primary, 'filled', ...
                'MarkerFaceAlpha', 0.7);
            
            % 添加对角线（完美拟合线）
            hold on;
            minVal = min([observed; predicted]);
            maxVal = max([observed; predicted]);
            plot([minVal, maxVal], [minVal, maxVal], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1.5);
            
            % 添加回归线
            p = polyfit(observed, predicted, 1);
            x_reg = linspace(minVal, maxVal, 100);
            y_reg = polyval(p, x_reg);
            plot(x_reg, y_reg, '-', 'Color', obj.colorScheme.secondary, 'LineWidth', 2);
            hold off;
            
            % 计算相关系数和RMSE
            r = corr(observed, predicted);
            rmse = sqrt(mean((observed - predicted).^2));
            
            % 添加文本说明
            text(0.05, 0.95, sprintf('R = %.3f\nRMSE = %.3f', r, rmse), ...
                'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [1 1 1 0.7]);
            
            grid on;
            xlabel('观测值');
            ylabel('预测值');
            title('观测值 vs. 预测值');
            
            % 残差直方图
            subplot(2, 2, 2);
            residuals = observed - predicted;
            histogram(residuals, min(30, max(10, ceil(sqrt(length(residuals))))), ...
                'FaceColor', obj.colorScheme.primary, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
            
            % 添加正态分布曲线
            hold on;
            [counts, edges] = histcounts(residuals, min(30, max(10, ceil(sqrt(length(residuals))))));
            binWidth = edges(2) - edges(1);
            x_norm = linspace(min(residuals), max(residuals), 100);
            y_norm = normpdf(x_norm, mean(residuals), std(residuals)) * length(residuals) * binWidth;
            plot(x_norm, y_norm, '-', 'Color', obj.colorScheme.secondary, 'LineWidth', 2);
            hold off;
            
            grid on;
            xlabel('残差');
            ylabel('频数');
            title('残差分布');
            
            % 残差与预测值
            subplot(2, 2, 3);
            scatter(predicted, residuals, 50, obj.colorScheme.primary, 'filled', ...
                'MarkerFaceAlpha', 0.7);
            
            % 添加零线
            hold on;
            plot([minVal, maxVal], [0, 0], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1.5);
            
            % 添加LOWESS平滑曲线
            try
                span = 0.75;  % LOWESS平滑参数
                [sortedPred, sortIdx] = sort(predicted);
                sortedResid = residuals(sortIdx);
                smoothedResid = smooth(sortedResid, span, 'lowess');
                plot(sortedPred, smoothedResid, '-', 'Color', obj.colorScheme.secondary, 'LineWidth', 2);
            catch
                % 如果LOWESS失败，使用简单移动平均
                obj.logger.warn('LOWESS平滑失败，使用简单移动平均');
                window = max(5, ceil(length(predicted) / 20));
                [sortedPred, sortIdx] = sort(predicted);
                sortedResid = residuals(sortIdx);
                movAvg = movmean(sortedResid, window);
                plot(sortedPred, movAvg, '-', 'Color', obj.colorScheme.secondary, 'LineWidth', 2);
            end
            hold off;
            
            grid on;
            xlabel('预测值');
            ylabel('残差');
            title('残差 vs. 预测值');
            
            % Q-Q图
            subplot(2, 2, 4);
            qqplot(residuals);
            grid on;
            title('残差Q-Q图');
            
            % 调整整体布局
            sgtitle(figTitle, 'FontSize', 14, 'FontWeight', 'bold');
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '模型拟合优度和残差分析图';
            obj.figuresInfo(obj.figureCounter).filename = 'model_fit';
            
            obj.logger.info('模型拟合优度图已生成');
        end
        
        function plotResidualDiagnostics(obj, residuals, leverages, cookDist, figTitle)
            % 绘制残差诊断图
            %
            % 参数:
            %   residuals - 残差向量
            %   leverages - 杠杆值向量
            %   cookDist - Cook距离向量
            %   figTitle - 图形标题（可选）
            
            if nargin < 5 || isempty(figTitle)
                figTitle = '残差诊断';
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 900, 700]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 标准化残差直方图
            subplot(2, 2, 1);
            stdResid = residuals / std(residuals);
            histogram(stdResid, min(30, max(10, ceil(sqrt(length(stdResid))))), ...
                'FaceColor', obj.colorScheme.primary, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
            
            % 添加正态分布曲线
            hold on;
            [counts, edges] = histcounts(stdResid, min(30, max(10, ceil(sqrt(length(stdResid))))));
            binWidth = edges(2) - edges(1);
            x_norm = linspace(min(stdResid), max(stdResid), 100);
            y_norm = normpdf(x_norm, 0, 1) * length(stdResid) * binWidth;
            plot(x_norm, y_norm, '-', 'Color', obj.colorScheme.secondary, 'LineWidth', 2);
            
            % 添加±2和±3标准差线
            for sd = [-3, -2, 2, 3]
                line([sd, sd], [0, max(counts)*1.1], 'Color', obj.colorScheme.dark, ...
                    'LineStyle', '--', 'LineWidth', 1);
            end
            hold off;
            
            grid on;
            xlabel('标准化残差');
            ylabel('频数');
            title('标准化残差分布');
            
            % 杠杆值-残差图
            subplot(2, 2, 2);
            scatter(leverages, stdResid, 50, obj.colorScheme.primary, 'filled', ...
                'MarkerFaceAlpha', 0.7);
            
            % 添加阈值线
            hold on;
            n = length(residuals);
            p = 10;  % 假设自变量数量，可以根据实际调整
            hThresh = 2 * p / n;
            plot([hThresh, hThresh], [-4, 4], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            plot([0, max(leverages)*1.1], [2, 2], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            plot([0, max(leverages)*1.1], [-2, -2], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            
            % 标记异常点
            outlierIdx = find((abs(stdResid) > 2) | (leverages > hThresh));
            if ~isempty(outlierIdx)
                scatter(leverages(outlierIdx), stdResid(outlierIdx), 100, ...
                    obj.colorScheme.secondary, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
                
                % 标记异常点索引
                for i = 1:length(outlierIdx)
                    text(leverages(outlierIdx(i)), stdResid(outlierIdx(i)), ...
                        sprintf(' %d', outlierIdx(i)), 'FontSize', 8, 'FontWeight', 'bold');
                end
            end
            hold off;
            
            grid on;
            xlabel('杠杆值');
            ylabel('标准化残差');
            title('杠杆值-残差图');
            
            % Cook距离图
            subplot(2, 2, 3);
            stem(cookDist, 'Color', obj.colorScheme.primary, 'LineWidth', 1.5, 'Marker', 'o', ...
                'MarkerSize', 6, 'MarkerFaceColor', obj.colorScheme.primary);
            
            % 添加阈值线
            hold on;
            cookThresh = 4 / n;
            plot([0, n+1], [cookThresh, cookThresh], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            
            % 标记异常点
            outlierIdx = find(cookDist > cookThresh);
            if ~isempty(outlierIdx)
                stem(outlierIdx, cookDist(outlierIdx), 'Color', obj.colorScheme.secondary, ...
                    'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 8, ...
                    'MarkerFaceColor', obj.colorScheme.secondary, 'MarkerEdgeColor', 'k');
                
                % 标记异常点索引
                for i = 1:length(outlierIdx)
                    text(outlierIdx(i), cookDist(outlierIdx(i)), ...
                        sprintf(' %d', outlierIdx(i)), 'FontSize', 8, 'FontWeight', 'bold');
                end
            end
            hold off;
            
            grid on;
            xlabel('观测值索引');
            ylabel('Cook距离');
            title('Cook距离图');
            
            % 残差-观测值索引图
            subplot(2, 2, 4);
            plot(stdResid, 'o-', 'Color', obj.colorScheme.primary, 'LineWidth', 1, ...
                'MarkerFaceColor', obj.colorScheme.primary, 'MarkerSize', 6);
            
            % 添加±2标准差线
            hold on;
            plot([0, n+1], [2, 2], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            plot([0, n+1], [-2, -2], '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            
            % 标记异常点
            outlierIdx = find(abs(stdResid) > 2);
            if ~isempty(outlierIdx)
                plot(outlierIdx, stdResid(outlierIdx), 'o', 'Color', obj.colorScheme.secondary, ...
                    'MarkerFaceColor', obj.colorScheme.secondary, 'MarkerSize', 8, 'LineWidth', 1.5);
                
                % 标记异常点索引
                for i = 1:length(outlierIdx)
                    text(outlierIdx(i), stdResid(outlierIdx(i)), ...
                        sprintf(' %d', outlierIdx(i)), 'FontSize', 8, 'FontWeight', 'bold');
                end
            end
            hold off;
            
            grid on;
            xlabel('观测值索引');
            ylabel('标准化残差');
            title('残差时序图');
            
            % 调整整体布局
            sgtitle(figTitle, 'FontSize', 14, 'FontWeight', 'bold');
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '残差诊断和异常点检测图';
            obj.figuresInfo(obj.figureCounter).filename = 'residual_diagnostics';
            
            obj.logger.info('残差诊断图已生成');
        end
        
        function plotCorrelationHeatmap(obj, corrMatrix, names, figTitle)
            % 绘制相关性热图
            %
            % 参数:
            %   corrMatrix - 相关系数矩阵
            %   names - 变量名称元胞数组（可选）
            %   figTitle - 图形标题（可选）
            
            if nargin < 4 || isempty(figTitle)
                figTitle = '相关性热图';
            end
            
            if nargin < 3 || isempty(names)
                names = arrayfun(@(i) sprintf('Var %d', i), ...
                    1:size(corrMatrix, 1), 'UniformOutput', false);
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 800, 700]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 创建热图
            imagesc(corrMatrix);
            
            % 设置颜色图谱
            colormap(jet);
            colorbar;
            
            % 设置坐标轴和标签
            xticks(1:length(names));
            yticks(1:length(names));
            xticklabels(names);
            yticklabels(names);
            xtickangle(45);
            
            % 添加相关系数文本
            textColors = repmat([1, 1, 1], size(corrMatrix, 1) * size(corrMatrix, 2), 1);
            textColors(abs(corrMatrix(:)) < 0.5, :) = repmat([0, 0, 0], sum(abs(corrMatrix(:)) < 0.5), 1);
            
            for i = 1:size(corrMatrix, 1)
                for j = 1:size(corrMatrix, 2)
                    text(j, i, sprintf('%.2f', corrMatrix(i, j)), ...
                        'HorizontalAlignment', 'center', ...
                        'Color', textColors((i-1)*size(corrMatrix, 2) + j, :), ...
                        'FontWeight', 'bold');
                end
            end
            
            title(figTitle);
            axis square;
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '变量相关性热图';
            obj.figuresInfo(obj.figureCounter).filename = 'correlation_heatmap';
            
            obj.logger.info('相关性热图已生成');
        end
        
        function plotModelComparison(obj, modelNames, metrics, metricNames, figTitle)
            % 绘制模型比较图
            %
            % 参数:
            %   modelNames - 模型名称元胞数组
            %   metrics - 度量值矩阵，每行对应一个模型，每列对应一个度量
            %   metricNames - 度量名称元胞数组（可选）
            %   figTitle - 图形标题（可选）
            
            if nargin < 5 || isempty(figTitle)
                figTitle = '模型比较';
            end
            
            if nargin < 4 || isempty(metricNames)
                metricNames = arrayfun(@(i) sprintf('Metric %d', i), ...
                    1:size(metrics, 2), 'UniformOutput', false);
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 900, 700]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 规格化指标，使每列的最大值为1
            normMetrics = metrics;
            for i = 1:size(metrics, 2)
                if max(abs(metrics(:, i))) > 0
                    normMetrics(:, i) = metrics(:, i) / max(abs(metrics(:, i)));
                end
            end
            
            % 创建并排条形图
            bar(normMetrics, 'FaceColor', 'flat');
            
            % 设置每个条形的颜色
            colorMap = jet(size(metrics, 2));
            for i = 1:size(metrics, 2)
                set(findobj(gca, 'Type', 'patch'), 'FaceColor', 'flat', ...
                    'CData', 1:size(metrics, 2), 'EdgeColor', 'none');
            end
            
            % 设置坐标轴和标签
            xticks(1:length(modelNames));
            xticklabels(modelNames);
            xtickangle(30);
            
            grid on;
            xlabel('模型');
            ylabel('归一化指标值');
            title(figTitle);
            legend(metricNames, 'Location', 'eastoutside');
            
            % 添加原始值标签
            hold on;
            for i = 1:size(metrics, 1)
                for j = 1:size(metrics, 2)
                    if normMetrics(i, j) > 0.1  % 只在足够高的条形上显示标签
                        text(i, normMetrics(i, j) + 0.02, sprintf('%.3g', metrics(i, j)), ...
                            'HorizontalAlignment', 'center', 'FontSize', 8);
                    end
                end
            end
            hold off;
            
            % 调整Y轴范围以容纳标签
            ylim([0, 1.2]);
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '模型比较图';
            obj.figuresInfo(obj.figureCounter).filename = 'model_comparison';
            
            obj.logger.info('模型比较图已生成');
        end
        
        function plotTimeSeriesAnalysis(obj, observed, predicted, dates, figTitle)
            % 绘制时间序列分析图
            %
            % 参数:
            %   observed - 观测值向量
            %   predicted - 预测值向量
            %   dates - 日期向量（可选）
            %   figTitle - 图形标题（可选）
            
            if nargin < 5 || isempty(figTitle)
                figTitle = '时间序列分析';
            end
            
            if nargin < 4 || isempty(dates)
                dates = 1:length(observed);
            end
            
            % 创建图形
            fig = figure('Name', figTitle, 'Position', [100, 100, 900, 700]);
            obj.figureCounter = obj.figureCounter + 1;
            obj.figureHandles{obj.figureCounter} = fig;
            
            % 绘制时间序列
            subplot(2, 1, 1);
            plot(dates, observed, '-', 'Color', obj.colorScheme.primary, 'LineWidth', 2);
            hold on;
            plot(dates, predicted, '--', 'Color', obj.colorScheme.secondary, 'LineWidth', 2);
            hold off;
            
            grid on;
            xlabel('时间');
            ylabel('值');
            title('观测值 vs. 预测值');
            legend({'观测值', '预测值'}, 'Location', 'best');
            
            % 如果日期是datetime类型，设置适当的日期格式
            if isdatetime(dates)
                datetick('x', 'yyyy-mm-dd', 'keepticks');
                xtickangle(45);
            end
            
            % 绘制残差
            subplot(2, 1, 2);
            residuals = observed - predicted;
            plot(dates, residuals, '-', 'Color', obj.colorScheme.tertiary, 'LineWidth', 1.5);
            hold on;
            plot(dates, zeros(size(dates)), '--', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            
            % 添加残差包络线
            envWidth = 2 * std(residuals);
            plot(dates, repmat(envWidth, size(dates)), ':', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            plot(dates, repmat(-envWidth, size(dates)), ':', 'Color', obj.colorScheme.dark, 'LineWidth', 1);
            hold off;
            
            grid on;
            xlabel('时间');
            ylabel('残差');
            title('残差 (观测值 - 预测值)');
            
            % 如果日期是datetime类型，设置适当的日期格式
            if isdatetime(dates)
                datetick('x', 'yyyy-mm-dd', 'keepticks');
                xtickangle(45);
            end
            
            % 调整整体布局
            sgtitle(figTitle, 'FontSize', 14, 'FontWeight', 'bold');
            
            % 记录图形信息
            obj.figuresInfo(obj.figureCounter).title = figTitle;
            obj.figuresInfo(obj.figureCounter).description = '时间序列分析和残差图';
            obj.figuresInfo(obj.figureCounter).filename = 'time_series_analysis';
            
            obj.logger.info('时间序列分析图已生成');
        end
        
        function plotAllFigures(obj, modelResults)
            % 一次性绘制所有相关图形
            %
            % 参数:
            %   modelResults - 模型结果结构体
            
            if nargin < 2
                modelResults = obj.modelResults;
            else
                obj.modelResults = modelResults;
            end
            
            if isempty(modelResults)
                obj.logger.error('未设置模型结果，无法生成图形');
                return;
            end

             % 绘制系数估计图
            if isfield(modelResults, 'coefficients') && isfield(modelResults, 'standardErrors')
                obj.plotCoefficientEstimates(modelResults.coefficients, ...
                    modelResults.standardErrors, modelResults.variableNames, ...
                    '系数估计与置信区间');
            end
            
            % 绘制变量重要性图
            if isfield(modelResults, 'importance')
                obj.plotVariableImportance(modelResults.importance, ...
                    modelResults.variableNames, '变量重要性');
            elseif isfield(modelResults, 'standardizedCoefficients')
                obj.plotVariableImportance(abs(modelResults.standardizedCoefficients), ...
                    modelResults.variableNames, '变量重要性 (基于标准化系数)');
            end
            
            % 绘制模型拟合图
            if isfield(modelResults, 'observed') && isfield(modelResults, 'predicted')
                obj.plotModelFit(modelResults.observed, modelResults.predicted, ...
                    '模型拟合优度');
            end
            
            % 绘制残差诊断图
            if isfield(modelResults, 'residuals')
                if isfield(modelResults, 'leverage') && isfield(modelResults, 'cookDistance')
                    obj.plotResidualDiagnostics(modelResults.residuals, ...
                        modelResults.leverage, modelResults.cookDistance, '残差诊断');
                else
                    % 创建简单的杠杆值和Cook距离
                    n = length(modelResults.residuals);
                    p = length(modelResults.coefficients);
                    obj.plotResidualDiagnostics(modelResults.residuals, ...
                        ones(n, 1) * p/n, zeros(n, 1), '残差诊断 (简化版)');
                end
            end
            
            % 绘制相关性热图
            if isfield(modelResults, 'correlationMatrix')
                obj.plotCorrelationHeatmap(modelResults.correlationMatrix, ...
                    modelResults.variableNames, '变量相关性热图');
            end
            
            % 绘制时间序列分析图（如果适用）
            if isfield(modelResults, 'observed') && isfield(modelResults, 'predicted')
                if isfield(modelResults, 'dates')
                    obj.plotTimeSeriesAnalysis(modelResults.observed, modelResults.predicted, ...
                        modelResults.dates, '时间序列分析');
                else
                    obj.plotTimeSeriesAnalysis(modelResults.observed, modelResults.predicted, ...
                        [], '时间序列分析');
                end
            end
            
            obj.logger.info('已生成所有图形，共 %d 张', obj.figureCounter);
        end
        
        function saveAllFigures(obj, format, resolution)
            % 保存所有图形
            %
            % 参数:
            %   format - 保存格式，可选 'png', 'pdf', 'fig'等（默认 'png'）
            %   resolution - 分辨率（默认 300 dpi）
            
            if nargin < 2 || isempty(format)
                format = 'png';
            end
            
            if nargin < 3 || isempty(resolution)
                if obj.highDPI
                    resolution = 300;
                else
                    resolution = 150;
                end
            end
            
            if isempty(obj.figureHandles)
                obj.logger.warn('没有图形可以保存');
                return;
            end
            
            % 确保保存路径存在
            if ~exist(obj.savePath, 'dir')
                mkdir(obj.savePath);
                obj.logger.info('创建保存路径: %s', obj.savePath);
            end
            
            % 遍历所有图形进行保存
            for i = 1:length(obj.figureHandles)
                if isvalid(obj.figureHandles{i})
                    figure(obj.figureHandles{i});
                    
                    % 构建文件名
                    if isfield(obj.figuresInfo, 'filename') && length(obj.figuresInfo) >= i
                        baseName = obj.figuresInfo(i).filename;
                    else
                        baseName = sprintf('figure_%02d', i);
                    end
                    
                    % 添加时间戳（可选）
                    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                    fileName = fullfile(obj.savePath, [baseName, '_', timestamp, '.', format]);
                    
                    % 保存图形
                    switch lower(format)
                        case 'png'
                            print(obj.figureHandles{i}, fileName, '-dpng', ['-r', num2str(resolution)]);
                        case 'pdf'
                            print(obj.figureHandles{i}, fileName, '-dpdf', ['-r', num2str(resolution)]);
                        case 'jpg'
                            print(obj.figureHandles{i}, fileName, '-djpeg', ['-r', num2str(resolution)]);
                        case 'fig'
                            savefig(obj.figureHandles{i}, fileName);
                        case 'svg'
                            print(obj.figureHandles{i}, fileName, '-dsvg');
                        case 'eps'
                            print(obj.figureHandles{i}, fileName, '-depsc');
                        otherwise
                            print(obj.figureHandles{i}, fileName, '-dpng', ['-r', num2str(resolution)]);
                    end
                    
                    obj.logger.info('已保存图形: %s', fileName);
                end
            end
            
            obj.logger.info('所有图形已保存到: %s', obj.savePath);
        end
        
        function closeAllFigures(obj)
            % 关闭所有图形
            
            for i = 1:length(obj.figureHandles)
                if isvalid(obj.figureHandles{i})
                    close(obj.figureHandles{i});
                end
            end
            
            obj.figureHandles = {};
            obj.figureCounter = 0;
            obj.figuresInfo = struct('title', {}, 'description', {}, 'filename', {});
            
            obj.logger.info('所有图形已关闭');
        end
        
        function figureInfo = getFigureInfo(obj)
            % 获取图形信息
            %
            % 返回值:
            %   figureInfo - 包含图形信息的结构体数组
            
            figureInfo = obj.figuresInfo;
        end
        
        function setColorScheme(obj, scheme)
            % 设置颜色方案
            %
            % 参数:
            %   scheme - 颜色方案名称或结构体
            
            if isstruct(scheme)
                obj.colorScheme = scheme;
                obj.logger.info('已设置自定义颜色方案');
                return;
            end
            
            % 预定义的颜色方案
            switch lower(scheme)
                case 'default'
                    obj.colorScheme = struct(...
                        'primary', [0.2, 0.4, 0.6], ...
                        'secondary', [0.8, 0.3, 0.3], ...
                        'tertiary', [0.3, 0.7, 0.4], ...
                        'light', [0.8, 0.8, 0.8], ...
                        'dark', [0.2, 0.2, 0.2], ...
                        'highlight', [1.0, 0.5, 0.0], ...
                        'gradient', {{'#3288bd', '#99d594', '#e6f598', '#fee08b', '#fc8d59', '#d53e4f'}});
                case 'colorful'
                    obj.colorScheme = struct(...
                        'primary', [0.9, 0.3, 0.3], ...
                        'secondary', [0.3, 0.7, 0.3], ...
                        'tertiary', [0.3, 0.3, 0.9], ...
                        'light', [0.9, 0.9, 0.9], ...
                        'dark', [0.2, 0.2, 0.2], ...
                        'highlight', [1.0, 0.8, 0.2], ...
                        'gradient', {{'#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd'}});
                case 'monochrome'
                    obj.colorScheme = struct(...
                        'primary', [0.3, 0.3, 0.3], ...
                        'secondary', [0.5, 0.5, 0.5], ...
                        'tertiary', [0.7, 0.7, 0.7], ...
                        'light', [0.9, 0.9, 0.9], ...
                        'dark', [0.1, 0.1, 0.1], ...
                        'highlight', [0.0, 0.0, 0.0], ...
                        'gradient', {{'#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff'}});
                case 'pastel'
                    obj.colorScheme = struct(...
                        'primary', [0.7, 0.8, 0.9], ...
                        'secondary', [0.9, 0.8, 0.7], ...
                        'tertiary', [0.8, 0.9, 0.7], ...
                        'light', [0.95, 0.95, 0.95], ...
                        'dark', [0.3, 0.3, 0.3], ...
                        'highlight', [1.0, 0.7, 0.7], ...
                        'gradient', {{'#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462'}});
                otherwise
                    obj.logger.warn('未知的颜色方案: %s，使用默认方案', scheme);
                    obj.setColorScheme('default');
            end
            
            obj.logger.info('已设置颜色方案: %s', scheme);
        end
        
        function setHighDPI(obj, flag)
            % 设置是否使用高DPI输出
            %
            % 参数:
            %   flag - 布尔值，true表示使用高DPI输出
            
            obj.highDPI = logical(flag);
            obj.logger.info('高DPI输出设置为: %d', obj.highDPI);
        end
    end
    
    methods (Static)
        function colors = getDefaultColors(n)
            % 获取默认的n种颜色
            %
            % 参数:
            %   n - 颜色数量
            %
            % 返回值:
            %   colors - n×3的颜色矩阵
            
            % 默认颜色谱
            baseColors = [
                0.0000, 0.4470, 0.7410;  % 蓝色
                0.8500, 0.3250, 0.0980;  % 红色
                0.9290, 0.6940, 0.1250;  % 黄色
                0.4940, 0.1840, 0.5560;  % 紫色
                0.4660, 0.6740, 0.1880;  % 绿色
                0.3010, 0.7450, 0.9330;  % 浅蓝
                0.6350, 0.0780, 0.1840;  % 深红
                0.0000, 0.0000, 1.0000;  % 纯蓝
                1.0000, 0.0000, 0.0000;  % 纯红
                0.0000, 1.0000, 0.0000;  % 纯绿
                0.7500, 0.7500, 0.0000;  % 橄榄
                0.7500, 0.0000, 0.7500;  % 洋红
                0.0000, 0.7500, 0.7500;  % 青色
                0.7500, 0.7500, 0.7500;  % 灰色
            ];
            
            % 如果需要的颜色数量大于基础颜色集，则使用插值生成
            if n <= size(baseColors, 1)
                colors = baseColors(1:n, :);
            else
                % 使用HSV颜色空间均匀分布生成颜色
                colors = hsv(n);
            end
        end
        
        function saveFigureAsImage(fig, fileName, format, resolution)
            % 保存单个图形为图像文件
            %
            % 参数:
            %   fig - 图形句柄
            %   fileName - 文件名（不含扩展名）
            %   format - 保存格式（默认 'png'）
            %   resolution - 分辨率（默认 300 dpi）
            
            if nargin < 3 || isempty(format)
                format = 'png';
            end
            
            if nargin < 4 || isempty(resolution)
                resolution = 300;
            end
            
            % 构建完整文件名
            fullFileName = [fileName, '.', format];
            
            % 保存图形
            figure(fig);
            switch lower(format)
                case 'png'
                    print(fig, fullFileName, '-dpng', ['-r', num2str(resolution)]);
                case 'pdf'
                    print(fig, fullFileName, '-dpdf', ['-r', num2str(resolution)]);
                case 'jpg'
                    print(fig, fullFileName, '-djpeg', ['-r', num2str(resolution)]);
                case 'fig'
                    savefig(fig, fullFileName);
                case 'svg'
                    print(fig, fullFileName, '-dsvg');
                case 'eps'
                    print(fig, fullFileName, '-depsc');
                otherwise
                    print(fig, fullFileName, '-dpng', ['-r', num2str(resolution)]);
            end
        end
    end
end