classdef ParameterStatisticsCalculator < handle
    % ParameterStatisticsCalculator - 计算模型参数的统计信息
    %
    % 该类负责计算和分析模型参数的统计特性，包括标准误、
    % 置信区间、显著性检验、VIF等统计量，帮助评估模型质量。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   coefficients - 系数估计值
    %   standardErrors - 系数标准误
    %   tStats - t统计量
    %   pValues - p值
    %   confidenceIntervals - 置信区间
    %   vifValues - 方差膨胀因子
    %   isSignificant - 显著性标志
    
    properties
        logger               % 日志记录器
        coefficients         % 模型系数
        standardErrors       % 标准误
        tStats               % t统计量
        pValues              % p值
        confidenceIntervals  % 置信区间 [lower, upper]
        vifValues            % 方差膨胀因子
        isSignificant        % 显著性标志
        parameterNames       % 参数名称
        residuals            % 残差
        dof                  % 自由度
        sigmaSquared         % 误差方差估计
        covarianceMatrix     % 参数协方差矩阵
        correlationMatrix    % 参数相关系数矩阵
        anovaTable           % 方差分析表
        modelQuality         % 模型质量指标 (R^2, 调整R^2等)
    end
    
    methods
        function obj = ParameterStatisticsCalculator(logger)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('ParameterStatisticsCalculator');
            else
                obj.logger = logger;
            end
            
            obj.logger.info('参数统计分析模块已初始化');
        end
        
        function calculate(obj, X, y, coefficients, paramNames)
            % 计算模型参数的统计量
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   coefficients - 模型系数向量
            %   paramNames - 参数名称元胞数组(可选)
            
            [n, p] = size(X);
            obj.coefficients = coefficients(:);
            
            if nargin < 5 || isempty(paramNames)
                obj.parameterNames = arrayfun(@(i) sprintf('Param_%d', i), ...
                    1:length(coefficients), 'UniformOutput', false);
            else
                obj.parameterNames = paramNames;
            end
            
            % 计算拟合值和残差
            yHat = X * obj.coefficients;
            obj.residuals = y - yHat;
            
            % 自由度和方差估计
            obj.dof = n - p;
            obj.sigmaSquared = sum(obj.residuals.^2) / obj.dof;
            
            % 计算参数协方差矩阵
            try
                XtX_inv = inv(X' * X);
                obj.covarianceMatrix = obj.sigmaSquared * XtX_inv;
                
                % 计算标准误差
                obj.standardErrors = sqrt(diag(obj.covarianceMatrix));
                
                % 计算相关系数矩阵
                D = diag(1 ./ sqrt(diag(obj.covarianceMatrix)));
                obj.correlationMatrix = D * obj.covarianceMatrix * D;
            catch ME
                obj.logger.error('计算参数协方差矩阵失败: %s', ME.message);
                obj.standardErrors = nan(size(obj.coefficients));
                obj.covarianceMatrix = nan(length(obj.coefficients));
                obj.correlationMatrix = nan(length(obj.coefficients));
            end
            
            % 计算t统计量和p值
            obj.tStats = obj.coefficients ./ obj.standardErrors;
            obj.pValues = 2 * (1 - tcdf(abs(obj.tStats), obj.dof));
            
            % 计算95%置信区间
            tCritical = tinv(0.975, obj.dof); % 95%置信区间，双侧
            halfWidth = tCritical * obj.standardErrors;
            obj.confidenceIntervals = [obj.coefficients - halfWidth, obj.coefficients + halfWidth];
            
            % 判断显著性 (α = 0.05)
            obj.isSignificant = obj.pValues < 0.05;
            
            % 计算VIF (方差膨胀因子)
            obj.calculateVIFs(X);
            
            % 计算ANOVA表和模型质量指标
            obj.calculateModelQuality(X, y, yHat);
            
            obj.logger.info('参数统计量计算完成');
        end
        
        function calculateVIFs(obj, X)
            % 计算方差膨胀因子
            %
            % 参数:
            %   X - 自变量矩阵
            
            p = size(X, 2);
            obj.vifValues = nan(p, 1);
            
            % 如果只有一个变量，无需计算VIF
            if p <= 1
                obj.vifValues = 1;
                return;
            end
            
            try
                % 对每个变量，使用其他变量做回归并计算R^2
                for i = 1:p
                    otherCols = setdiff(1:p, i);
                    Xi = X(:, i);
                    X_others = X(:, otherCols);
                    
                    % 添加常数项
                    X_others = [ones(size(X_others, 1), 1), X_others];
                    
                    % 计算回归系数
                    b = (X_others' * X_others) \ (X_others' * Xi);
                    
                    % 计算R^2
                    y_pred = X_others * b;
                    SST = sum((Xi - mean(Xi)).^2);
                    SSE = sum((Xi - y_pred).^2);
                    R2 = 1 - SSE/SST;
                    
                    % 计算VIF
                    if R2 < 1 - eps
                        obj.vifValues(i) = 1 / (1 - R2);
                    else
                        obj.vifValues(i) = Inf;
                    end
                end
            catch ME
                obj.logger.error('计算VIF失败: %s', ME.message);
                obj.vifValues = nan(p, 1);
            end
        end
        
        function calculateModelQuality(obj, X, y, yHat)
            % 计算模型质量指标和ANOVA表
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   yHat - 拟合值
            
            n = length(y);
            p = size(X, 2);
            
            % 计算总平方和(SST)、回归平方和(SSR)和误差平方和(SSE)
            yMean = mean(y);
            SST = sum((y - yMean).^2);
            SSR = sum((yHat - yMean).^2);
            SSE = sum((y - yHat).^2);
            
            % 准备ANOVA表
            obj.anovaTable = struct();
            obj.anovaTable.Source = {'回归'; '残差'; '总计'};
            obj.anovaTable.SS = [SSR; SSE; SST];
            obj.anovaTable.DF = [p; n-p; n-1];
            obj.anovaTable.MS = [SSR/p; SSE/(n-p); NaN];
            obj.anovaTable.F = [obj.anovaTable.MS(1)/obj.anovaTable.MS(2); NaN; NaN];
            obj.anovaTable.pValue = [1 - fcdf(obj.anovaTable.F(1), p, n-p); NaN; NaN];
            
            % 计算模型质量指标
            obj.modelQuality = struct();
            obj.modelQuality.R2 = SSR / SST;
            obj.modelQuality.AdjustedR2 = 1 - (SSE/(n-p)) / (SST/(n-1));
            
            % 计算AIC和BIC
            logL = -n/2 * log(2*pi*obj.sigmaSquared) - SSE/(2*obj.sigmaSquared);
            obj.modelQuality.AIC = -2 * logL + 2 * p;
            obj.modelQuality.BIC = -2 * logL + p * log(n);
            
            % 均方误差和均方根误差
            obj.modelQuality.MSE = SSE / n;
            obj.modelQuality.RMSE = sqrt(obj.modelQuality.MSE);
            
            % 计算标准误差
            obj.modelQuality.StandardError = sqrt(SSE / (n - p));
            
            % F检验的p值
            obj.modelQuality.FPValue = obj.anovaTable.pValue(1);
            
            obj.logger.info('模型质量指标计算完成: R^2 = %.4f, 调整R^2 = %.4f', ...
                obj.modelQuality.R2, obj.modelQuality.AdjustedR2);
        end
        
        function report = generateSummaryReport(obj)
            % 生成参数统计摘要报告
            %
            % 返回值:
            %   report - 包含统计分析结果的结构体
            
            report = struct();
            
            % 参数估计结果表
            paramTable = table(obj.coefficients, obj.standardErrors, ...
                obj.tStats, obj.pValues, obj.vifValues, ...
                obj.confidenceIntervals(:,1), obj.confidenceIntervals(:,2), ...
                obj.isSignificant, ...
                'VariableNames', {'Estimate', 'StdError', 'tStat', 'pValue', ...
                'VIF', 'CI_Lower', 'CI_Upper', 'IsSignificant'});
            
            if ~isempty(obj.parameterNames)
                paramTable.Properties.RowNames = obj.parameterNames;
            end
            
            report.ParameterTable = paramTable;
            report.ModelQuality = obj.modelQuality;
            report.ANOVA = struct2table(obj.anovaTable);
            
            % 参数相关性
            report.ParameterCorrelation = array2table(obj.correlationMatrix, ...
                'RowNames', obj.parameterNames, ...
                'VariableNames', obj.parameterNames);
            
            % 参数协方差
            report.ParameterCovariance = array2table(obj.covarianceMatrix, ...
                'RowNames', obj.parameterNames, ...
                'VariableNames', obj.parameterNames);
            
            obj.logger.info('参数统计摘要报告已生成');
        end
        
        function plotParameterEstimates(obj)
            % 可视化参数估计和置信区间
            
            figure('Name', '参数估计及置信区间', 'Position', [100, 100, 900, 600]);
            
            numParams = length(obj.coefficients);
            
            % 系数估计和置信区间
            subplot(2, 2, 1);
            errorbar(1:numParams, obj.coefficients, ...
                obj.coefficients - obj.confidenceIntervals(:,1), ...
                obj.confidenceIntervals(:,2) - obj.coefficients, 'o');
            grid on;
            title('参数估计及95%置信区间');
            xlabel('参数');
            ylabel('估计值');
            
            if ~isempty(obj.parameterNames)
                xticks(1:numParams);
                xticklabels(obj.parameterNames);
                xtickangle(45);
            end
            
            sgtitle('参数统计分析');
            
            obj.logger.info('参数估计可视化图已生成');
        end
        
        function plotParameterCorrelations(obj)
            % 可视化参数相关性
            
            if isempty(obj.correlationMatrix)
                obj.logger.warn('参数相关矩阵为空，无法绘制相关性热图');
                return;
            end
            
            figure('Name', '参数相关性矩阵', 'Position', [100, 100, 800, 600]);
            
            % 使用imagesc绘制相关性热图
            imagesc(obj.correlationMatrix);
            colormap(jet);
            colorbar;
            
            % 在每个单元格中显示相关系数值
            [rows, cols] = size(obj.correlationMatrix);
            for i = 1:rows
                for j = 1:cols
                    text(j, i, sprintf('%.2f', obj.correlationMatrix(i, j)), ...
                        'HorizontalAlignment', 'center', ...
                        'Color', 'w');
                end
            end
            
            % 设置坐标轴
            if ~isempty(obj.parameterNames)
                xticks(1:length(obj.parameterNames));
                yticks(1:length(obj.parameterNames));
                xticklabels(obj.parameterNames);
                yticklabels(obj.parameterNames);
                xtickangle(45);
            end
            
            title('参数相关性矩阵');
            axis square;
            
            obj.logger.info('参数相关性热图已生成');
        end
        
        function compareMultipleModels(obj, modelResults)
            % 比较多个模型的参数估计
            %
            % 参数:
            %   modelResults - 包含多个模型结果的元胞数组或结构体数组
            
            if isempty(modelResults) || length(modelResults) < 2
                obj.logger.warn('需要至少两个模型结果才能进行比较');
                return;
            end
            
            numModels = length(modelResults);
            
            % 提取质量指标进行比较
            r2Values = zeros(numModels, 1);
            adjR2Values = zeros(numModels, 1);
            aicValues = zeros(numModels, 1);
            bicValues = zeros(numModels, 1);
            rmseValues = zeros(numModels, 1);
            
            for i = 1:numModels
                if isstruct(modelResults{i}) && isfield(modelResults{i}, 'modelQuality')
                    r2Values(i) = modelResults{i}.modelQuality.R2;
                    adjR2Values(i) = modelResults{i}.modelQuality.AdjustedR2;
                    aicValues(i) = modelResults{i}.modelQuality.AIC;
                    bicValues(i) = modelResults{i}.modelQuality.BIC;
                    rmseValues(i) = modelResults{i}.modelQuality.RMSE;
                else
                    obj.logger.warn('模型 %d 缺少必要的质量指标信息', i);
                end
            end
            
            % 创建比较图
            figure('Name', '模型比较', 'Position', [100, 100, 1000, 800]);
            
            % R^2 和调整R^2
            subplot(2, 3, 1);
            bar([r2Values, adjR2Values]);
            title('R^2 和调整R^2');
            xlabel('模型');
            ylabel('值');
            legend({'R^2', '调整R^2'}, 'Location', 'best');
            grid on;
            
            % AIC
            subplot(2, 3, 2);
            bar(aicValues);
            title('AIC (越小越好)');
            xlabel('模型');
            ylabel('AIC');
            grid on;
            
            % BIC
            subplot(2, 3, 3);
            bar(bicValues);
            title('BIC (越小越好)');
            xlabel('模型');
            ylabel('BIC');
            grid on;
            
            % RMSE
            subplot(2, 3, 4);
            bar(rmseValues);
            title('RMSE (越小越好)');
            xlabel('模型');
            ylabel('RMSE');
            grid on;
            
            % 模型参数数量
            paramCounts = zeros(numModels, 1);
            for i = 1:numModels
                if isstruct(modelResults{i}) && isfield(modelResults{i}, 'coefficients')
                    paramCounts(i) = length(modelResults{i}.coefficients);
                end
            end
            
            subplot(2, 3, 5);
            bar(paramCounts);
            title('参数数量');
            xlabel('模型');
            ylabel('参数个数');
            grid on;
            
            % 总体比较表格
            subplot(2, 3, 6);
            axis off;
            
            % 创建表格文本
            tableData = {
                '模型', 'R^2', '调整R^2', 'AIC', 'BIC', 'RMSE', '参数个数';
            };
            
            for i = 1:numModels
                tableData(i+1, :) = {
                    sprintf('模型 %d', i), ...
                    sprintf('%.4f', r2Values(i)), ...
                    sprintf('%.4f', adjR2Values(i)), ...
                    sprintf('%.1f', aicValues(i)), ...
                    sprintf('%.1f', bicValues(i)), ...
                    sprintf('%.4f', rmseValues(i)), ...
                    sprintf('%d', paramCounts(i))
                };
            end
            
            % 找出每列最佳值（除了第一列和最后一列）
            bestIndices = zeros(1, 5);
            bestIndices(1) = find(r2Values == max(r2Values), 1);  % R^2 最大
            bestIndices(2) = find(adjR2Values == max(adjR2Values), 1);  % 调整R^2 最大
            bestIndices(3) = find(aicValues == min(aicValues), 1);  % AIC 最小
            bestIndices(4) = find(bicValues == min(bicValues), 1);  % BIC 最小
            bestIndices(5) = find(rmseValues == min(rmseValues), 1);  % RMSE 最小
            
            % 高亮显示最佳值
            text(0.1, 0.9, '最佳模型比较:', 'FontWeight', 'bold');
            for i = 1:5
                bestModelIdx = bestIndices(i);
                text(0.1, 0.9 - 0.1*i, sprintf('%s: 模型 %d (%.4f)', ...
                    tableData{1, i+1}, bestModelIdx, ...
                    eval([lower(tableData{1, i+1}) 'Values(bestModelIdx)'])));
            end
            
            sgtitle('多模型比较分析');
            
            obj.logger.info('多模型比较分析图已生成');
        end
    end
    
    methods (Static)
        function result = runWaldTest(beta, covMatrix, constraintMatrix, constraintValues)
            % 执行Wald检验
            %
            % 参数:
            %   beta - 系数向量
            %   covMatrix - 协方差矩阵
            %   constraintMatrix - 约束矩阵 R
            %   constraintValues - 约束向量 r，检验假设 R*beta = r
            %
            % 返回值:
            %   result - 包含检验结果的结构体
            
            % 检查输入
            if nargin < 4
                constraintValues = zeros(size(constraintMatrix, 1), 1);
            end
            
            % 计算Wald统计量
            R = constraintMatrix;
            r = constraintValues;
            
            q = size(R, 1);  % 约束个数
            
            diff = R * beta - r;
            W = diff' * inv(R * covMatrix * R') * diff;
            
            % 在约束下，Wald统计量服从自由度为q的卡方分布
            pValue = 1 - chi2cdf(W, q);
            
            % 返回结果
            result = struct();
            result.WaldStat = W;
            result.df = q;
            result.pValue = pValue;
            result.isSignificant = (pValue < 0.05);
        end
        
        function result = runLikelihoodRatioTest(logL_unrestricted, logL_restricted, df_diff)
            % 执行似然比检验
            %
            % 参数:
            %   logL_unrestricted - 无约束模型的对数似然
            %   logL_restricted - 有约束模型的对数似然
            %   df_diff - 两个模型的自由度差异(参数个数差异)
            %
            % 返回值:
            %   result - 包含检验结果的结构体
            
            % 计算似然比统计量
            LR = -2 * (logL_restricted - logL_unrestricted);
            
            % 在约束下，LR统计量服从自由度为df_diff的卡方分布
            pValue = 1 - chi2cdf(LR, df_diff);
            
            % 返回结果
            result = struct();
            result.LRStat = LR;
            result.df = df_diff;
            result.pValue = pValue;
            result.isSignificant = (pValue < 0.05);
        end
        
        function [beta, se, tStat, pValue, ci] = robustRegression(X, y, method)
            % 执行稳健回归，处理异常值和异方差问题
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   method - 稳健方法: 'huber', 'bisquare', 'andrews', 等
            %
            % 返回值:
            %   beta - 稳健回归系数
            %   se - 稳健标准误
            %   tStat - t统计量
            %   pValue - p值
            %   ci - 置信区间 [lower, upper]
            
            if nargin < 3 || isempty(method)
                method = 'huber';
            end
            
            % 使用MATLAB内置的robustfit函数
            [beta, stats] = robustfit(X, y, method);
            
            % 提取结果
            se = stats.se;
            tStat = stats.t;
            pValue = stats.p;
            
            % 计算95%置信区间
            dof = stats.dfe;  % 误差自由度
            tCritical = tinv(0.975, dof);
            ci = [beta - tCritical * se, beta + tCritical * se];
        end
    end
end(45);
            end
            
            % 显著性 (-log10(p值))
            subplot(2, 2, 2);
            logp = -log10(obj.pValues);
            bar(logp);
            hold on;
            % 添加显著性线 (p=0.05, -log10(0.05)≈1.3)
            yline(1.3, '--r', 'p=0.05');
            hold off;
            title('参数显著性 (-log10(p值))');
            xlabel('参数');
            ylabel('-log10(p值)');
            
            if ~isempty(obj.parameterNames)
                xticks(1:numParams);
                xticklabels(obj.parameterNames);
                xtickangle(45);
            end
            
            % VIF值
            subplot(2, 2, 3);
            bar(obj.vifValues);
            hold on;
            % 添加VIF警戒线 (VIF=5和VIF=10)
            yline(5, '--g', 'VIF=5');
            yline(10, '--r', 'VIF=10');
            hold off;
            title('方差膨胀因子 (VIF)');
            xlabel('参数');
            ylabel('VIF值');
            
            if ~isempty(obj.parameterNames)
                xticks(1:numParams);
                xticklabels(obj.parameterNames);
                xtickangle(45);
            end
            
            % 标准化估计值
            subplot(2, 2, 4);
            stdCoefs = obj.coefficients ./ max(abs(obj.coefficients)) * 100;
            bar(stdCoefs);
            title('标准化参数估计 (% of max)');
            xlabel('参数');
            ylabel('标准化估计值 (%)');
            
            if ~isempty(obj.parameterNames)
                xticks(1:numParams);
                xticklabels(obj.parameterNames);
                xtickangle