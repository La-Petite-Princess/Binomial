classdef ResidualAnalyzer < handle
    % ResidualAnalyzer - 模型残差分析
    %
    % 该类负责分析回归模型残差的各种特性，检测异常点、
    % 评估模型假设是否成立，并提供诊断图和统计量。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   residuals - 原始残差
    %   standardizedResiduals - 标准化残差
    %   studentizedResiduals - 学生化残差
    %   cookDistance - Cook距离
    %   leverage - 杠杆值
    %   dffits - DFFITS统计量
    %   dfbetas - DFBETAS统计量
    %   normalityTests - 正态性检验结果
    %   autocorrelationTests - 自相关检验结果
    %   heteroskedasticityTests - 异方差检验结果
    
    properties
        logger                   % 日志记录器
        residuals                % 原始残差
        standardizedResiduals    % 标准化残差
        studentizedResiduals     % 学生化残差
        cookDistance             % Cook距离
        leverage                 % 杠杆值
        dffits                   % DFFITS统计量
        dfbetas                  % DFBETAS统计量
        hatMatrix                % 帽子矩阵
        fittedValues             % 拟合值
        X                        % 自变量矩阵
        y                        % 因变量向量
        coefficients             % 模型系数
        normalityTests           % 正态性检验结果
        autocorrelationTests     % 自相关检验结果
        heteroskedasticityTests  % 异方差检验结果
        outliers                 % 异常值指标
    end
    
    methods
        function obj = ResidualAnalyzer(logger)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('ResidualAnalyzer');
            else
                obj.logger = logger;
            end
            
            obj.logger.info('残差分析模块已初始化');
        end
        
        function analyze(obj, X, y, coefficients, residuals)
            % 分析残差
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   coefficients - 模型系数
            %   residuals - 残差（可选，如不提供则计算）
            
            obj.X = X;
            obj.y = y;
            obj.coefficients = coefficients;
            
            % 如果未提供残差，则计算
            if nargin < 5 || isempty(residuals)
                obj.fittedValues = X * coefficients;
                obj.residuals = y - obj.fittedValues;
            else
                obj.residuals = residuals;
                obj.fittedValues = y - residuals;
            end
            
            % 计算各种残差指标
            obj.computeResidualDiagnostics();
            
            % 进行残差相关检验
            obj.testResidualAssumptions();
            
            % 检测异常值
            obj.detectOutliers();
            
            obj.logger.info('残差分析完成');
        end
        
        function computeResidualDiagnostics(obj)
            % 计算各种残差诊断指标
            
            [n, p] = size(obj.X);
            
            try
                % 计算帽子矩阵 H = X(X'X)^(-1)X'
                obj.hatMatrix = obj.X * inv(obj.X' * obj.X) * obj.X';
                
                % 计算杠杆值（帽子矩阵对角线元素）
                obj.leverage = diag(obj.hatMatrix);
                
                % 计算残差方差
                sigmaSquared = sum(obj.residuals.^2) / (n - p);
                
                % 计算标准化残差
                obj.standardizedResiduals = obj.residuals ./ sqrt(sigmaSquared * (1 - obj.leverage));
                
                % 计算学生化残差
                obj.studentizedResiduals = zeros(n, 1);
                for i = 1:n
                    % 删除第i个观测值后的残差方差估计
                    sigma_i_sq = ((n-p)*sigmaSquared - obj.residuals(i)^2/(1-obj.leverage(i))) / (n-p-1);
                    
                    % 学生化残差
                    obj.studentizedResiduals(i) = obj.residuals(i) / sqrt(sigma_i_sq * (1 - obj.leverage(i)));
                end
                
                % 计算Cook距离
                obj.cookDistance = (obj.standardizedResiduals.^2 .* obj.leverage) / (p * (1 - obj.leverage));
                
                % 计算DFFITS
                obj.dffits = obj.studentizedResiduals .* sqrt(obj.leverage ./ (1 - obj.leverage));
                
                % 计算DFBETAS
                obj.dfbetas = zeros(n, p);
                X_with_intercept = [ones(n, 1), obj.X];
                
                for i = 1:n
                    % 计算影响矩阵
                    c_i = obj.leverage(i) / (1 - obj.leverage(i));
                    
                    % 计算DFBETAS
                    dfbeta_i = (X_with_intercept' * X_with_intercept)^(-1) * X_with_intercept(i, :)' * obj.studentizedResiduals(i) * sqrt(c_i);
                    obj.dfbetas(i, :) = dfbeta_i';
                end
            catch ME
                obj.logger.error('计算残差诊断指标失败: %s', ME.message);
            end
        end
        
        function testResidualAssumptions(obj)
            % 检验残差的各种假设
            
            % 检验残差正态性
            obj.testNormality();
            
            % 检验残差自相关性
            obj.testAutocorrelation();
            
            % 检验残差异方差性
            obj.testHeteroskedasticity();
        end
        
        function testNormality(obj)
            % 检验残差正态性
            
            obj.normalityTests = struct();
            
            try
                % Jarque-Bera 检验
                [jbH, jbPValue, jbStat] = obj.jarqueBera(obj.standardizedResiduals);
                obj.normalityTests.jarqueBera = struct(...
                    'statistic', jbStat, ...
                    'pValue', jbPValue, ...
                    'isNormal', ~jbH);
                
                % Shapiro-Wilk 检验
                [swH, swPValue, swStat] = obj.shapiroWilk(obj.standardizedResiduals);
                obj.normalityTests.shapiroWilk = struct(...
                    'statistic', swStat, ...
                    'pValue', swPValue, ...
                    'isNormal', ~swH);
                
                % Anderson-Darling 检验
                [adH, adPValue, adStat] = obj.andersonDarling(obj.standardizedResiduals);
                obj.normalityTests.andersonDarling = struct(...
                    'statistic', adStat, ...
                    'pValue', adPValue, ...
                    'isNormal', ~adH);
                
                % 计算偏度和峰度
                obj.normalityTests.skewness = skewness(obj.standardizedResiduals);
                obj.normalityTests.kurtosis = kurtosis(obj.standardizedResiduals) - 3; % 超额峰度
                
                obj.logger.info(['正态性检验完成: JB p值=%.4f, SW p值=%.4f, AD p值=%.4f, ', ...
                    '偏度=%.4f, 超额峰度=%.4f'], ...
                    jbPValue, swPValue, adPValue, ...
                    obj.normalityTests.skewness, obj.normalityTests.kurtosis);
            catch ME
                obj.logger.error('残差正态性检验失败: %s', ME.message);
            end
        end
        
        function testAutocorrelation(obj)
            % 检验残差自相关性
            
            obj.autocorrelationTests = struct();
            
            try
                % Durbin-Watson 检验
                [dwStat, dwPValue] = obj.durbinWatson(obj.residuals);
                obj.autocorrelationTests.durbinWatson = struct(...
                    'statistic', dwStat, ...
                    'pValue', dwPValue, ...
                    'hasAutocorrelation', dwStat < 1.5 || dwStat > 2.5);
                
                % Breusch-Godfrey 检验
                [bgH, bgPValue, bgStat, bgCValue] = obj.breuschGodfrey(obj.residuals, obj.X, 2);
                obj.autocorrelationTests.breuschGodfrey = struct(...
                    'statistic', bgStat, ...
                    'criticalValue', bgCValue, ...
                    'pValue', bgPValue, ...
                    'hasAutocorrelation', bgH);
                
                % 计算自相关系数（滞后1-5）
                maxLag = min(5, length(obj.residuals) - 1);
                [acf, lags, bounds] = autocorr(obj.residuals, maxLag);
                obj.autocorrelationTests.autocorrelation = struct(...
                    'acf', acf(2:end), ...
                    'lags', lags(2:end), ...
                    'bounds', bounds);
                
                obj.logger.info(['自相关检验完成: DW=%.4f, BG p值=%.4f, ', ...
                    'ACF(1)=%.4f'], ...
                    dwStat, bgPValue, acf(2));
            catch ME
                obj.logger.error('残差自相关检验失败: %s', ME.message);
            end
        end
        
        function testHeteroskedasticity(obj)
            % 检验残差异方差性
            
            obj.heteroskedasticityTests = struct();
            
            try
                % White 检验
                [whiteH, whitePValue, whiteStat] = obj.whiteTest(obj.residuals, obj.X);
                obj.heteroskedasticityTests.white = struct(...
                    'statistic', whiteStat, ...
                    'pValue', whitePValue, ...
                    'hasHeteroskedasticity', whiteH);
                
                % Breusch-Pagan 检验
                [bpH, bpPValue, bpStat] = obj.breuschPagan(obj.residuals, obj.X);
                obj.heteroskedasticityTests.breuschPagan = struct(...
                    'statistic', bpStat, ...
                    'pValue', bpPValue, ...
                    'hasHeteroskedasticity', bpH);
                
                % 计算残差与拟合值的相关系数
                absResid = abs(obj.residuals);
                corrCoef = corr(absResid, obj.fittedValues);
                obj.heteroskedasticityTests.residFittedCorr = corrCoef;
                
                obj.logger.info(['异方差检验完成: White p值=%.4f, BP p值=%.4f, ', ...
                    '残差-拟合值相关系数=%.4f'], ...
                    whitePValue, bpPValue, corrCoef);
            catch ME
                obj.logger.error('残差异方差检验失败: %s', ME.message);
            end
        end
        
        function detectOutliers(obj)
            % 检测异常点
            
            n = length(obj.residuals);
            p = size(obj.X, 2);
            
            obj.outliers = struct();
            
            % 根据标准化残差识别异常点（|z| > 2.5）
            obj.outliers.byStandardizedResiduals = find(abs(obj.standardizedResiduals) > 2.5);
            
            % 根据学生化残差识别异常点（|t| > tinv(0.975, n-p-1)）
            tCritical = tinv(0.975, n-p-1);
            obj.outliers.byStudentizedResiduals = find(abs(obj.studentizedResiduals) > tCritical);
            
            % 根据Cook距离识别异常点（D > 4/(n-p)）
            cookThreshold = 4/(n-p);
            obj.outliers.byCookDistance = find(obj.cookDistance > cookThreshold);
            
            % 根据杠杆值识别异常点（h > 2*p/n）
            leverageThreshold = 2*p/n;
            obj.outliers.byLeverage = find(obj.leverage > leverageThreshold);
            
            % 根据DFFITS统计量识别异常点（|DFFITS| > 2*sqrt(p/n)）
            dffitsThreshold = 2*sqrt(p/n);
            obj.outliers.byDffits = find(abs(obj.dffits) > dffitsThreshold);
            
            % 综合判定
            obj.outliers.consensus = [];
            for i = 1:n
                % 如果一个观测值至少在两个指标上被判定为异常点
                count = nnz([
                    ismember(i, obj.outliers.byStandardizedResiduals),
                    ismember(i, obj.outliers.byStudentizedResiduals),
                    ismember(i, obj.outliers.byCookDistance),
                    ismember(i, obj.outliers.byLeverage),
                    ismember(i, obj.outliers.byDffits)
                ]);
                
                if count >= 2
                    obj.outliers.consensus = [obj.outliers.consensus; i];
                end
            end
            
            obj.logger.info('异常点检测完成，发现 %d 个共识异常点', length(obj.outliers.consensus));
        end
        
        function plotResidualDiagnostics(obj)
            % 绘制残差诊断图
            
            figure('Name', '残差诊断图', 'Position', [100, 100, 1000, 800]);
            
            % 残差散点图
            subplot(2, 2, 1);
            plot(obj.fittedValues, obj.standardizedResiduals, 'o');
            hold on;
            plot([min(obj.fittedValues), max(obj.fittedValues)], [0, 0], 'k--');
            % 添加±2.5区域线
            plot([min(obj.fittedValues), max(obj.fittedValues)], [2.5, 2.5], 'r--');
            plot([min(obj.fittedValues), max(obj.fittedValues)], [-2.5, -2.5], 'r--');
            hold off;
            title('标准化残差 vs. 拟合值');
            xlabel('拟合值');
            ylabel('标准化残差');
            grid on;
            
            % 标记异常点
            if ~isempty(obj.outliers.consensus)
                hold on;
                plot(obj.fittedValues(obj.outliers.consensus), ...
                    obj.standardizedResiduals(obj.outliers.consensus), 'ro', ...
                    'MarkerFaceColor', 'r');
                hold off;
            end
            
            % QQ图
            subplot(2, 2, 2);
            qqplot(obj.standardizedResiduals);
            title('残差QQ图');
            grid on;
            
            % 杠杆值图
            subplot(2, 2, 3);
            stem(obj.leverage, 'filled');
            hold on;
            n = length(obj.leverage);
            p = size(obj.X, 2);
            leverageThreshold = 2*p/n;
            plot([1, n], [leverageThreshold, leverageThreshold], 'r--');
            hold off;
            title('杠杆值');
            xlabel('观测值索引');
            ylabel('杠杆值');
            grid on;
            
            % Cook距离图
            subplot(2, 2, 4);
            stem(obj.cookDistance, 'filled');
            hold on;
            cookThreshold = 4/(n-p);
            plot([1, n], [cookThreshold, cookThreshold], 'r--');
            hold off;
            title('Cook距离');
            xlabel('观测值索引');
            ylabel('Cook距离');
            grid on;
            
            sgtitle('残差诊断图');
            
            obj.logger.info('残差诊断图已生成');
        end
        
        function plotAdvancedDiagnostics(obj)
            % 绘制高级诊断图
            
            figure('Name', '高级残差诊断图', 'Position', [100, 100, 1000, 800]);
            
            % 学生化残差图
            subplot(2, 2, 1);
            stem(obj.studentizedResiduals, 'filled');
            hold on;
            n = length(obj.studentizedResiduals);
            p = size(obj.X, 2);
            tCritical = tinv(0.975, n-p-1);
            plot([1, n], [tCritical, tCritical], 'r--');
            plot([1, n], [-tCritical, -tCritical], 'r--');
            hold off;
            title('学生化残差');
            xlabel('观测值索引');
            ylabel('学生化残差');
            grid on;
            
            % DFFITS图
            subplot(2, 2, 2);
            stem(obj.dffits, 'filled');
            hold on;
            dffitsThreshold = 2*sqrt(p/n);
            plot([1, n], [dffitsThreshold, dffitsThreshold], 'r--');
            plot([1, n], [-dffitsThreshold, -dffitsThreshold], 'r--');
            hold off;
            title('DFFITS');
            xlabel('观测值索引');
            ylabel('DFFITS');
            grid on;
            
            % 残差自相关图
            subplot(2, 2, 3);
            try
                maxLag = min(20, length(obj.residuals) - 1);
                autocorr(obj.residuals, maxLag);
                title('残差自相关函数');
            catch
                text(0.5, 0.5, '无法计算自相关函数', 'HorizontalAlignment', 'center');
            end
            
            % 残差部分自相关图
            subplot(2, 2, 4);
            try
                maxLag = min(20, length(obj.residuals) - 1);
                parcorr(obj.residuals, maxLag);
                title('残差偏自相关函数');
            catch
                text(0.5, 0.5, '无法计算偏自相关函数', 'HorizontalAlignment', 'center');
            end
            
            sgtitle('高级残差诊断');
            
            obj.logger.info('高级残差诊断图已生成');
        end
        
        function report = generateResidualReport(obj)
            % 生成残差分析报告
            %
            % 返回值:
            %   report - 包含残差分析结果的结构体
            
            report = struct();
            
            % 残差分布统计量
            report.residualStats = struct(...
                'mean', mean(obj.residuals), ...
                'std', std(obj.residuals), ...
                'min', min(obj.residuals), ...
                'max', max(obj.residuals), ...
                'skewness', skewness(obj.residuals), ...
                'kurtosis', kurtosis(obj.residuals) - 3 ...
            );
            
            % 各种检验结果
            report.normalityTests = obj.normalityTests;
            report.autocorrelationTests = obj.autocorrelationTests;
            report.heteroskedasticityTests = obj.heteroskedasticityTests;
            
            % 异常点信息
            report.outliers = obj.outliers;
            
            % 异常点详细信息（如果有）
            if ~isempty(obj.outliers.consensus)
                outlierDetails = table();
                outlierDetails.Index = obj.outliers.consensus;
                outlierDetails.StandardizedResidual = obj.standardizedResiduals(obj.outliers.consensus);
                outlierDetails.StudentizedResidual = obj.studentizedResiduals(obj.outliers.consensus);
                outlierDetails.CookDistance = obj.cookDistance(obj.outliers.consensus);
                outlierDetails.Leverage = obj.leverage(obj.outliers.consensus);
                outlierDetails.DFFITS = obj.dffits(obj.outliers.consensus);
                report.outlierDetails = outlierDetails;
            end
            
            % 模型诊断总结
            report.diagnosticSummary = struct();
            
            % 评估残差正态性
            if isfield(obj.normalityTests, 'jarqueBera') && isfield(obj.normalityTests.jarqueBera, 'pValue')
                report.diagnosticSummary.isNormal = obj.normalityTests.jarqueBera.pValue > 0.05;
            else
                report.diagnosticSummary.isNormal = abs(report.residualStats.skewness) < 0.5 && ...
                    abs(report.residualStats.kurtosis) < 1;
            end
            
            % 评估残差自相关性
            if isfield(obj.autocorrelationTests, 'durbinWatson') && isfield(obj.autocorrelationTests.durbinWatson, 'statistic')
                dw = obj.autocorrelationTests.durbinWatson.statistic;
                report.diagnosticSummary.hasAutocorrelation = dw < 1.5 || dw > 2.5;
            else
                report.diagnosticSummary.hasAutocorrelation = false;
            end
            
            % 评估残差异方差性
            if isfield(obj.heteroskedasticityTests, 'breuschPagan') && isfield(obj.heteroskedasticityTests.breuschPagan, 'pValue')
                report.diagnosticSummary.hasHeteroskedasticity = obj.heteroskedasticityTests.breuschPagan.pValue < 0.05;
            else
                report.diagnosticSummary.hasHeteroskedasticity = abs(report.heteroskedasticityTests.residFittedCorr) > 0.3;
            end
            
            % 评估异常点
            report.diagnosticSummary.hasOutliers = ~isempty(obj.outliers.consensus);
            report.diagnosticSummary.outlierPercentage = 100 * length(obj.outliers.consensus) / length(obj.residuals);
            
            % 总体评估
            report.diagnosticSummary.issuesCount = nnz([
                ~report.diagnosticSummary.isNormal,
                report.diagnosticSummary.hasAutocorrelation,
                report.diagnosticSummary.hasHeteroskedasticity,
                report.diagnosticSummary.outlierPercentage > 5
            ]);
            
            if report.diagnosticSummary.issuesCount == 0
                report.diagnosticSummary.overallAssessment = '模型残差符合良好的回归假设';
            elseif report.diagnosticSummary.issuesCount == 1
                report.diagnosticSummary.overallAssessment = '模型残差存在轻微问题，但整体可接受';
            elseif report.diagnosticSummary.issuesCount == 2
                report.diagnosticSummary.overallAssessment = '模型残差存在中等问题，建议考虑改进模型';
            else
                report.diagnosticSummary.overallAssessment = '模型残差存在严重问题，需要修改模型或考虑稳健回归方法';
            end
            
            obj.logger.info('残差分析报告已生成');
        end
    end
    
    methods (Static)
        function [h, pValue, stat] = jarqueBera(x)
            % Jarque-Bera 正态性检验
            %
            % 参数:
            %   x - 数据向量
            %
            % 返回值:
            %   h - 假设检验结果（0表示不拒绝正态性假设）
            %   pValue - p值
            %   stat - 检验统计量
            
            n = length(x);
            
            % 标准化数据
            x = (x - mean(x)) / std(x);
            
            % 计算偏度和超额峰度
            s = skewness(x);
            k = kurtosis(x) - 3;
            
            % 计算JB统计量
            stat = n/6 * (s^2 + k^2/4);
            
            % 计算p值（JB服从自由度为2的卡方分布）
            pValue = 1 - chi2cdf(stat, 2);
            
            % 确定假设检验结果（α = 0.05）
            h = (pValue < 0.05);
        end
        
        function [h, pValue, stat] = shapiroWilk(x)
            % Shapiro-Wilk 正态性检验
            %
            % 参数:
            %   x - 数据向量
            %
            % 返回值:
            %   h - 假设检验结果（0表示不拒绝正态性假设）
            %   pValue - p值
            %   stat - 检验统计量
            
            % 注意：MATLAB没有内置的Shapiro-Wilk检验
            % 这里是一个简化实现
            
            n = length(x);
            
            % 对数据排序
            y = sort(x);
            
            % 标准化
            y = (y - mean(y)) / std(y);
            
            % 计算权重（这是一个简化版本，精确的权重计算更复杂）
            i = 1:floor(n/2);
            a = zeros(n, 1);
            a(i) = -1 ./ sqrt(n * (n-1) / 2);
            a(n-i+1) = -a(i);
            
            % 计算W统计量
            W = (sum(a .* y))^2 / sum((y - mean(y)).^2);
            
            % 近似p值（这是一个简化版本）
            if n <= 100
                % 简化的p值计算
                mu = 0.0038915 * n^3 + 0.0678034 * n^2 - 0.5912929 * n - 1.5857152;
                sigma = exp(1.0348433 * log(n) - 2.7756821);
                pValue = 1 - normcdf((log(1-W) - mu) / sigma);
            else
                % 大样本近似
                r = log(n);
                u = log(n) - log(1-W);
                mu = -1.5861 - 0.31082*r - 0.083751*r^2 + 0.0038915*r^3;
                sigma = exp(-0.4803 - 0.082676*r + 0.0030302*r^2);
                pValue = 1 - normcdf((u - mu) / sigma);
            end
            
            stat = W;
            h = (pValue < 0.05);
        end
        
        function [h, pValue, stat] = andersonDarling(x)
            % Anderson-Darling 正态性检验
            %
            % 参数:
            %   x - 数据向量
            %
            % 返回值:
            %   h - 假设检验结果（0表示不拒绝正态性假设）
            %   pValue - p值
            %   stat - 检验统计量
            
            n = length(x);
            
            % 标准化数据
            x = sort((x - mean(x)) / std(x));
            
            % 计算累积分布函数
            cdf = normcdf(x);
            
            % 避免0和1
            cdf(cdf < eps) = eps;
            cdf(cdf > 1-eps) = 1-eps;
            
            % 计算A^2统计量
            A2 = -n - (1/n) * sum((2*(1:n) - 1) .* (log(cdf) + log(1 - cdf(n:-1:1))));
            
            % 修正A^2
            A2_star = A2 * (1 + 0.75/n + 2.25/n^2);
            
            % 计算p值
            if A2_star < 0.2
                pValue = 1 - exp(-13.436 + 101.14*A2_star - 223.73*A2_star^2);
            elseif A2_star < 0.34
                pValue = 1 - exp(-8.318 + 42.796*A2_star - 59.938*A2_star^2);
            elseif A2_star < 0.6
                pValue = exp(0.9177 - 4.279*A2_star - 1.38*A2_star^2);
            elseif A2_star < 10
                pValue = exp(1.2937 - 5.709*A2_star + 0.0186*A2_star^2);
            else
                pValue = 3.7e-24;
            end
            
            stat = A2_star;
            h = (pValue < 0.05);
        end
        
        function [stat, pValue] = durbinWatson(resid)
            % Durbin-Watson 自相关检验
            %
            % 参数:
            %   resid - 残差向量
            %
            % 返回值:
            %   stat - DW统计量
            %   pValue - 近似p值
            
            n = length(resid);
            
            % 计算DW统计量
            diff_resid = diff(resid);
            stat = sum(diff_resid.^2) / sum(resid.^2);
            
            % 近似p值（注意：精确p值需查表或更复杂的计算）
            % DW接近2表示无自相关，接近0表示正自相关，接近4表示负自相关
            if stat < 1.5 || stat > 2.5
                pValue = 0.01;  % 强烈拒绝无自相关假设
            elseif stat < 1.7 || stat > 2.3
                pValue = 0.05;  % 拒绝无自相关假设
            else
                pValue = 0.2;   % 不拒绝无自相关假设
            end
        end
        
        function [h, pValue, stat, cValue] = breuschGodfrey(resid, X, lags)
            % Breusch-Godfrey 自相关检验
            %
            % 参数:
            %   resid - 残差向量
            %   X - 自变量矩阵
            %   lags - 滞后阶数
            %
            % 返回值:
            %   h - 假设检验结果（1表示拒绝无自相关假设）
            %   pValue - p值
            %   stat - 检验统计量
            %   cValue - 临界值
            
            n = length(resid);
            X = [ones(n, 1), X];
            k = size(X, 2);
            
            % 构建滞后残差矩阵
            Z = zeros(n, lags);
            for i = 1:lags
                Z(i+1:n, i) = resid(1:n-i);
            end
            
            % 丢弃缺失值
            validIdx = (lags+1):n;
            X_valid = X(validIdx, :);
            Z_valid = Z(validIdx, :);
            resid_valid = resid(validIdx);
            
            % 辅助回归
            XZ = [X_valid, Z_valid];
            b = (XZ' * XZ) \ (XZ' * resid_valid);
            
            % 计算R^2
            e = resid_valid - XZ * b;
            SST = sum((resid_valid - mean(resid_valid)).^2);
            SSE = sum(e.^2);
            R2 = 1 - SSE/SST;
            
            % 计算LM统计量
            stat = (n - lags) * R2;
            
            % 计算p值（LM服从自由度为lags的卡方分布）
            pValue = 1 - chi2cdf(stat, lags);
            
            % 临界值
            cValue = chi2inv(0.95, lags);
            
            % 确定假设检验结果（α = 0.05）
            h = (pValue < 0.05);
        end
        
        function [h, pValue, stat] = whiteTest(resid, X)
            % White 异方差检验
            %
            % 参数:
            %   resid - 残差向量
            %   X - 自变量矩阵
            %
            % 返回值:
            %   h - 假设检验结果（1表示拒绝同方差假设）
            %   pValue - p值
            %   stat - 检验统计量
            
            n = length(resid);
            X = [ones(n, 1), X];
            k = size(X, 2);
            
            % 构建交叉项矩阵
            Z = X;
            p = 1;
            
            % 添加平方项
            for i = 2:k
                Z = [Z, X(:, i).^2];
                p = p + 1;
            end
            
            % 添加交叉项
            for i = 2:k
                for j = i+1:k
                    Z = [Z, X(:, i) .* X(:, j)];
                    p = p + 1;
                end
            end
            
            % 辅助回归
            e2 = resid.^2;
            b = (Z' * Z) \ (Z' * e2);
            
            % 计算R^2
            e = e2 - Z * b;
            SST = sum((e2 - mean(e2)).^2);
            SSE = sum(e.^2);
            R2 = 1 - SSE/SST;
            
            % 计算LM统计量
            stat = n * R2;
            
            % 计算p值（LM服从自由度为p的卡方分布）
            pValue = 1 - chi2cdf(stat, p);
            
            % 确定假设检验结果（α = 0.05）
            h = (pValue < 0.05);
        end
        
        function [h, pValue, stat] = breuschPagan(resid, X)
            % Breusch-Pagan 异方差检验
            %
            % 参数:
            %   resid - 残差向量
            %   X - 自变量矩阵
            %
            % 返回值:
            %   h - 假设检验结果（1表示拒绝同方差假设）
            %   pValue - p值
            %   stat - 检验统计量
            
            n = length(resid);
            X = [ones(n, 1), X];
            k = size(X, 2);
            
            % 标准化残差
            e2 = resid.^2;
            sigma2 = mean(e2);
            z = e2 / sigma2;
            
            % 辅助回归
            b = (X' * X) \ (X' * z);
            
            % 计算拟合值和残差
            z_hat = X * b;
            e = z - z_hat;
            
            % 计算回归平方和
            ESS = sum((z_hat - mean(z)).^2);
            
            % 计算LM统计量
            stat = ESS / 2;
            
            % 计算p值（LM服从自由度为k-1的卡方分布）
            pValue = 1 - chi2cdf(stat, k-1);
            
            % 确定假设检验结果（α = 0.05）
            h = (pValue < 0.05);
        end
    end
end