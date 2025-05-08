classdef CoefficientStabilityMonitor < handle
    % CoefficientStabilityMonitor - 监控模型系数的稳定性
    %
    % 该类实现了模型系数稳定性的监控与评估功能，包括系数变化率计算、
    % 稳定性评分、系数历史跟踪等功能，帮助评估模型的稳健性和可靠性。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   historyCoefficients - 存储历史系数数据
    %   stabilityThreshold - 系数稳定性阈值
    %   baselineCoefficients - 基准系数
    %   coefficientNames - 系数名称列表
    %   stabilityScores - 每个系数的稳定性得分
    
    properties
        logger                  % 日志记录器
        historyCoefficients     % 存储历史系数矩阵 [迭代次数 x 系数个数]
        stabilityThreshold      % 稳定性阈值
        baselineCoefficients    % 基准系数
        coefficientNames        % 系数名称列表
        stabilityScores         % 系数稳定性得分
    end
    
    methods
        function obj = CoefficientStabilityMonitor(logger, stabilityThreshold)
            % 构造函数，初始化系数稳定性监控器
            %
            % 参数:
            %   logger - BinomialLogger实例
            %   stabilityThreshold - 稳定性阈值 (默认为0.05，表示5%的变化率)
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('CoefficientStabilityMonitor');
            else
                obj.logger = logger;
            end
            
            if nargin < 2 || isempty(stabilityThreshold)
                obj.stabilityThreshold = 0.05;  % 默认5%的变化率阈值
            else
                obj.stabilityThreshold = stabilityThreshold;
            end
            
            obj.historyCoefficients = [];
            obj.baselineCoefficients = [];
            obj.coefficientNames = {};
            obj.stabilityScores = [];
            
            obj.logger.info('系数稳定性监控器已初始化，阈值设置为 %.3f', obj.stabilityThreshold);
        end
        
        function initialize(obj, coefficients, coeffNames)
            % 初始化基准系数和系数名称
            %
            % 参数:
            %   coefficients - 初始系数向量
            %   coeffNames - 系数名称元胞数组
            
            obj.baselineCoefficients = coefficients(:);
            
            if nargin < 3 || isempty(coeffNames)
                obj.coefficientNames = arrayfun(@(i) sprintf('Coef_%d', i), ...
                    1:length(coefficients), 'UniformOutput', false);
            else
                obj.coefficientNames = coeffNames;
            end
            
            obj.historyCoefficients = coefficients(:)';
            obj.stabilityScores = ones(length(coefficients), 1);
            
            obj.logger.info('系数稳定性监控初始化完成，基准系数已设置');
        end
        
        function addCoefficients(obj, newCoefficients)
            % 添加新一轮的系数到历史记录
            %
            % 参数:
            %   newCoefficients - 新的系数向量
            
            if isempty(obj.historyCoefficients)
                obj.initialize(newCoefficients);
                return;
            end
            
            newCoefficients = newCoefficients(:)';
            
            if length(newCoefficients) ~= size(obj.historyCoefficients, 2)
                obj.logger.error('新系数长度 %d 与历史系数长度 %d 不匹配', ...
                    length(newCoefficients), size(obj.historyCoefficients, 2));
                return;
            end
            
            obj.historyCoefficients = [obj.historyCoefficients; newCoefficients];
            obj.calculateStabilityScores();
            
            obj.logger.debug('添加了新的系数集，当前历史记录中有 %d 组系数', ...
                size(obj.historyCoefficients, 1));
        end
        
        function calculateStabilityScores(obj)
            % 计算每个系数的稳定性得分
            % 得分基于系数的变异系数(CV)和相对于基准的变化率
            
            if size(obj.historyCoefficients, 1) < 2
                obj.stabilityScores = ones(size(obj.historyCoefficients, 2), 1);
                return;
            end
            
            % 计算变异系数 (标准差/均值)
            meanCoefs = mean(obj.historyCoefficients);
            stdCoefs = std(obj.historyCoefficients);
            
            % 避免除以零
            cvs = zeros(size(meanCoefs));
            nonZeroMeans = abs(meanCoefs) > eps;
            cvs(nonZeroMeans) = stdCoefs(nonZeroMeans) ./ abs(meanCoefs(nonZeroMeans));
            
            % 计算相对于基准的变化率
            latestCoefs = obj.historyCoefficients(end, :);
            changeRates = zeros(size(latestCoefs));
            
            nonZeroBase = abs(obj.baselineCoefficients) > eps;
            changeRates(nonZeroBase) = abs((latestCoefs(nonZeroBase) - obj.baselineCoefficients(nonZeroBase)') ./ ...
                obj.baselineCoefficients(nonZeroBase)');
            
            % 综合得分: 稳定性得分 = 1 - min(1, (CV + 变化率) / 2 / 阈值)
            combinedMetric = (cvs + changeRates') / 2;
            obj.stabilityScores = 1 - min(1, combinedMetric / obj.stabilityThreshold);
            
            obj.logger.debug('系数稳定性得分已更新');
        end
        
        function [unstableCoefs, indices] = getUnstableCoefficients(obj, threshold)
            % 获取不稳定的系数及其索引
            %
            % 参数:
            %   threshold - 稳定性阈值 (可选，默认为构造函数中设置的值)
            %
            % 返回值:
            %   unstableCoefs - 不稳定系数的名称列表
            %   indices - 不稳定系数的索引
            
            if nargin < 2 || isempty(threshold)
                threshold = 1 - obj.stabilityThreshold;
            end
            
            indices = find(obj.stabilityScores < threshold);
            
            if ~isempty(indices) && ~isempty(obj.coefficientNames)
                unstableCoefs = obj.coefficientNames(indices);
            else
                unstableCoefs = {};
            end
        end
        
        function plotStabilityTrends(obj)
            % 绘制系数稳定性趋势图
            
            if size(obj.historyCoefficients, 1) < 2
                obj.logger.warn('系数历史记录不足，无法绘制趋势图');
                return;
            end
            
            iterations = 1:size(obj.historyCoefficients, 1);
            numCoefs = size(obj.historyCoefficients, 2);
            
            % 创建多面板图
            figure('Name', '系数稳定性趋势', 'Position', [100, 100, 1000, 800]);
            
            % 系数值随迭代变化
            subplot(2, 2, 1);
            plot(iterations, obj.historyCoefficients);
            title('系数值随迭代变化');
            xlabel('迭代次数');
            ylabel('系数值');
            if ~isempty(obj.coefficientNames) && length(obj.coefficientNames) == numCoefs
                legend(obj.coefficientNames, 'Location', 'eastoutside');
            end
            grid on;
            
            % 系数相对变化率
            subplot(2, 2, 2);
            relativeChanges = zeros(length(iterations)-1, numCoefs);
            for i = 2:length(iterations)
                prev = obj.historyCoefficients(i-1, :);
                curr = obj.historyCoefficients(i, :);
                nonZeroPrev = abs(prev) > eps;
                relChange = zeros(1, numCoefs);
                relChange(nonZeroPrev) = abs((curr(nonZeroPrev) - prev(nonZeroPrev)) ./ prev(nonZeroPrev));
                relativeChanges(i-1, :) = relChange;
            end
            
            plot(iterations(2:end), relativeChanges);
            title('系数相对变化率');
            xlabel('迭代次数');
            ylabel('相对变化率');
            yline(obj.stabilityThreshold, '--r', '稳定性阈值');
            if ~isempty(obj.coefficientNames) && length(obj.coefficientNames) == numCoefs
                legend(obj.coefficientNames, 'Location', 'eastoutside');
            end
            grid on;
            
            % 系数稳定性得分
            subplot(2, 2, 3);
            bar(obj.stabilityScores);
            title('系数稳定性得分');
            xlabel('系数索引');
            ylabel('稳定性得分');
            yline(1 - obj.stabilityThreshold, '--r', '稳定性阈值');
            if ~isempty(obj.coefficientNames) && length(obj.coefficientNames) == numCoefs
                xticks(1:numCoefs);
                xticklabels(obj.coefficientNames);
                xtickangle(45);
            end
            grid on;
            
            % 累积变化情况
            subplot(2, 2, 4);
            cumulativeChanges = abs(obj.historyCoefficients - repmat(obj.baselineCoefficients', size(obj.historyCoefficients, 1), 1));
            nonZeroBase = abs(obj.baselineCoefficients) > eps;
            for i = 1:size(cumulativeChanges, 1)
                cumulativeChanges(i, nonZeroBase) = cumulativeChanges(i, nonZeroBase) ./ abs(obj.baselineCoefficients(nonZeroBase)');
            end
            
            plot(iterations, cumulativeChanges);
            title('相对于基准的累积变化');
            xlabel('迭代次数');
            ylabel('相对变化');
            if ~isempty(obj.coefficientNames) && length(obj.coefficientNames) == numCoefs
                legend(obj.coefficientNames, 'Location', 'eastoutside');
            end
            grid on;
            
            sgtitle('系数稳定性分析');
            
            obj.logger.info('系数稳定性趋势图已生成');
        end
        
        function result = analyzeStability(obj)
            % 分析系数稳定性并返回报告
            %
            % 返回值:
            %   result - 包含稳定性分析结果的结构体
            
            obj.calculateStabilityScores();
            
            [unstableCoefs, unstableIndices] = obj.getUnstableCoefficients();
            
            result = struct();
            result.numIterations = size(obj.historyCoefficients, 1);
            result.numCoefficients = size(obj.historyCoefficients, 2);
            result.stabilityScores = obj.stabilityScores;
            result.meanStabilityScore = mean(obj.stabilityScores);
            result.unstableCoefficients = unstableCoefs;
            result.unstableIndices = unstableIndices;
            result.percentUnstable = length(unstableIndices) / length(obj.stabilityScores) * 100;
            
            % 计算每个系数的变化趋势
            if size(obj.historyCoefficients, 1) >= 3
                trends = zeros(1, result.numCoefficients);
                for i = 1:result.numCoefficients
                    coefHistory = obj.historyCoefficients(:, i);
                    p = polyfit(1:length(coefHistory), coefHistory, 1);
                    trends(i) = p(1);  % 斜率
                end
                result.coefficientTrends = trends;
            else
                result.coefficientTrends = [];
            end
            
            % 计算收敛性 - 最近两次迭代的平均相对变化
            if size(obj.historyCoefficients, 1) >= 2
                last = obj.historyCoefficients(end, :);
                prevLast = obj.historyCoefficients(end-1, :);
                nonZeroPrev = abs(prevLast) > eps;
                
                relChanges = zeros(1, result.numCoefficients);
                relChanges(nonZeroPrev) = abs((last(nonZeroPrev) - prevLast(nonZeroPrev)) ./ prevLast(nonZeroPrev));
                result.recentConvergenceRate = mean(relChanges);
            else
                result.recentConvergenceRate = 0;
            end
            
            obj.logger.info(['系数稳定性分析完成: ', ...
                '平均稳定性得分 = %.4f, ', ...
                '不稳定系数百分比 = %.2f%%, ', ...
                '最近收敛率 = %.6f'], ...
                result.meanStabilityScore, ...
                result.percentUnstable, ...
                result.recentConvergenceRate);
        end
        
        function resetHistory(obj)
            % 重置历史记录，保留基准系数
            
            if ~isempty(obj.historyCoefficients) && ~isempty(obj.baselineCoefficients)
                obj.historyCoefficients = obj.baselineCoefficients(:)';
                obj.stabilityScores = ones(length(obj.baselineCoefficients), 1);
                obj.logger.info('系数历史记录已重置，保留了基准系数');
            else
                obj.historyCoefficients = [];
                obj.baselineCoefficients = [];
                obj.stabilityScores = [];
                obj.logger.info('系数历史记录和基准系数已完全重置');
            end
        end
    end
end