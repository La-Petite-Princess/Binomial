classdef BootstrapSampler < handle
    % BootstrapSampler - 自助法采样和分析
    %
    % 该类实现了Bootstrap自助法抽样功能，用于估计模型参数的不确定性、
    % 生成置信区间，并进行稳健性分析。支持多种抽样策略和并行计算。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   X - 自变量矩阵
    %   y - 因变量向量
    %   samples - Bootstrap样本数
    %   bootIndices - Bootstrap抽样索引
    %   bootStats - Bootstrap统计结果
    %   strategy - 抽样策略
    %   sampleRatio - 样本比例
    %   confLevel - 置信水平
    %   seed - 随机数种子
    %   useParallel - 是否使用并行计算
    
    properties
        logger          % 日志记录器
        X               % 自变量矩阵
        y               % 因变量向量
        samples         % Bootstrap样本数
        bootIndices     % Bootstrap抽样索引
        bootStats       % Bootstrap统计结果
        strategy        % 抽样策略
        sampleRatio     % 样本比例
        confLevel       % 置信水平
        seed            % 随机数种子
        useParallel     % 是否使用并行计算
    end
    
    methods
        function obj = BootstrapSampler(logger)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('BootstrapSampler');
            else
                obj.logger = logger;
            end
            
            % 初始化默认参数
            obj.samples = 1000;
            obj.bootIndices = {};
            obj.bootStats = struct();
            obj.strategy = 'resample';  % 默认使用有放回抽样
            obj.sampleRatio = 1.0;      % 默认样本比例为100%
            obj.confLevel = 0.95;       % 默认置信水平为95%
            obj.seed = [];              % 默认随机种子
            obj.useParallel = false;    % 默认不使用并行计算
            
            obj.logger.info('Bootstrap采样器已初始化');
        end
        
        function setSamples(obj, n)
            % 设置Bootstrap样本数
            %
            % 参数:
            %   n - 样本数
            
            if n > 0
                obj.samples = round(n);
                obj.logger.debug('Bootstrap样本数设置为%d', obj.samples);
            else
                obj.logger.warn('无效的样本数%d, 保持原有设置%d', n, obj.samples);
            end
        end
        
        function setStrategy(obj, strategy)
            % 设置抽样策略
            %
            % 参数:
            %   strategy - 抽样策略:
            %     'resample' - 有放回抽样
            %     'jackknife' - 刀切法(留一法)
            %     'subsampling' - 子采样法(无放回)
            %     'balanced' - 平衡自助法
            
            validStrategies = {'resample', 'jackknife', 'subsampling', 'balanced'};
            
            if ismember(lower(strategy), validStrategies)
                obj.strategy = lower(strategy);
                obj.logger.debug('抽样策略设置为: %s', obj.strategy);
                
                % 如果是jackknife策略，样本数等于观测数
                if strcmp(obj.strategy, 'jackknife') && ~isempty(obj.X)
                    obj.samples = size(obj.X, 1);
                    obj.logger.debug('Jackknife策略: 样本数自动设置为%d', obj.samples);
                end
            else
                obj.logger.warn('无效的抽样策略: %s，保持原有设置: %s', ...
                    strategy, obj.strategy);
            end
        end
        
        function setSampleRatio(obj, ratio)
            % 设置样本比例
            %
            % 参数:
            %   ratio - 样本比例(0,1]
            
            if ratio > 0 && ratio <= 1
                obj.sampleRatio = ratio;
                obj.logger.debug('样本比例设置为%.2f', obj.sampleRatio);
            else
                obj.logger.warn('无效的样本比例%.2f，保持原有设置%.2f', ...
                    ratio, obj.sampleRatio);
            end
        end
        
        function setConfidenceLevel(obj, level)
            % 设置置信水平
            %
            % 参数:
            %   level - 置信水平(0,1)
            
            if level > 0 && level < 1
                obj.confLevel = level;
                obj.logger.debug('置信水平设置为%.2f', obj.confLevel);
            else
                obj.logger.warn('无效的置信水平%.2f，保持原有设置%.2f', ...
                    level, obj.confLevel);
            end
        end
        
        function setSeed(obj, seed)
            % 设置随机数种子
            %
            % 参数:
            %   seed - 随机数种子
            
            obj.seed = seed;
            obj.logger.debug('随机数种子已设置');
        end
        
        function setParallel(obj, useParallel)
            % 设置是否使用并行计算
            %
            % 参数:
            %   useParallel - 布尔值，true表示使用并行计算
            
            obj.useParallel = logical(useParallel);
            obj.logger.debug('并行计算设置为: %d', obj.useParallel);
        end
        
        function generateSamples(obj, X, y)
            % 生成Bootstrap样本
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            
            obj.X = X;
            obj.y = y;
            [n, ~] = size(X);
            
            % 设置随机数种子(如果有)
            if ~isempty(obj.seed)
                rng(obj.seed);
            end
            
            % 清空现有样本
            obj.bootIndices = cell(obj.samples, 1);
            
            % 根据策略生成样本
            switch obj.strategy
                case 'resample'
                    % 有放回抽样
                    sampleSize = round(n * obj.sampleRatio);
                    for i = 1:obj.samples
                        obj.bootIndices{i} = randi(n, sampleSize, 1);
                    end
                    
                case 'jackknife'
                    % 刀切法(留一法)
                    obj.samples = n;  % 确保样本数等于观测数
                    for i = 1:n
                        obj.bootIndices{i} = setdiff(1:n, i)';
                    end
                    
                case 'subsampling'
                    % 子采样法(无放回)
                    sampleSize = round(n * obj.sampleRatio);
                    for i = 1:obj.samples
                        obj.bootIndices{i} = randsample(n, sampleSize, false);
                    end
                    
                case 'balanced'
                    % 平衡自助法(确保每个观测被选择相同次数)
                    sampleSize = round(n * obj.sampleRatio);
                    totalSelections = sampleSize * obj.samples;
                    selectionsPerObs = floor(totalSelections / n);
                    
                    % 创建平衡的选择池
                    selectionPool = repmat(1:n, 1, selectionsPerObs);
                    
                    % 添加剩余的随机选择
                    remainder = totalSelections - length(selectionPool);
                    if remainder > 0
                        selectionPool = [selectionPool, randsample(n, remainder, true)];
                    end
                    
                    % 打乱选择池并分配给各样本
                    selectionPool = selectionPool(randperm(length(selectionPool)));
                    for i = 1:obj.samples
                        startIdx = (i-1) * sampleSize + 1;
                        endIdx = i * sampleSize;
                        obj.bootIndices{i} = selectionPool(startIdx:endIdx)';
                    end
            end
            
            obj.logger.info('已生成%d个Bootstrap样本，采样策略: %s，样本比例: %.2f', ...
                obj.samples, obj.strategy, obj.sampleRatio);
        end
        
        function results = analyze(obj, modelFun)
            % 分析Bootstrap样本
            %
            % 参数:
            %   modelFun - 建模函数句柄，形式为 function model = modelFun(X, y)
            %
            % 返回值:
            %   results - Bootstrap分析结果结构体
            
            if isempty(obj.bootIndices)
                obj.logger.error('尚未生成Bootstrap样本，请先调用generateSamples方法');
                results = struct();
                return;
            end
            
            n_samples = length(obj.bootIndices);
            n_obs = size(obj.X, 1);
            
            % 记录要收集的指标名称，动态适应不同的模型函数
            firstModel = modelFun(obj.X(obj.bootIndices{1}, :), obj.y(obj.bootIndices{1}));
            fieldNames = fieldnames(firstModel);
            
            % 初始化存储不同样本中系数估计和统计量的数组
            stats = struct();
            
            % 初始化一个临时数组来存储训练集大小，而不是直接修改结构体
            temp_train_sizes = zeros(n_samples, 1);
            
            % 为每个字段创建单元格数组
            for i = 1:length(fieldNames)
                field = fieldNames{i};
                fieldValue = firstModel.(field);
                
                % 只存储数值型字段
                if isnumeric(fieldValue)
                    % 对于1D或2D数组，保留其形状
                    stats.(field) = cell(n_samples, 1);
                    stats.(field){1} = fieldValue;
                end
            end
            
            % 保存第一个样本的训练集大小
            temp_train_sizes(1) = length(obj.bootIndices{1});
            
            % 创建临时单元格数组存储模型结果
            models = cell(n_samples, 1);
            models{1} = firstModel;
            
            % 使用并行或顺序计算处理剩余的Bootstrap样本
            if obj.useParallel
                obj.logger.info('使用并行计算处理Bootstrap样本...');
                
                % 创建临时变量以在parfor循环中使用
                X = obj.X;
                y = obj.y;
                bootIndices = obj.bootIndices;
                
                parfor i = 2:n_samples
                    % 获取当前样本的训练数据
                    train_idx = bootIndices{i};
                    X_train = X(train_idx, :);
                    y_train = y(train_idx);
                    
                    % 调用模型函数处理当前样本
                    models{i} = modelFun(X_train, y_train);
                    
                    % 保存训练集大小
                    temp_train_sizes(i) = length(train_idx);
                end
            else
                % 顺序处理
                for i = 2:n_samples
                    % 获取当前样本的训练数据
                    train_idx = obj.bootIndices{i};
                    X_train = obj.X(train_idx, :);
                    y_train = obj.y(train_idx);
                    
                    % 调用模型函数处理当前样本
                    models{i} = modelFun(X_train, y_train);
                    
                    % 保存训练集大小
                    temp_train_sizes(i) = length(train_idx);
                end
            end
            
            % 将模型结果存入stats结构体
            for i = 2:n_samples
                model = models{i};
                for j = 1:length(fieldNames)
                    field = fieldNames{j};
                    if isfield(stats, field)
                        stats.(field){i} = model.(field);
                    end
                end
            end
            
            % 整合结果
            results = struct();
            results.n_samples = n_samples;
            results.train_sizes = temp_train_sizes; % 使用临时数组
            
            % 计算统计量（均值、标准差、置信区间等）
            alpha = 1 - obj.confLevel;
            lower_percentile = alpha / 2 * 100;
            upper_percentile = (1 - alpha / 2) * 100;
            
            fieldNames = fieldnames(stats);
            for i = 1:length(fieldNames)
                field = fieldNames{i};
                
                % 提取所有样本的当前字段值
                all_values = stats.(field);
                
                % 只处理数值型数据
                if iscell(all_values) && all(cellfun(@isnumeric, all_values))
                    % 对于系数等向量/矩阵，需要特殊处理
                    first_value = all_values{1};
                    
                    if isscalar(first_value)
                        % 对于标量字段
                        all_scalars = cellfun(@(x) x, all_values);
                        results.([field '_mean']) = mean(all_scalars);
                        results.([field '_std']) = std(all_scalars);
                        results.([field '_median']) = median(all_scalars);
                        results.([field '_ci_lower']) = prctile(all_scalars, lower_percentile);
                        results.([field '_ci_upper']) = prctile(all_scalars, upper_percentile);
                    else
                        % 对于向量/矩阵字段
                        % 首先检查所有样本的维度是否一致
                        dims = cellfun(@size, all_values, 'UniformOutput', false);
                        if ~isequal(dims{:})
                            obj.logger.warn('字段 %s 在不同样本中的维度不一致，无法计算统计量', field);
                            continue;
                        end
                        
                        % 将单元格数组转为3D数组以便计算
                        all_arrays = cat(3, all_values{:});
                        
                        % 计算沿第三维的统计量
                        results.([field '_mean']) = mean(all_arrays, 3);
                        results.([field '_std']) = std(all_arrays, 0, 3);
                        results.([field '_median']) = median(all_arrays, 3);
                        
                        % 对于每个元素计算置信区间
                        sz = size(all_arrays);
                        results.([field '_ci_lower']) = zeros(sz(1), sz(2));
                        results.([field '_ci_upper']) = zeros(sz(1), sz(2));
                        
                        for row = 1:sz(1)
                            for col = 1:sz(2)
                                values = squeeze(all_arrays(row, col, :));
                                results.([field '_ci_lower'])(row, col) = prctile(values, lower_percentile);
                                results.([field '_ci_upper'])(row, col) = prctile(values, upper_percentile);
                            end
                        end
                    end
                end
            end
            
            % 保存原始数据供后续分析
            results.raw = stats;
            
            % 保存结果
            obj.bootStats = results;
            
            obj.logger.info('Bootstrap分析完成，样本数: %d，置信水平: %.2f%%', ...
                n_samples, obj.confLevel * 100);
            
            return;
        end
        
        function [ci_lower, ci_upper] = getConfidenceInterval(obj, paramName, paramIndex)
            % 获取参数的置信区间
            %
            % 参数:
            %   paramName - 参数名称（如 'coefficients'）
            %   paramIndex - 参数索引（可选，用于向量参数）
            %
            % 返回值:
            %   ci_lower - 置信区间下限
            %   ci_upper - 置信区间上限
            
            if isempty(obj.bootStats)
                obj.logger.error('尚未进行Bootstrap分析，请先调用analyze方法');
                ci_lower = [];
                ci_upper = [];
                return;
            end
            
            % 检查参数是否存在
            lower_field = [paramName '_ci_lower'];
            upper_field = [paramName '_ci_upper'];
            
            if ~isfield(obj.bootStats, lower_field) || ~isfield(obj.bootStats, upper_field)
                obj.logger.error('参数 %s 不存在或未计算置信区间', paramName);
                ci_lower = [];
                ci_upper = [];
                return;
            end
            
            % 提取置信区间
            if nargin < 3 || isempty(paramIndex)
                % 返回完整置信区间
                ci_lower = obj.bootStats.(lower_field);
                ci_upper = obj.bootStats.(upper_field);
            else
                % 返回特定索引的置信区间
                ci_lower = obj.bootStats.(lower_field)(paramIndex);
                ci_upper = obj.bootStats.(upper_field)(paramIndex);
            end
        end
        
        function result = getBootstrapDistribution(obj, paramName, paramIndex)
            % 获取参数的Bootstrap分布
            %
            % 参数:
            %   paramName - 参数名称（如 'coefficients'）
            %   paramIndex - 参数索引（可选，用于向量参数）
            %
            % 返回值:
            %   result - 包含参数分布的向量
            
            if isempty(obj.bootStats) || ~isfield(obj.bootStats, 'raw') || ...
                    ~isfield(obj.bootStats.raw, paramName)
                obj.logger.error('尚未进行Bootstrap分析或参数 %s 不存在', paramName);
                result = [];
                return;
            end
            
            % 提取所有样本的参数值
            all_values = obj.bootStats.raw.(paramName);
            
            % 检查是否是标量参数
            first_value = all_values{1};
            if isscalar(first_value)
                result = cellfun(@(x) x, all_values);
            else
                % 对于向量/矩阵参数，提取特定索引的值
                if nargin < 3 || isempty(paramIndex)
                    obj.logger.error('向量参数 %s 需要指定索引', paramName);
                    result = [];
                    return;
                end
                
                % 提取所有样本中特定索引的值
                % 处理一维和多维索引
                if isscalar(paramIndex)
                    result = cellfun(@(x) x(paramIndex), all_values);
                else
                    result = cellfun(@(x) x(paramIndex(1), paramIndex(2)), all_values);
                end
            end
        end
        
        function plotParameterDistribution(obj, paramName, paramIndex, nbins)
            % 绘制参数分布直方图
            %
            % 参数:
            %   paramName - 参数名称（如 'coefficients'）
            %   paramIndex - 参数索引（可选，用于向量参数）
            %   nbins - 直方图箱数（可选，默认为自动确定）
            
            % 获取参数分布
            if nargin < 3 
                paramDist = obj.getBootstrapDistribution(paramName);
            else
                paramDist = obj.getBootstrapDistribution(paramName, paramIndex);
            end
            
            if isempty(paramDist)
                return;
            end
            
            % 确定直方图箱数
            if nargin < 4 || isempty(nbins)
                nbins = min(max(10, ceil(sqrt(length(paramDist)))), 50);
            end
            
            % 创建图形
            figure;
            
            % 绘制直方图
            histogram(paramDist, nbins, 'Normalization', 'probability');
            hold on;
            
            % 添加核密度估计曲线
            [f, xi] = ksdensity(paramDist);
            plot(xi, f, 'r-', 'LineWidth', 2);
            
            % 添加置信区间
            [ci_lower, ci_upper] = obj.getConfidenceInterval(paramName, paramIndex);
            if ~isempty(ci_lower) && ~isempty(ci_upper)
                xline(ci_lower, '--g', sprintf('%.2f%%', obj.confLevel*50));
                xline(ci_upper, '--g', sprintf('%.2f%%', (1+obj.confLevel)*50));
            end
            
            % 添加均值和中位数
            if isfield(obj.bootStats, [paramName '_mean'])
                if nargin < 3 || isempty(paramIndex)
                    mean_val = obj.bootStats.([paramName '_mean']);
                    median_val = obj.bootStats.([paramName '_median']);
                else
                    mean_val = obj.bootStats.([paramName '_mean'])(paramIndex);
                    median_val = obj.bootStats.([paramName '_median'])(paramIndex);
                end
                
                xline(mean_val, '-b', '均值');
                xline(median_val, '-m', '中位数');
            end
            
            % 设置标题和标签
            if nargin < 3 || isempty(paramIndex)
                title(sprintf('%s的Bootstrap分布', paramName));
            else
                if isscalar(paramIndex)
                    title(sprintf('%s(%d)的Bootstrap分布', paramName, paramIndex));
                else
                    title(sprintf('%s(%d,%d)的Bootstrap分布', paramName, paramIndex(1), paramIndex(2)));
                end
            end
            
            xlabel('参数值');
            ylabel('频率');
            grid on;
            
            % 添加图例
            legend({'直方图', '核密度估计', ...
                [num2str(obj.confLevel*50) '%分位数'], ...
                [num2str((1+obj.confLevel)*50) '%分位数'], ...
                '均值', '中位数'}, 'Location', 'best');
            
            hold off;
        end
        
        function plotConfidenceIntervals(obj, paramName)
            % 绘制参数的置信区间
            %
            % 参数:
            %   paramName - 参数名称（如 'coefficients'）
            
            % 检查参数是否存在
            mean_field = [paramName '_mean'];
            lower_field = [paramName '_ci_lower'];
            upper_field = [paramName '_ci_upper'];
            
            if ~isfield(obj.bootStats, mean_field) || ...
               ~isfield(obj.bootStats, lower_field) || ...
               ~isfield(obj.bootStats, upper_field)
                obj.logger.error('参数 %s 不存在或未计算置信区间', paramName);
                return;
            end
            
            % 提取数据
            means = obj.bootStats.(mean_field);
            lowers = obj.bootStats.(lower_field);
            uppers = obj.bootStats.(upper_field);
            
            % 处理标量情况
            if isscalar(means)
                obj.plotParameterDistribution(paramName);
                return;
            end
            
            % 处理向量参数
            n_params = length(means);
            
            % 创建图形
            figure;
            
            % 绘制均值和置信区间
            errorbar(1:n_params, means, means - lowers, uppers - means, 'o-');
            
            % 添加零线（如果有跨越零的区间）
            if min(lowers) < 0 && max(uppers) > 0
                hold on;
                plot([1, n_params], [0, 0], 'k--');
                hold off;
            end
            
            % 设置x轴刻度和标签
            if isfield(obj.bootStats, 'parameterNames') && ...
               length(obj.bootStats.parameterNames) == n_params
                xticks(1:n_params);
                xticklabels(obj.bootStats.parameterNames);
                xtickangle(45);
            end
            
            % 设置标题和标签
            title(sprintf('%s的%.0f%%置信区间', paramName, obj.confLevel*100));
            xlabel('参数');
            ylabel('估计值');
            grid on;
        end
    end
end