classdef VariableContributionEvaluator < handle
    % VariableContributionEvaluator - 评估变量对模型的贡献
    %
    % 该类负责评估模型中各个变量的相对重要性和贡献度，
    % 包括偏相关分析、弹性分析、边际效应等度量方法。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   X - 自变量矩阵
    %   y - 因变量向量
    %   coefficients - 模型系数
    %   variableNames - 变量名称
    %   partialCorrelations - 偏相关系数
    %   elasticities - 弹性系数
    %   marginalEffects - 边际效应
    %   standardizedCoefficients - 标准化系数
    %   dominanceMetrics - 显性度量指标
    %   shapleyValues - Shapley值
    %   importanceRanking - 变量重要性排名
    
    properties
        logger                   % 日志记录器
        X                        % 自变量矩阵
        y                        % 因变量向量
        coefficients             % 模型系数
        variableNames            % 变量名称
        partialCorrelations      % 偏相关系数
        elasticities            % 弹性系数
        marginalEffects          % 边际效应
        standardizedCoefficients % 标准化系数
        dominanceMetrics         % 显性度量指标
        shapleyValues            % Shapley值
        importanceRanking        % 变量重要性排名
        contributionMetrics      % 贡献度量指标集合
    end
    
    methods
        function obj = VariableContributionEvaluator(logger)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('VariableContributionEvaluator');
            else
                obj.logger = logger;
            end
            
            obj.logger.info('变量贡献评估模块已初始化');
        end
        
        function evaluate(obj, X, y, coefficients, variableNames)
            % 评估变量贡献
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   coefficients - 模型系数
            %   variableNames - 变量名称元胞数组(可选)
            
            obj.X = X;
            obj.y = y;
            obj.coefficients = coefficients(:);
            
            % 处理变量名称
            if nargin < 5 || isempty(variableNames)
                obj.variableNames = arrayfun(@(i) sprintf('Var_%d', i), ...
                    1:size(X, 2), 'UniformOutput', false);
            else
                obj.variableNames = variableNames;
            end
            
            % 计算各种贡献指标
            obj.calculatePartialCorrelations();
            obj.calculateElasticities();
            obj.calculateMarginalEffects();
            obj.calculateStandardizedCoefficients();
            
            % 高级贡献度量（可能计算量较大）
            obj.calculateDominanceMetrics();
            obj.calculateShapleyValues();
            
            % 综合排名
            obj.rankVariableImportance();
            
            obj.logger.info('变量贡献评估完成');
        end
        
        function calculatePartialCorrelations(obj)
            % 计算偏相关系数
            
            [n, p] = size(obj.X);
            obj.partialCorrelations = zeros(p, 1);
            
            try
                % 对每个变量，计算其与y的偏相关系数
                for i = 1:p
                    % 选择除了当前变量之外的所有变量
                    otherVars = setdiff(1:p, i);
                    
                    if isempty(otherVars)
                        % 如果只有一个变量，直接计算相关系数
                        obj.partialCorrelations(i) = corr(obj.X(:,i), obj.y);
                    else
                        % 将y对其他变量回归
                        X_others = obj.X(:, otherVars);
                        beta_others = (X_others' * X_others) \ (X_others' * obj.y);
                        y_res = obj.y - X_others * beta_others;
                        
                        % 将Xi对其他变量回归
                        Xi = obj.X(:, i);
                        beta_xi = (X_others' * X_others) \ (X_others' * Xi);
                        xi_res = Xi - X_others * beta_xi;
                        
                        % 计算残差之间的相关系数
                        obj.partialCorrelations(i) = corr(xi_res, y_res);
                    end
                end
            catch ME
                obj.logger.error('计算偏相关系数失败: %s', ME.message);
                obj.partialCorrelations = nan(p, 1);
            end
        end
        
        function calculateElasticities(obj)
            % 计算弹性系数
            % 弹性 = β * (X的均值/y的均值)
            
            p = size(obj.X, 2);
            obj.elasticities = zeros(p, 1);
            
            try
                % 计算X和y的均值
                meanX = mean(obj.X);
                meanY = mean(obj.y);
                
                % 计算弹性系数
                for i = 1:p
                    if meanY ~= 0
                        obj.elasticities(i) = obj.coefficients(i) * meanX(i) / meanY;
                    else
                        obj.elasticities(i) = NaN;
                    end
                end
            catch ME
                obj.logger.error('计算弹性系数失败: %s', ME.message);
                obj.elasticities = nan(p, 1);
            end
        end
        
        function calculateMarginalEffects(obj)
            % 计算边际效应
            % 线性模型中，边际效应就是系数本身
            % 非线性模型中，需要在特定点上评估斜率
            
            obj.marginalEffects = obj.coefficients;
            
            % 这里只实现线性模型的简单情况
            % 对于非线性模型，可扩展此方法
        end
        
        function calculateStandardizedCoefficients(obj)
            % 计算标准化系数
            % 标准化系数 = β * (X的标准差/y的标准差)
            
            p = size(obj.X, 2);
            obj.standardizedCoefficients = zeros(p, 1);
            
            try
                % 计算X和y的标准差
                stdX = std(obj.X);
                stdY = std(obj.y);
                
                % 计算标准化系数
                for i = 1:p
                    if stdY ~= 0
                        obj.standardizedCoefficients(i) = obj.coefficients(i) * stdX(i) / stdY;
                    else
                        obj.standardizedCoefficients(i) = NaN;
                    end
                end
            catch ME
                obj.logger.error('计算标准化系数失败: %s', ME.message);
                obj.standardizedCoefficients = nan(p, 1);
            end
        end
        
        function calculateDominanceMetrics(obj)
            % 计算显性度量指标
            % 基于所有可能的变量子集组合来评估变量贡献
            
            p = size(obj.X, 2);
            obj.dominanceMetrics = zeros(p, 1);
            
            % 如果变量较多，计算量会非常大
            if p > 15
                obj.logger.warn('变量数量大于15，跳过显性度量计算，计算量太大');
                obj.dominanceMetrics = nan(p, 1);
                return;
            end
            
            try
                % 对每个变量，计算其在所有可能的子集模型中的平均边际贡献
                % 这里计算基于R^2增量的显性度量
                
                % 预计算所有变量的单独R^2
                singleR2 = zeros(p, 1);
                for i = 1:p
                    Xi = obj.X(:, i);
                    Xi = [ones(size(Xi, 1), 1), Xi]; % 添加常数项
                    bi = (Xi' * Xi) \ (Xi' * obj.y);
                    yHat = Xi * bi;
                    SST = sum((obj.y - mean(obj.y)).^2);
                    SSE = sum((obj.y - yHat).^2);
                    singleR2(i) = 1 - SSE/SST;
                end
                
                % 计算每个变量在所有可能子集中的平均贡献
                totalContribution = zeros(p, 1);
                totalCount = zeros(p, 1);
                
                % 生成所有可能的变量子集
                for setSize = 0:p-1
                    % 获取大小为setSize的所有子集
                    combs = nchoosek(1:p, setSize);
                    
                    for c = 1:size(combs, 1)
                        currentSet = combs(c, :);
                        
                        % 计算当前子集的R^2
                        if isempty(currentSet)
                            currentR2 = 0;
                        else
                            Xsub = obj.X(:, currentSet);
                            Xsub = [ones(size(Xsub, 1), 1), Xsub]; % 添加常数项
                            bSub = (Xsub' * Xsub) \ (Xsub' * obj.y);
                            yHat = Xsub * bSub;
                            SST = sum((obj.y - mean(obj.y)).^2);
                            SSE = sum((obj.y - yHat).^2);
                            currentR2 = 1 - SSE/SST;
                        end
                        
                        % 计算添加每个剩余变量的边际贡献
                        remainingVars = setdiff(1:p, currentSet);
                        for v = remainingVars
                            % 将变量v添加到当前子集
                            newSet = [currentSet, v];
                            Xnew = obj.X(:, newSet);
                            Xnew = [ones(size(Xnew, 1), 1), Xnew]; % 添加常数项
                            bNew = (Xnew' * Xnew) \ (Xnew' * obj.y);
                            yHat = Xnew * bNew;
                            SST = sum((obj.y - mean(obj.y)).^2);
                            SSE = sum((obj.y - yHat).^2);
                            newR2 = 1 - SSE/SST;
                            
                            % 计算变量v的边际贡献
                            contribution = newR2 - currentR2;
                            totalContribution(v) = totalContribution(v) + contribution;
                            totalCount(v) = totalCount(v) + 1;
                        end
                    end
                end
                
                % 计算平均贡献
                obj.dominanceMetrics = totalContribution ./ totalCount;
            catch ME
                obj.logger.error('计算显性度量指标失败: %s', ME.message);
                obj.dominanceMetrics = nan(p, 1);
            end
        end
        
        function calculateShapleyValues(obj)
            % 计算Shapley值
            % 基于合作博弈理论，评估每个变量的边际贡献
            
            p = size(obj.X, 2);
            obj.shapleyValues = zeros(p, 1);
            
            % 如果变量较多，计算量会非常大
            if p > 12
                obj.logger.warn('变量数量大于12，跳过Shapley值计算，计算量太大');
                obj.shapleyValues = nan(p, 1);
                return;
            end
            
            try
                % 计算空模型的R^2(只有常数项)
                X0 = ones(size(obj.X, 1), 1);
                b0 = (X0' * X0) \ (X0' * obj.y);
                yHat0 = X0 * b0;
                SST = sum((obj.y - mean(obj.y)).^2);
                SSE0 = sum((obj.y - yHat0).^2);
                R2_0 = 1 - SSE0/SST;
                
                % 计算完整模型的R^2
                Xfull = [ones(size(obj.X, 1), 1), obj.X];
                bFull = (Xfull' * Xfull) \ (Xfull' * obj.y);
                yHatFull = Xfull * bFull;
                SSEfull = sum((obj.y - yHatFull).^2);
                R2_full = 1 - SSEfull/SST;
                
                % 计算每个变量的Shapley值
                for i = 1:p
                    shapley_i = 0;
                    
                    % 枚举所有不包括变量i的子集
                    otherVars = setdiff(1:p, i);
                    
                    for setSize = 0:length(otherVars)
                        % 所有大小为setSize的子集
                        if setSize == 0
                            combs = {[]};
                        else
                            combs_indices = nchoosek(1:length(otherVars), setSize);
                            combs = cell(size(combs_indices, 1), 1);
                            for c = 1:size(combs_indices, 1)
                                combs{c} = otherVars(combs_indices(c, :));
                            end
                        end
                        
                        % 对每个子集，计算添加变量i的边际贡献
                        for c = 1:length(combs)
                            S = combs{c};
                            
                            % 子集S的R^2
                            if isempty(S)
                                R2_S = R2_0;
                            else
                                XS = [ones(size(obj.X, 1), 1), obj.X(:, S)];
                                bS = (XS' * XS) \ (XS' * obj.y);
                                yHatS = XS * bS;
                                SSES = sum((obj.y - yHatS).^2);
                                R2_S = 1 - SSES/SST;
                            end
                            
                            % 子集S加入变量i后的R^2
                            S_with_i = [S, i];
                            if length(S_with_i) == p
                                R2_S_with_i = R2_full;
                            else
                                XS_with_i = [ones(size(obj.X, 1), 1), obj.X(:, S_with_i)];
                                bS_with_i = (XS_with_i' * XS_with_i) \ (XS_with_i' * obj.y);
                                yHatS_with_i = XS_with_i * bS_with_i;
                                SSES_with_i = sum((obj.y - yHatS_with_i).^2);
                                R2_S_with_i = 1 - SSES_with_i/SST;
                            end
                            
                            % 边际贡献
                            marginal_contribution = R2_S_with_i - R2_S;
                            
                            % Shapley公式的权重
                            weight = factorial(setSize) * factorial(p - setSize - 1) / factorial(p);
                            
                            % 累加加权贡献
                            shapley_i = shapley_i + weight * marginal_contribution;
                        end
                    end
                    
                    obj.shapleyValues(i) = shapley_i;
                end
            catch ME
                obj.logger.error('计算Shapley值失败: %s', ME.message);
                obj.shapleyValues = nan(p, 1);
            end
        end
        
        function rankVariableImportance(obj)
            % 基于多种贡献指标对变量进行综合排名
            
            p = size(obj.X, 2);
            
            % 收集所有可用的贡献指标
            metrics = {
                abs(obj.standardizedCoefficients), ...
                abs(obj.partialCorrelations), ...
                abs(obj.elasticities)
            };
            
            % 添加高级指标（如果计算了的话）
            if ~all(isnan(obj.dominanceMetrics))
                metrics{end+1} = obj.dominanceMetrics;
            end
            
            if ~all(isnan(obj.shapleyValues))
                metrics{end+1} = obj.shapleyValues;
            end
            
            % 计算每个指标的排名
            ranks = zeros(p, length(metrics));
            for i = 1:length(metrics)
                metric = metrics{i};
                if all(isnan(metric))
                    ranks(:, i) = nan(p, 1);
                else
                    [~, idx] = sort(metric, 'descend');
                    tempRank = zeros(p, 1);
                    tempRank(idx) = 1:p;
                    ranks(:, i) = tempRank;
                end
            end
            
            % 计算平均排名
            obj.importanceRanking = mean(ranks, 2, 'omitnan');
            
            % 整合所有贡献度量指标
            obj.contributionMetrics = struct();
            obj.contributionMetrics.standardizedCoefficients = obj.standardizedCoefficients;
            obj.contributionMetrics.partialCorrelations = obj.partialCorrelations;
            obj.contributionMetrics.elasticities = obj.elasticities;
            obj.contributionMetrics.marginalEffects = obj.marginalEffects;
            obj.contributionMetrics.dominanceMetrics = obj.dominanceMetrics;
            obj.contributionMetrics.shapleyValues = obj.shapleyValues;
            obj.contributionMetrics.importanceRanking = obj.importanceRanking;
            
            obj.logger.info('变量重要性排名完成');
        end
        
        function result = getTopVariables(obj, n)
            % 获取最重要的n个变量
            %
            % 参数:
            %   n - 返回的变量数量
            %
            % 返回值:
            %   result - 包含top变量信息的结构体
            
            if nargin < 2 || isempty(n)
                n = min(5, length(obj.importanceRanking));
            else
                n = min(n, length(obj.importanceRanking));
            end
            
            [~, idx] = sort(obj.importanceRanking);
            topIndices = idx(1:n);
            
            result = struct();
            result.indices = topIndices;
            
            if ~isempty(obj.variableNames)
                result.names = obj.variableNames(topIndices);
            else
                result.names = arrayfun(@(i) sprintf('Var_%d', i), ...
                    topIndices, 'UniformOutput', false);
            end
            
            result.standardizedCoefficients = obj.standardizedCoefficients(topIndices);
            result.partialCorrelations = obj.partialCorrelations(topIndices);
            result.elasticities = obj.elasticities(topIndices);
            
            if ~all(isnan(obj.shapleyValues))
                result.shapleyValues = obj.shapleyValues(topIndices);
            end
            
            if ~all(isnan(obj.dominanceMetrics))
                result.dominanceMetrics = obj.dominanceMetrics(topIndices);
            end
            
            obj.logger.info('已获取前 %d 个重要变量', n);
        end
        
        function plotContribution(obj)
            % 可视化变量贡献
            
            p = size(obj.X, 2);
            
            % 根据重要性排名排序
            [sortedRanks, sortIdx] = sort(obj.importanceRanking);
            sortedNames = obj.variableNames(sortIdx);
            
            % 创建可视化图表
            figure('Name', '变量贡献度分析', 'Position', [100, 100, 1000, 800]);
            
            % 标准化系数和偏相关系数对比
            subplot(2, 2, 1);
            metrics = [obj.standardizedCoefficients(sortIdx), obj.partialCorrelations(sortIdx)];
            bar(metrics);
            title('标准化系数和偏相关系数');
            xlabel('变量');
            ylabel('值');
            legend({'标准化系数', '偏相关系数'}, 'Location', 'best');
            set(gca, 'XTick', 1:p, 'XTickLabel', sortedNames);
            xtickangle(45);
            grid on;
            
            % 变量重要性综合排名
            subplot(2, 2, 2);
            barh(flip(sortedRanks));
            title('变量重要性排名（值越小越重要）');
            xlabel('平均排名');
            yticks(1:p);
            yticklabels(flip(sortedNames));
            grid on;
            
            % 使用雷达图展示前5个变量的多维指标
            subplot(2, 2, 3);
            
            top5Idx = sortIdx(1:min(5, p));
            metrics = {
                '标准化系数', ...
                '偏相关系数', ...
                '弹性系数'
            };
            
            % 添加高级指标（如果计算了的话）
            if ~all(isnan(obj.dominanceMetrics))
                metrics{end+1} = '显性度量';
            end
            
            if ~all(isnan(obj.shapleyValues))
                metrics{end+1} = 'Shapley值';
            end
            
            % 准备雷达图数据
            numMetrics = length(metrics);
            angles = linspace(0, 2*pi, numMetrics+1);
            radarData = zeros(min(5, p), numMetrics);
            
            % 填充雷达图数据，并标准化
            for i = 1:numMetrics
                switch metrics{i}
                    case '标准化系数'
                        vals = abs(obj.standardizedCoefficients(top5Idx));
                    case '偏相关系数'
                        vals = abs(obj.partialCorrelations(top5Idx));
                    case '弹性系数'
                        vals = abs(obj.elasticities(top5Idx));
                    case '显性度量'
                        vals = obj.dominanceMetrics(top5Idx);
                    case 'Shapley值'
                        vals = obj.shapleyValues(top5Idx);
                end
                
                % 标准化到0-1之间
                maxVal = max(abs(vals));
                if maxVal > 0
                    radarData(:, i) = abs(vals) / maxVal;
                end
            end
            
            % 绘制雷达图
            radarData = [radarData, radarData(:, 1)]; % 闭合雷达图
            polarplot([angles, angles(1)], [ones(1, numMetrics+1); radarData]');
            thetaticks(rad2deg(angles(1:end-1)));
            thetaticklabels(metrics);
            legend(obj.variableNames(top5Idx), 'Location', 'eastoutside');
            title('前5个重要变量的多维指标对比');
            
            % 变量贡献热图
            subplot(2, 2, 4);
            
            % 准备热图数据
            heatData = zeros(min(10, p), 3);
            topVars = min(10, p);
            topIdx = sortIdx(1:topVars);
            
            heatData(:, 1) = abs(obj.standardizedCoefficients(topIdx));
            heatData(:, 2) = abs(obj.partialCorrelations(topIdx));
            heatData(:, 3) = abs(obj.elasticities(topIdx));
            
            % 标准化每列
            for i = 1:size(heatData, 2)
                maxVal = max(heatData(:, i));
                if maxVal > 0
                    heatData(:, i) = heatData(:, i) / maxVal;
                end
            end
            
            % 绘制热图
            imagesc(heatData);
            colormap(jet);
            colorbar;
            
            % 设置坐标轴标签
            xticks(1:3);
            xticklabels({'标准化系数', '偏相关系数', '弹性系数'});
            yticks(1:topVars);
            yticklabels(obj.variableNames(topIdx));
            title('前10个变量的贡献指标热图');
            
            sgtitle('变量贡献分析');
            
            obj.logger.info('变量贡献分析图已生成');
        end
        
        function report = generateContributionReport(obj)
            % 生成变量贡献报告
            %
            % 返回值:
            %   report - 包含贡献分析结果的结构体
            
            % 创建贡献度指标表
            metrics = {
                obj.standardizedCoefficients, ...
                obj.partialCorrelations, ...
                obj.elasticities, ...
                obj.dominanceMetrics, ...
                obj.shapleyValues, ...
                obj.importanceRanking
            };
            
            metricNames = {
                'StandardizedCoef', ...
                'PartialCorr', ...
                'Elasticity', ...
                'Dominance', ...
                'Shapley', ...
                'AvgRank'
            };
            
            % 筛选有效指标
            validIdx = ~cellfun(@(x) all(isnan(x)), metrics);
            validMetrics = metrics(validIdx);
            validNames = metricNames(validIdx);
            
            % 创建表格
            T = array2table([validMetrics{:}], 'VariableNames', validNames);
            
            % 添加变量名
            if ~isempty(obj.variableNames)
                T.Properties.RowNames = obj.variableNames;
            end
            
            % 按重要性排序
            if ~isempty(obj.importanceRanking) && ~all(isnan(obj.importanceRanking))
                [~, sortIdx] = sort(obj.importanceRanking);
                T = T(sortIdx, :);
            end
            
            % 创建报告结构体
            report = struct();
            report.contributionTable = T;
            report.topVariables = obj.getTopVariables(5);
            
            % 计算变量贡献百分比
            if ~all(isnan(obj.shapleyValues))
                totalContribution = sum(obj.shapleyValues);
                if totalContribution > 0
                    report.contributionPercentage = 100 * obj.shapleyValues / totalContribution;
                else
                    report.contributionPercentage = zeros(size(obj.shapleyValues));
                end
            elseif ~all(isnan(obj.dominanceMetrics))
                totalContribution = sum(obj.dominanceMetrics);
                if totalContribution > 0
                    report.contributionPercentage = 100 * obj.dominanceMetrics / totalContribution;
                else
                    report.contributionPercentage = zeros(size(obj.dominanceMetrics));
                end
            else
                stdCoefs = abs(obj.standardizedCoefficients);
                totalStd = sum(stdCoefs);
                if totalStd > 0
                    report.contributionPercentage = 100 * stdCoefs / totalStd;
                else
                    report.contributionPercentage = zeros(size(stdCoefs));
                end
            end
            
            obj.logger.info('变量贡献报告已生成');
        end
    end
    
    methods (Static)
        function [coefs, R2s] = calculateSequentialR2(X, y)
            % 计算序贯R^2，评估变量添加顺序对模型拟合的影响
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %
            % 返回值:
            %   coefs - 每个步骤的系数
            %   R2s - 每个步骤的R^2
            
            [n, p] = size(X);
            
            % 存储每一步的R^2和系数
            R2s = zeros(p, 1);
            coefs = cell(p, 1);
            
            % 计算总平方和
            yMean = mean(y);
            SST = sum((y - yMean).^2);
            
            % 逐步添加变量
            Xcurrent = ones(n, 1);
            
            for step = 1:p
                % 添加当前变量
                Xcurrent = [Xcurrent, X(:, step)];
                
                % 计算回归系数
                beta = (Xcurrent' * Xcurrent) \ (Xcurrent' * y);
                coefs{step} = beta;
                
                % 计算拟合值和R^2
                yHat = Xcurrent * beta;
                SSE = sum((y - yHat).^2);
                R2s(step) = 1 - SSE/SST;
            end
        end
        
        function lmg = calculateLMG(X, y)
            % 计算Lindeman-Merenda-Gold (LMG) 指标
            % LMG是Shapley值的一种实现，用于评估变量的相对重要性
            %
            % 参数:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %
            % 返回值:
            %   lmg - 每个变量的LMG指标
            
            [n, p] = size(X);
            
            % 如果变量太多，计算量会非常大
            if p > 12
                warning('变量数量大于12，LMG计算可能非常耗时');
            end
            
            % 计算总平方和
            yMean = mean(y);
            SST = sum((y - yMean).^2);
            
            % 用于存储每个变量的LMG值
            lmg = zeros(p, 1);
            
            % 枚举所有可能的变量排列
            allPerms = perms(1:p);
            numPerms = size(allPerms, 1);
            
            % 对每个排列计算序贯R^2增量
            for perm = 1:numPerms
                currentOrder = allPerms(perm, :);
                currentR2 = 0;
                
                % 按当前排列顺序逐步添加变量
                for i = 1:p
                    varIdx = currentOrder(1:i);
                    Xi = [ones(n, 1), X(:, varIdx)];
                    bi = (Xi' * Xi) \ (Xi' * y);
                    yHati = Xi * bi;
                    SSEi = sum((y - yHati).^2);
                    newR2 = 1 - SSEi/SST;
                    
                    % 当前变量的边际贡献
                    varContribution = newR2 - currentR2;
                    currentVar = currentOrder(i);
                    
                    % 累加到对应变量的LMG
                    lmg(currentVar) = lmg(currentVar) + varContribution;
                    
                    % 更新当前R^2
                    currentR2 = newR2;
                end
            end
            
            % 计算平均值
            lmg = lmg / numPerms;
        end
    end
end