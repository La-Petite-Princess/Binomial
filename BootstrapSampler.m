classdef BootstrapSampler < handle
    % Bootstrap采样器类：实现高效的分层Bootstrap采样
    % 包括平衡采样、扰动增强和质量控制
    
    properties (Access = private)
        Config
        Logger
        SamplingStats
    end
    
    properties (Access = public)
        SampleStats
    end
    
    methods (Access = public)
        function obj = BootstrapSampler(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.SamplingStats = struct();
            obj.SampleStats = struct();
        end
        
        function [train_indices, test_indices] = Sample(obj, y)
            % 执行分层Bootstrap采样
            % 输入:
            %   y - 因变量
            % 输出:
            %   train_indices - 训练集索引（细胞数组）
            %   test_indices - 测试集索引（细胞数组）
            
            obj.Logger.Log('info', '开始Bootstrap分层抽样');
            
            try
                % 初始化
                n_samples = obj.Config.NumBootstrapSamples;
                train_ratio = obj.Config.TrainRatio;
                
                % 分析标签分布
                obj.AnalyzeLabelDistribution(y);
                
                % 生成采样策略
                sampling_strategy = obj.DetermineSamplingStrategy(y);
                
                % 执行采样
                [train_indices, test_indices] = obj.PerformSampling(y, n_samples, train_ratio, sampling_strategy);
                
                % 验证采样质量
                obj.ValidateSamples(y, train_indices, test_indices);
                
                % 生成采样报告
                obj.GenerateSamplingReport();
                
                obj.Logger.Log('info', sprintf('Bootstrap采样完成，生成了%d个训练/测试集对', n_samples));
                
            catch ME
                obj.Logger.LogException(ME, 'BootstrapSampler.Sample');
                rethrow(ME);
            end
        end
        
        function stats = GetSamplingStats(obj)
            % 获取采样统计信息
            stats = obj.SamplingStats;
        end
        
        function ExportSamples(obj, train_indices, test_indices, output_dir)
            % 导出采样结果
            try
                % 创建导出目录
                export_dir = fullfile(output_dir, 'bootstrap_samples');
                if ~exist(export_dir, 'dir')
                    mkdir(export_dir);
                end
                
                % 保存索引
                save(fullfile(export_dir, 'train_indices.mat'), 'train_indices', '-v7.3');
                save(fullfile(export_dir, 'test_indices.mat'), 'test_indices', '-v7.3');
                
                % 保存统计信息
                stats = obj.SamplingStats;
                save(fullfile(export_dir, 'sampling_stats.mat'), 'stats', '-v7.3');
                
                % 创建CSV报告
                obj.CreateCSVReport(export_dir);
                
                obj.Logger.Log('info', '采样结果已导出');
                
            catch ME
                obj.Logger.LogException(ME, 'BootstrapSampler.ExportSamples');
            end
        end
    end
    
    methods (Access = private)
        function AnalyzeLabelDistribution(obj, y)
            % 分析标签分布
            obj.SamplingStats.label_analysis = struct();
            
            % 计算基本统计
            unique_labels = unique(y);
            label_counts = histcounts(y, [unique_labels; unique_labels(end)+1]);
            label_props = label_counts / length(y);
            
            obj.SamplingStats.label_analysis.unique_labels = unique_labels;
            obj.SamplingStats.label_analysis.label_counts = label_counts;
            obj.SamplingStats.label_analysis.label_proportions = label_props;
            obj.SamplingStats.label_analysis.imbalance_ratio = max(label_counts) / min(label_counts);
            
            % 记录分布信息
            obj.Logger.Log('info', '标签分布分析:');
            for i = 1:length(unique_labels)
                obj.Logger.Log('info', sprintf('  标签 %d: %d 个样本 (%.1f%%)', ...
                    unique_labels(i), label_counts(i), label_props(i) * 100));
            end
            obj.Logger.Log('info', sprintf('  不平衡比: %.2f:1', obj.SamplingStats.label_analysis.imbalance_ratio));
        end
        
        function strategy = DetermineSamplingStrategy(obj, y)
            % 确定采样策略
            strategy = struct();
            
            % 基础策略
            strategy.type = 'stratified';  % 默认分层采样
            strategy.preserve_distribution = true;
            
            % 根据不平衡程度调整策略
            imbalance_ratio = obj.SamplingStats.label_analysis.imbalance_ratio;
            
            if imbalance_ratio > 5
                strategy.type = 'balanced_stratified';  % 平衡分层采样
                strategy.balancing_method = 'oversampling';
            elseif imbalance_ratio > 10
                strategy.type = 'balanced_stratified';
                strategy.balancing_method = 'hybrid';  % 混合方法
            end
            
            % 特殊处理小样本
            if length(y) < 100
                strategy.type = 'systematic';
                strategy.replace_on_rare = true;
            end
            
            % 设置额外参数
            strategy.bootstrap_perturbation = true;  % 启用扰动
            strategy.ensure_all_classes = true;      % 确保所有类别都存在
            
            obj.Logger.Log('info', sprintf('采样策略: %s', strategy.type));
            if isfield(strategy, 'balancing_method')
                obj.Logger.Log('info', sprintf('平衡方法: %s', strategy.balancing_method));
            end
        end
        
        function [train_indices, test_indices] = PerformSampling(obj, y, n_samples, train_ratio, strategy)
            % 执行采样
            
            % 预分配结果
            train_indices = cell(n_samples, 1);
            test_indices = cell(n_samples, 1);
            
            % 分析类别
            unique_labels = unique(y);
            label_indices = cell(length(unique_labels), 1);
            for i = 1:length(unique_labels)
                label_indices{i} = find(y == unique_labels(i));
            end
            
            % 预分配随机种子
            rng_seeds = randi(2^32-1, n_samples, 1);
            
            % 初始化进度统计
            sample_stats = struct();
            sample_stats.train_sizes = zeros(n_samples, 1);
            sample_stats.test_sizes = zeros(n_samples, 1);
            sample_stats.label_distributions = cell(n_samples, 1);
            
            % 并行采样
            parfor i = 1:n_samples
                % 设置随机种子
                rng(rng_seeds(i));
                
                % 根据策略执行采样
                [train_idx, test_idx] = obj.SampleSingleIteration(y, train_ratio, strategy, label_indices);
                
                % 存储结果
                train_indices{i} = train_idx;
                test_indices{i} = test_idx;
                
                % 记录统计信息
                sample_stats.train_sizes(i) = length(train_idx);
                sample_stats.test_sizes(i) = length(test_idx);
                sample_stats.label_distributions{i} = obj.CalculateLabelDistribution(y(train_idx));
                
                % 进度更新
                if mod(i, max(1, round(n_samples/10))) == 0
                    progress = i / n_samples * 100;
                end
            end
            
            % 保存采样统计
            obj.SampleStats = sample_stats;
            
            % 分析采样质量
            obj.AnalyzeSamplingQuality(sample_stats);
        end
        
        function [train_idx, test_idx] = SampleSingleIteration(obj, y, train_ratio, strategy, label_indices)
            % 执行单次采样
            
            train_idx = [];
            test_idx = [];
            
            switch strategy.type
                case 'stratified'
                    [train_idx, test_idx] = obj.StratifiedSample(y, train_ratio, label_indices);
                    
                case 'balanced_stratified'
                    [train_idx, test_idx] = obj.BalancedStratifiedSample(y, train_ratio, label_indices, strategy);
                    
                case 'systematic'
                    [train_idx, test_idx] = obj.SystematicSample(y, train_ratio, label_indices);
                    
                otherwise
                    error('未知的采样策略: %s', strategy.type);
            end
            
            % 应用扰动（如果启用）
            if strategy.bootstrap_perturbation
                train_idx = obj.ApplyBootstrapPerturbation(train_idx, y);
            end
            
            % 确保所有类别都存在
            if strategy.ensure_all_classes
                train_idx = obj.EnsureAllClassesPresent(train_idx, y, label_indices);
            end
            
            % 计算test_idx
            total_idx = 1:length(y);
            test_idx = setdiff(total_idx, train_idx);
        end
        
        function [train_idx, test_idx] = StratifiedSample(obj, y, train_ratio, label_indices)
            % 分层采样
            train_idx = [];
            
            for i = 1:length(label_indices)
                class_idx = label_indices{i};
                n_class = length(class_idx);
                n_train = round(train_ratio * n_class);
                
                % 保证至少有一个样本
                n_train = max(1, n_train);
                n_train = min(n_train, n_class);
                
                % 随机选择
                selected = randsample(n_class, n_train);
                train_idx = [train_idx; class_idx(selected)];
            end
            
            % 计算test_idx
            total_idx = 1:length(y);
            test_idx = setdiff(total_idx, train_idx);
        end
        
        function [train_idx, test_idx] = BalancedStratifiedSample(obj, y, train_ratio, label_indices, strategy)
            % 平衡分层采样
            
            % 计算每个类别的目标样本数
            n_classes = length(label_indices);
            class_sizes = cellfun(@length, label_indices);
            
            if strcmp(strategy.balancing_method, 'oversampling')
                % 上采样到最大类别的大小
                target_size_per_class = max(class_sizes);
            elseif strcmp(strategy.balancing_method, 'undersampling')
                % 下采样到最小类别的大小
                target_size_per_class = min(class_sizes);
            else  % hybrid
                % 平衡到中位数大小
                target_size_per_class = median(class_sizes);
            end
            
            % 调整目标大小以适应训练比例
            target_train_size = round(target_size_per_class * train_ratio);
            target_train_size = max(1, target_train_size);
            
            train_idx = [];
            
            for i = 1:length(label_indices)
                class_idx = label_indices{i};
                n_class = length(class_idx);
                
                if n_class >= target_train_size
                    % 下采样
                    selected = randsample(n_class, target_train_size, false);
                else
                    % 上采样（有放回抽样）
                    selected = randsample(n_class, target_train_size, true);
                end
                
                train_idx = [train_idx; class_idx(selected)];
            end
            
            % 计算test_idx
            total_idx = 1:length(y);
            test_idx = setdiff(total_idx, train_idx);
        end
        
        function [train_idx, test_idx] = SystematicSample(obj, y, train_ratio, label_indices)
            % 系统抽样（适用于小样本）
            
            n_total = length(y);
            n_train = round(train_ratio * n_total);
            
            % 确保合理范围
            n_train = max(round(n_total * 0.5), min(n_train, round(n_total * 0.9)));
            
            % 系统抽样
            step = round(n_total / n_train);
            step = max(1, step);
            
            start_idx = randi(step);
            train_idx = start_idx:step:n_total;
            train_idx = train_idx(train_idx <= n_total);
            
            % 确保训练集大小
            if length(train_idx) < n_train
                % 补充随机样本
                remaining = setdiff(1:n_total, train_idx);
                additional = randsample(remaining, n_train - length(train_idx), false);
                train_idx = [train_idx, additional];
            end
            
            % 计算test_idx
            total_idx = 1:n_total;
            test_idx = setdiff(total_idx, train_idx);
        end
        
        function perturbed_idx = ApplyBootstrapPerturbation(obj, train_idx, y)
            % 应用Bootstrap扰动
            
            n_train = length(train_idx);
            
            % 计算扰动比例（通常10-20%）
            perturbation_rate = 0.15;
            n_perturb = round(n_train * perturbation_rate);
            
            % 选择要替换的索引
            replace_idx = randsample(n_train, n_perturb, false);
            
            % 从原始训练集中重新采样
            new_samples = randsample(train_idx, n_perturb, true);
            
            % 创建扰动后的索引
            perturbed_idx = train_idx;
            perturbed_idx(replace_idx) = new_samples;
            
            % 去重
            perturbed_idx = unique(perturbed_idx);
        end
        
        function enhanced_idx = EnsureAllClassesPresent(obj, train_idx, y, label_indices)
            % 确保所有类别都在训练集中
            
            enhanced_idx = train_idx;
            
            % 检查每个类别
            for i = 1:length(label_indices)
                class_idx = label_indices{i};
                
                % 检查该类别是否存在于训练集中
                if ~any(ismember(train_idx, class_idx))
                    % 随机添加一个该类别的样本
                    random_sample = datasample(class_idx, 1);
                    enhanced_idx = [enhanced_idx; random_sample];
                end
            end
            
            % 去重
            enhanced_idx = unique(enhanced_idx);
        end
        
        function distribution = CalculateLabelDistribution(~, y_subset)
            % 计算子集的标签分布
            unique_labels = unique(y_subset);
            distribution = struct();
            
            for i = 1:length(unique_labels)
                label = unique_labels(i);
                count = sum(y_subset == label);
                distribution.(sprintf('label_%d', label)) = count;
                distribution.(sprintf('prop_%d', label)) = count / length(y_subset);
            end
        end
        
        function AnalyzeSamplingQuality(obj, sample_stats)
            % 分析采样质量
            
            quality_analysis = struct();
            
            % 训练集大小统计
            quality_analysis.train_size = struct();
            quality_analysis.train_size.mean = mean(sample_stats.train_sizes);
            quality_analysis.train_size.std = std(sample_stats.train_sizes);
            quality_analysis.train_size.min = min(sample_stats.train_sizes);
            quality_analysis.train_size.max = max(sample_stats.train_sizes);
            quality_analysis.train_size.cv = quality_analysis.train_size.std / quality_analysis.train_size.mean;
            
            % 测试集大小统计
            quality_analysis.test_size = struct();
            quality_analysis.test_size.mean = mean(sample_stats.test_sizes);
            quality_analysis.test_size.std = std(sample_stats.test_sizes);
            quality_analysis.test_size.min = min(sample_stats.test_sizes);
            quality_analysis.test_size.max = max(sample_stats.test_sizes);
            quality_analysis.test_size.cv = quality_analysis.test_size.std / quality_analysis.test_size.mean;
            
            % 标签分布一致性
            quality_analysis.distribution_stability = obj.AnalyzeDistributionStability(sample_stats.label_distributions);
            
            % 样本覆盖率（每个样本被选中的次数）
            quality_analysis.sample_coverage = obj.CalculateSampleCoverage(sample_stats);
            
            obj.SamplingStats.quality_analysis = quality_analysis;
            
            % 记录质量分析结果
            obj.Logger.Log('info', '采样质量分析:');
            obj.Logger.Log('info', sprintf('  训练集大小: %.1f ± %.1f (CV=%.3f)', ...
                quality_analysis.train_size.mean, quality_analysis.train_size.std, quality_analysis.train_size.cv));
            obj.Logger.Log('info', sprintf('  测试集大小: %.1f ± %.1f (CV=%.3f)', ...
                quality_analysis.test_size.mean, quality_analysis.test_size.std, quality_analysis.test_size.cv));
            obj.Logger.Log('info', sprintf('  分布稳定性: %.3f', quality_analysis.distribution_stability));
            obj.Logger.Log('info', sprintf('  平均样本覆盖率: %.1f%%', quality_analysis.sample_coverage.mean_coverage * 100));
        end
        
        function stability = AnalyzeDistributionStability(obj, distributions)
            % 分析标签分布的稳定性
            
            n_samples = length(distributions);
            if n_samples < 2
                stability = 1;
                return;
            end
            
            % 获取所有可能的标签
            all_labels = {};
            for i = 1:n_samples
                fields = fieldnames(distributions{i});
                for j = 1:length(fields)
                    if startswith(fields{j}, 'prop_')
                        if ~any(strcmp(all_labels, fields{j}))
                            all_labels{end+1} = fields{j};
                        end
                    end
                end
            end
            
            % 计算每个标签的分布稳定性
            stabilities = zeros(length(all_labels), 1);
            
            for i = 1:length(all_labels)
                label = all_labels{i};
                proportions = zeros(n_samples, 1);
                
                for j = 1:n_samples
                    if isfield(distributions{j}, label)
                        proportions(j) = distributions{j}.(label);
                    end
                end
                
                % 计算变异系数
                if mean(proportions) > 0
                    stabilities(i) = 1 - (std(proportions) / mean(proportions));
                else
                    stabilities(i) = 1;
                end
            end
            
            % 总体稳定性
            stability = mean(stabilities);
        end
        
        function coverage = CalculateSampleCoverage(obj, sample_stats)
            % 计算样本覆盖率
            
            % 汇总所有训练集索引
            all_train_idx = [];
            for i = 1:length(sample_stats.train_sizes)
                % 这里需要访问实际的索引，但由于是并行处理，我们用估计
                all_train_idx = [all_train_idx; ones(sample_stats.train_sizes(i), 1) * i];
            end
            
            % 计算每个样本的覆盖次数（近似）
            n_samples = obj.Config.NumBootstrapSamples;
            estimated_total_samples = sum(sample_stats.train_sizes) + sum(sample_stats.test_sizes);
            avg_sample_occurrence = n_samples * obj.Config.TrainRatio;
            
            coverage = struct();
            coverage.mean_coverage = avg_sample_occurrence / n_samples;
            coverage.estimated_coverage_rate = mean(sample_stats.train_sizes) / mean(sample_stats.train_sizes + sample_stats.test_sizes);
        end
        
        function CreateCSVReport(obj, output_dir)
            % 创建CSV格式的采样报告
            
            try
                % 基本统计报告
                stats = obj.SamplingStats;
                
                % 创建标签分布报告
                if isfield(stats, 'label_analysis')
                    label_table = table();
                    label_table.Label = stats.label_analysis.unique_labels;
                    label_table.Count = stats.label_analysis.label_counts';
                    label_table.Proportion = stats.label_analysis.label_proportions';
                    
                    writetable(label_table, fullfile(output_dir, 'label_distribution.csv'));
                end
                
                % 创建采样质量报告
                if isfield(stats, 'quality_analysis')
                    quality = stats.quality_analysis;
                    
                    quality_table = table();
                    quality_table.Metric = {'Train Size Mean', 'Train Size Std', 'Train Size CV', ...
                        'Test Size Mean', 'Test Size Std', 'Test Size CV', ...
                        'Distribution Stability', 'Mean Coverage'};
                    quality_table.Value = [quality.train_size.mean, quality.train_size.std, quality.train_size.cv, ...
                        quality.test_size.mean, quality.test_size.std, quality.test_size.cv, ...
                        quality.distribution_stability, quality.sample_coverage.mean_coverage];
                    
                    writetable(quality_table, fullfile(output_dir, 'sampling_quality.csv'));
                end
                
                obj.Logger.Log('info', 'CSV报告已生成');
                
            catch ME
                obj.Logger.LogException(ME, 'BootstrapSampler.CreateCSVReport');
            end
        end
        
        function GenerateSamplingReport(obj)
            % 生成采样报告
            
            report = struct();
            report.timestamp = datetime('now');
            report.config = struct();
            report.config.n_samples = obj.Config.NumBootstrapSamples;
            report.config.train_ratio = obj.Config.TrainRatio;
            
            if isfield(obj.SamplingStats, 'label_analysis')
                report.label_analysis = obj.SamplingStats.label_analysis;
            end
            
            if isfield(obj.SamplingStats, 'quality_analysis')
                report.quality_analysis = obj.SamplingStats.quality_analysis;
            end
            
            % 计算关键指标
            if isfield(obj.SampleStats, 'train_sizes')
                report.summary = struct();
                report.summary.avg_train_size = mean(obj.SampleStats.train_sizes);
                report.summary.avg_test_size = mean(obj.SampleStats.test_sizes);
                report.summary.size_stability = 1 - (std(obj.SampleStats.train_sizes) / mean(obj.SampleStats.train_sizes));
            end
            
            % 保存报告
            obj.SamplingStats.final_report = report;
            
            % 记录关键信息
            obj.Logger.CreateSection('Bootstrap采样报告');
            obj.Logger.Log('info', sprintf('总采样次数: %d', obj.Config.NumBootstrapSamples));
            obj.Logger.Log('info', sprintf('训练集比例: %.2f', obj.Config.TrainRatio));
            
            if isfield(report, 'summary')
                obj.Logger.Log('info', sprintf('平均训练集大小: %.1f', report.summary.avg_train_size));
                obj.Logger.Log('info', sprintf('平均测试集大小: %.1f', report.summary.avg_test_size));
                obj.Logger.Log('info', sprintf('大小稳定性: %.3f', report.summary.size_stability));
            end
            
            if isfield(report, 'quality_analysis')
                obj.Logger.Log('info', sprintf('分布稳定性: %.3f', report.quality_analysis.distribution_stability));
            end
        end
    end
end