classdef CrossValidator < handle
    % K折交叉验证器类：实现全面的交叉验证分析
    % 包括多种性能指标、系数稳定性分析和可视化
    
    properties (Access = private)
        Config
        Logger
        CVResults
        FoldStats
    end
    
    methods (Access = public)
        function obj = CrossValidator(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.CVResults = struct();
            obj.FoldStats = struct();
        end
        
        function cv_results = Validate(obj, X, y, var_names)
            % 执行K折交叉验证
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   var_names - 变量名称
            % 输出:
            %   cv_results - 交叉验证结果
            
            obj.Logger.Log('info', sprintf('开始%d折交叉验证', obj.Config.KFolds));
            
            try
                % 1. 准备交叉验证
                obj.PrepareCrossValidation(X, y);
                
                % 2. 执行折叠
                obj.ExecuteFolds(X, y, var_names);
                
                % 3. 计算总体指标
                obj.CalculateOverallMetrics();
                
                % 4. 分析系数稳定性
                obj.AnalyzeCoefficientStability();
                
                % 5. 进行统计检验
                obj.PerformStatisticalTests();
                
                % 6. 创建可视化
                obj.CreateCrossValidationVisualizations();
                
                % 7. 生成报告
                obj.GenerateCrossValidationReport();
                
                cv_results = obj.CVResults;
                
                obj.Logger.Log('info', sprintf('%d折交叉验证完成', obj.Config.KFolds));
                
            catch ME
                obj.Logger.LogException(ME, 'CrossValidator.Validate');
                rethrow(ME);
            end
        end
        
        function results = GetResults(obj)
            % 获取交叉验证结果
            results = obj.CVResults;
        end
        
        function ExportResults(obj, output_dir)
            % 导出交叉验证结果
            try
                % 创建导出目录
                cv_dir = fullfile(output_dir, 'cross_validation');
                if ~exist(cv_dir, 'dir')
                    mkdir(cv_dir);
                end
                
                % 保存结果
                results = obj.CVResults;
                save(fullfile(cv_dir, 'cv_results.mat'), 'results', '-v7.3');
                
                % 导出性能指标
                obj.ExportPerformanceMetrics(cv_dir);
                
                % 导出系数分析
                obj.ExportCoefficientAnalysis(cv_dir);
                
                % 导出预测结果
                obj.ExportPredictions(cv_dir);
                
                obj.Logger.Log('info', '交叉验证结果已导出');
                
            catch ME
                obj.Logger.LogException(ME, 'CrossValidator.ExportResults');
            end
        end
    end
    
    methods (Access = private)
        function PrepareCrossValidation(obj, X, y)
            % 准备交叉验证
            
            n_samples = length(y);
            k = obj.Config.KFolds;
            
            % 验证K值
            if k < 2
                error('K值必须大于等于2');
            end
            
            if k > n_samples
                obj.Logger.Log('warning', sprintf('K值(%d)大于样本数(%d)，调整为%d', k, n_samples, n_samples));
                k = n_samples;
                obj.Config.KFolds = k;  % 更新配置
            end
            
            % 创建分层交叉验证分组
            try
                cv = cvpartition(y, 'KFold', k, 'Stratify', true);
            catch
                obj.Logger.Log('warning', '分层创建失败，使用随机分组');
                cv = cvpartition(y, 'KFold', k);
            end
            
            % 保存分组信息
            obj.CVResults.cv_partition = cv;
            obj.CVResults.k_folds = k;
            obj.CVResults.n_samples = n_samples;
            
            % 初始化结果存储
            obj.InitializeResultsStorage(k);
            
            obj.Logger.Log('info', sprintf('准备%d折交叉验证，样本数：%d', k, n_samples));
        end
        
        function InitializeResultsStorage(obj, k)
            % 初始化结果存储结构
            
            % 性能指标
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', ...
                       'aic', 'bic', 'log_loss', 'brier_score'};
            
            for i = 1:length(metrics)
                obj.CVResults.(metrics{i}) = zeros(k, 1);
            end
            
            % 预测结果存储
            obj.CVResults.fold_indices = cell(k, 1);
            obj.CVResults.y_pred = cell(k, 1);
            obj.CVResults.y_test = cell(k, 1);
            obj.CVResults.y_pred_prob = cell(k, 1);
            
            % 系数存储
            obj.CVResults.coefficients = cell(k, 1);
            obj.CVResults.feature_importance = cell(k, 1);
            
            % 性能细节
            obj.CVResults.confusion_matrices = cell(k, 1);
            obj.CVResults.roc_curves = cell(k, 1);
            obj.CVResults.precision_recall_curves = cell(k, 1);
            
            % 折叠统计
            obj.FoldStats = struct();
            obj.FoldStats.training_time = zeros(k, 1);
            obj.FoldStats.prediction_time = zeros(k, 1);
            obj.FoldStats.convergence_info = cell(k, 1);
        end
        
        function ExecuteFolds(obj, X, y, var_names)
            % 执行所有折叠
            
            k = obj.Config.KFolds;
            cv = obj.CVResults.cv_partition;
            
            % 并行执行折叠
            fold_results = cell(k, 1);
            
            parfor i = 1:k
                fold_results{i} = obj.ExecuteSingleFold(i, X, y, var_names, cv);
            end
            
            % 合并结果
            obj.MergeFoldResults(fold_results);
            
            obj.Logger.Log('info', sprintf('完成%d折叠的模型训练和评估', k));
        end
        
        function fold_result = ExecuteSingleFold(obj, fold_idx, X, y, var_names, cv)
            % 执行单个折叠
            
            fold_result = struct();
            fold_start_time = tic;
            
            try
                % 获取训练和测试集
                train_idx = cv.training(fold_idx);
                test_idx = cv.test(fold_idx);
                
                X_train = X(train_idx, :);
                y_train = y(train_idx);
                X_test = X(test_idx, :);
                y_test = y(test_idx);
                
                % 存储索引
                fold_result.fold_indices = struct('train', find(train_idx), 'test', find(test_idx));
                
                % 训练模型
                training_start = tic;
                [model, model_info] = obj.TrainFoldModel(X_train, y_train);
                training_time = toc(training_start);
                
                % 进行预测
                prediction_start = tic;
                [y_pred, y_pred_prob] = obj.PredictFold(model, X_test);
                prediction_time = toc(prediction_start);
                
                % 计算性能指标
                metrics = obj.CalculateFoldMetrics(y_test, y_pred, y_pred_prob, model, X_train, y_train);
                
                % 存储结果
                fold_result.metrics = metrics;
                fold_result.y_pred = y_pred;
                fold_result.y_test = y_test;
                fold_result.y_pred_prob = y_pred_prob;
                fold_result.coefficients = obj.ExtractCoefficients(model, var_names);
                fold_result.feature_importance = obj.ExtractFeatureImportance(model, var_names);
                fold_result.confusion_matrix = obj.CalculateConfusionMatrix(y_test, y_pred);
                fold_result.roc_curve = obj.CalculateROCCurve(y_test, y_pred_prob);
                fold_result.pr_curve = obj.CalculatePRCurve(y_test, y_pred_prob);
                
                % 时间和收敛信息
                fold_result.training_time = training_time;
                fold_result.prediction_time = prediction_time;
                fold_result.convergence_info = model_info;
                
            catch ME
                obj.Logger.Log('warning', sprintf('折叠%d执行失败: %s', fold_idx, ME.message));
                
                % 返回空结果
                fold_result.metrics = struct();
                fold_result.errors = ME;
            end
            
            fold_result.total_time = toc(fold_start_time);
        end
        
        function [model, model_info] = TrainFoldModel(obj, X_train, y_train)
            % 训练单个折叠的模型
            
            model_info = struct();
            
            try
                % 使用逻辑回归模型
                mdl = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');
                
                % 检查收敛性
                model_info.converged = mdl.DispersionEstimated;
                model_info.iterations = mdl.NumIterations;
                model_info.deviance = mdl.Deviance;
                
                model = mdl;
                
            catch ME
                obj.Logger.Log('warning', sprintf('模型训练失败: %s', ME.message));
                model = [];
                model_info.error = ME;
            end
        end
        
        function [y_pred, y_pred_prob] = PredictFold(obj, model, X_test)
            % 对测试集进行预测
            
            try
                if ~isempty(model)
                    y_pred_prob = predict(model, X_test);
                    y_pred = y_pred_prob > 0.5;
                else
                    % 模型为空，返回随机预测
                    y_pred_prob = rand(size(X_test, 1), 1);
                    y_pred = y_pred_prob > 0.5;
                end
            catch
                % 预测失败，返回随机预测
                y_pred_prob = rand(size(X_test, 1), 1);
                y_pred = y_pred_prob > 0.5;
            end
        end
        
        function metrics = CalculateFoldMetrics(obj, y_test, y_pred, y_pred_prob, model, X_train, y_train)
            % 计算单个折叠的性能指标
            
            metrics = struct();
            
            try
                % 基本分类指标
                [metrics.accuracy, metrics.precision, metrics.recall, metrics.specificity, metrics.f1_score] = ...
                    obj.CalculateClassificationMetrics(y_test, y_pred);
                
                % AUC
                if length(unique(y_test)) > 1
                    [~, ~, ~, metrics.auc] = perfcurve(y_test, y_pred_prob, 1);
                else
                    metrics.auc = NaN;
                end
                
                % 模型选择指标
                if ~isempty(model)
                    [metrics.aic, metrics.bic] = obj.CalculateModelSelectionMetrics(model, X_train, y_train);
                else
                    metrics.aic = NaN;
                    metrics.bic = NaN;
                end
                
                % 概率指标
                metrics.log_loss = obj.CalculateLogLoss(y_test, y_pred_prob);
                metrics.brier_score = obj.CalculateBrierScore(y_test, y_pred_prob);
                
            catch ME
                obj.Logger.Log('warning', sprintf('指标计算失败: %s', ME.message));
                % 返回默认值
                metrics = obj.GetDefaultMetrics();
            end
        end
        
        function [accuracy, precision, recall, specificity, f1_score] = CalculateClassificationMetrics(obj, y_test, y_pred)
            % 计算分类指标
            
            % 计算混淆矩阵元素
            TP = sum(y_pred == 1 & y_test == 1);
            TN = sum(y_pred == 0 & y_test == 0);
            FP = sum(y_pred == 1 & y_test == 0);
            FN = sum(y_pred == 0 & y_test == 1);
            
            % 计算指标
            total = length(y_test);
            accuracy = (TP + TN) / total;
            
            if (TP + FP) > 0
                precision = TP / (TP + FP);
            else
                precision = 0;
            end
            
            if (TP + FN) > 0
                recall = TP / (TP + FN);
            else
                recall = 0;
            end
            
            if (TN + FP) > 0
                specificity = TN / (TN + FP);
            else
                specificity = 0;
            end
            
            if (precision + recall) > 0
                f1_score = 2 * precision * recall / (precision + recall);
            else
                f1_score = 0;
            end
        end
        
        function [aic, bic] = CalculateModelSelectionMetrics(obj, model, X_train, y_train)
            % 计算模型选择指标
            
            try
                % 提取模型信息
                deviance = model.Deviance;
                n_params = length(model.Coefficients.Estimate);
                n_samples = size(X_train, 1);
                
                % 计算AIC和BIC
                aic = deviance + 2 * n_params;
                bic = deviance + log(n_samples) * n_params;
                
            catch
                aic = Inf;
                bic = Inf;
            end
        end
        
        function log_loss = CalculateLogLoss(obj, y_true, y_prob)
            % 计算对数损失
            
            try
                % 避免边界情况
                eps = 1e-15;
                y_prob = max(min(y_prob, 1 - eps), eps);
                
                % 计算对数损失
                log_loss = -mean(y_true .* log(y_prob) + (1 - y_true) .* log(1 - y_prob));
                
            catch
                log_loss = NaN;
            end
        end
        
        function brier_score = CalculateBrierScore(obj, y_true, y_prob)
            % 计算Brier分数
            
            try
                brier_score = mean((y_prob - y_true).^2);
            catch
                brier_score = NaN;
            end
        end
        
        function coefs = ExtractCoefficients(obj, model, var_names)
            % 提取模型系数
            
            try
                if ~isempty(model)
                    coefs = table();
                    coefs.Variable = model.CoefficientNames';
                    coefs.Estimate = model.Coefficients.Estimate;
                    coefs.SE = model.Coefficients.SE;
                    coefs.tStat = model.Coefficients.tStat;
                    coefs.pValue = model.Coefficients.pValue;
                else
                    coefs = table();
                end
            catch
                coefs = table();
            end
        end
        
        function importance = ExtractFeatureImportance(obj, model, var_names)
            % 提取特征重要性
            
            try
                if ~isempty(model)
                    % 使用标准化系数的绝对值作为重要性
                    coefs = model.Coefficients.Estimate(2:end);  % 排除截距
                    
                    importance = table();
                    importance.Variable = var_names';
                    importance.Importance = abs(coefs);
                    importance = sortrows(importance, 'Importance', 'descend');
                else
                    importance = table();
                end
            catch
                importance = table();
            end
        end
        
        function cm = CalculateConfusionMatrix(obj, y_true, y_pred)
            % 计算混淆矩阵
            
            try
                cm = struct();
                
                % 计算元素
                cm.TN = sum(y_pred == 0 & y_true == 0);
                cm.FP = sum(y_pred == 1 & y_true == 0);
                cm.FN = sum(y_pred == 0 & y_true == 1);
                cm.TP = sum(y_pred == 1 & y_true == 1);
                
                % 创建矩阵
                cm.matrix = [cm.TN, cm.FP; cm.FN, cm.TP];
                
                % 归一化矩阵
                cm.normalized = cm.matrix ./ sum(cm.matrix(:));
                
            catch
                cm = struct();
            end
        end
        
        function roc = CalculateROCCurve(obj, y_true, y_scores)
            % 计算ROC曲线
            
            try
                [X, Y, T, AUC] = perfcurve(y_true, y_scores, 1);
                
                roc = struct();
                roc.FPR = X;
                roc.TPR = Y;
                roc.Thresholds = T;
                roc.AUC = AUC;
                
            catch
                roc = struct();
            end
        end
        
        function pr = CalculatePRCurve(obj, y_true, y_scores)
            % 计算Precision-Recall曲线
            
            try
                % 获取唯一阈值
                thresholds = unique(y_scores);
                thresholds = sort(thresholds, 'descend');
                
                precisions = zeros(length(thresholds), 1);
                recalls = zeros(length(thresholds), 1);
                
                for i = 1:length(thresholds)
                    y_pred = y_scores >= thresholds(i);
                    
                    TP = sum(y_pred == 1 & y_true == 1);
                    FP = sum(y_pred == 1 & y_true == 0);
                    FN = sum(y_pred == 0 & y_true == 1);
                    
                    if (TP + FP) > 0
                        precisions(i) = TP / (TP + FP);
                    else
                        precisions(i) = 1;
                    end
                    
                    if (TP + FN) > 0
                        recalls(i) = TP / (TP + FN);
                    else
                        recalls(i) = 0;
                    end
                end
                
                pr = struct();
                pr.Precision = precisions;
                pr.Recall = recalls;
                pr.Thresholds = thresholds;
                pr.AUC_PR = trapz(fliplr(recalls'), fliplr(precisions'));
                
            catch
                pr = struct();
            end
        end
        
        function MergeFoldResults(obj, fold_results)
            % 合并所有折叠的结果
            
            k = length(fold_results);
            
            % 合并性能指标
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', ...
                       'aic', 'bic', 'log_loss', 'brier_score'};
            
            for i = 1:k
                for j = 1:length(metrics)
                    metric = metrics{j};
                    if isfield(fold_results{i}, 'metrics') && isfield(fold_results{i}.metrics, metric)
                        obj.CVResults.(metric)(i) = fold_results{i}.metrics.(metric);
                    else
                        obj.CVResults.(metric)(i) = NaN;
                    end
                end
                
                % 合并其他结果
                fields = {'fold_indices', 'y_pred', 'y_test', 'y_pred_prob', 'coefficients', ...
                         'feature_importance', 'confusion_matrix', 'roc_curve', 'pr_curve'};
                
                for field = fields
                    field_name = field{1};
                    if isfield(fold_results{i}, field_name)
                        obj.CVResults.(field_name){i} = fold_results{i}.(field_name);
                    else
                        obj.CVResults.(field_name){i} = [];
                    end
                end
                
                % 合并统计信息
                if isfield(fold_results{i}, 'training_time')
                    obj.FoldStats.training_time(i) = fold_results{i}.training_time;
                end
                if isfield(fold_results{i}, 'prediction_time')
                    obj.FoldStats.prediction_time(i) = fold_results{i}.prediction_time;
                end
                if isfield(fold_results{i}, 'convergence_info')
                    obj.FoldStats.convergence_info{i} = fold_results{i}.convergence_info;
                end
            end
        end
        
        function CalculateOverallMetrics(obj)
            % 计算总体性能指标
            
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', ...
                       'aic', 'bic', 'log_loss', 'brier_score'};
            
            obj.CVResults.summary = struct();
            
            for i = 1:length(metrics)
                metric = metrics{i};
                values = obj.CVResults.(metric);
                
                % 计算统计量
                obj.CVResults.summary.(['avg_' metric]) = mean(values, 'omitnan');
                obj.CVResults.summary.(['std_' metric]) = std(values, 'omitnan');
                obj.CVResults.summary.(['cv_' metric]) = std(values, 'omitnan') / mean(values, 'omitnan');
                obj.CVResults.summary.(['min_' metric]) = min(values);
                obj.CVResults.summary.(['max_' metric]) = max(values);
                obj.CVResults.summary.(['median_' metric]) = median(values, 'omitnan');
            end
            
            % 计算置信区间
            obj.CalculateConfidenceIntervals();
            
            % 记录总体指标
            obj.Logger.Log('info', '总体交叉验证指标：');
            obj.Logger.Log('info', sprintf('准确率: %.3f ± %.3f', ...
                obj.CVResults.summary.avg_accuracy, obj.CVResults.summary.std_accuracy));
            obj.Logger.Log('info', sprintf('F1分数: %.3f ± %.3f', ...
                obj.CVResults.summary.avg_f1_score, obj.CVResults.summary.std_f1_score));
            obj.Logger.Log('info', sprintf('AUC: %.3f ± %.3f', ...
                obj.CVResults.summary.avg_auc, obj.CVResults.summary.std_auc));
        end
        
        function CalculateConfidenceIntervals(obj)
            % 计算置信区间
            
            metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
            alpha = 0.05;  % 95%置信区间
            
            obj.CVResults.confidence_intervals = struct();
            
            for i = 1:length(metrics)
                metric = metrics{i};
                values = obj.CVResults.(metric);
                n = sum(~isnan(values));
                
                if n > 1
                    % 使用t分布计算置信区间
                    t_stat = tinv(1 - alpha/2, n - 1);
                    mean_val = mean(values, 'omitnan');
                    std_val = std(values, 'omitnan');
                    margin = t_stat * std_val / sqrt(n);
                    
                    obj.CVResults.confidence_intervals.(metric) = [mean_val - margin, mean_val + margin];
                else
                    obj.CVResults.confidence_intervals.(metric) = [NaN, NaN];
                end
            end
        end
        
        function AnalyzeCoefficientStability(obj)
            % 分析系数稳定性
            
            try
                % 收集所有系数
                all_coefficients = obj.CVResults.coefficients;
                
                % 检查是否有有效的系数数据
                valid_coefs = cellfun(@(x) ~isempty(x) && height(x) > 0, all_coefficients);
                
                if sum(valid_coefs) < 2
                    obj.Logger.Log('warning', '系数数据不足，无法进行稳定性分析');
                    return;
                end
                
                % 获取所有变量名
                all_var_names = {};
                for i = 1:length(all_coefficients)
                    if valid_coefs(i)
                        vars = all_coefficients{i}.Variable;
                        all_var_names = union(all_var_names, vars);
                    end
                end
                
                % 创建系数矩阵
                coef_matrix = NaN(sum(valid_coefs), length(all_var_names));
                
                valid_idx = 1;
                for i = 1:length(all_coefficients)
                    if valid_coefs(i)
                        for j = 1:length(all_var_names)
                            var_name = all_var_names{j};
                            idx = strcmp(all_coefficients{i}.Variable, var_name);
                            if any(idx)
                                coef_matrix(valid_idx, j) = all_coefficients{i}.Estimate(idx);
                            end
                        end
                        valid_idx = valid_idx + 1;
                    end
                end
                
                % 计算稳定性指标
                stability = struct();
                stability.variable_names = all_var_names;
                stability.mean_coefficients = nanmean(coef_matrix, 1);
                stability.std_coefficients = nanstd(coef_matrix, 0, 1);
                stability.cv_coefficients = abs(stability.std_coefficients ./ stability.mean_coefficients);
                
                % 识别不稳定的系数
                stability.unstable_threshold = 0.5;
                stability.unstable_variables = all_var_names(stability.cv_coefficients > stability.unstable_threshold);
                
                obj.CVResults.coefficient_stability = stability;
                
                % 记录结果
                obj.Logger.Log('info', '系数稳定性分析完成：');
                obj.Logger.Log('info', sprintf('分析了 %d 个变量的系数', length(all_var_names)));
                obj.Logger.Log('info', sprintf('发现 %d 个不稳定变量 (CV > %.2f)', ...
                    length(stability.unstable_variables), stability.unstable_threshold));
                
            catch ME
                obj.Logger.LogException(ME, 'AnalyzeCoefficientStability');
            end
        end
        
        function PerformStatisticalTests(obj)
            % 执行统计检验
            
            obj.CVResults.statistical_tests = struct();
            
            try
                % 1. Friedman秩和检验（对不同指标）
                metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc'};
                metric_data = [];
                
                for i = 1:length(metrics)
                    metric_data = [metric_data, obj.CVResults.(metrics{i})];
                end
                
                % 执行Friedman检验
                [p_friedman, tbl_friedman, stats_friedman] = friedman(metric_data, 1, 'off');
                
                obj.CVResults.statistical_tests.friedman = struct();
                obj.CVResults.statistical_tests.friedman.p_value = p_friedman;
                obj.CVResults.statistical_tests.friedman.table = tbl_friedman;
                obj.CVResults.statistical_tests.friedman.stats = stats_friedman;
                
                % 2. 检验方差齐性
                obj.CVResults.statistical_tests.variance_homogeneity = struct();
                
                for i = 1:length(metrics)
                    metric = metrics{i};
                    values = obj.CVResults.(metric);
                    
                    % Levene's test（方差齐性）
                    if length(values) >= 3
                        [p_levene, stat_levene] = obj.LeveneTest(values);
                        obj.CVResults.statistical_tests.variance_homogeneity.(metric) = struct();
                        obj.CVResults.statistical_tests.variance_homogeneity.(metric).p_value = p_levene;
                        obj.CVResults.statistical_tests.variance_homogeneity.(metric).statistic = stat_levene;
                    end
                end
                
                % 3. 正态性检验
                obj.CVResults.statistical_tests.normality = struct();
                
                for i = 1:length(metrics)
                    metric = metrics{i};
                    values = obj.CVResults.(metric);
                    
                    % Kolmogorov-Smirnov检验
                    if length(values) >= 3
                        [h_ks, p_ks] = kstest(values);
                        obj.CVResults.statistical_tests.normality.(metric) = struct();
                        obj.CVResults.statistical_tests.normality.(metric).ks_p_value = p_ks;
                        obj.CVResults.statistical_tests.normality.(metric).ks_reject = h_ks;
                    end
                end
                
                obj.Logger.Log('info', '统计检验完成');
                
            catch ME
                obj.Logger.LogException(ME, 'PerformStatisticalTests');
            end
        end
        
        function [p_value, statistic] = LeveneTest(obj, data)
            % Levene方差齐性检验
            
            try
                % 这是一个简化版的Levene检验
                n = length(data);
                grand_mean = mean(data);
                
                % 计算组内偏差
                deviations = abs(data - grand_mean);
                
                % 计算F统计量
                mean_deviation = mean(deviations);
                ss_between = n * (mean_deviation - mean(deviations))^2;
                ss_within = sum((deviations - mean_deviation).^2);
                
                if ss_within > 0
                    statistic = ss_between / (ss_within / (n - 1));
                    % 使用F分布计算p值（简化）
                    p_value = 1 - fcdf(statistic, 1, n - 1);
                else
                    statistic = 0;
                    p_value = 1;
                end
                
            catch
                statistic = NaN;
                p_value = NaN;
            end
        end
        
        function CreateCrossValidationVisualizations(obj)
            % 创建交叉验证可视化
            
            try
                figure_dir = fullfile(obj.Config.OutputDirectory, 'figures', 'cross_validation');
                if ~exist(figure_dir, 'dir')
                    mkdir(figure_dir);
                end
                
                % 1. 折叠性能图
                obj.CreateFoldPerformancePlot(figure_dir);
                
                % 2. 指标分布图
                obj.CreateMetricDistributionPlot(figure_dir);
                
                % 3. 系数稳定性图
                obj.CreateCoefficientStabilityPlot(figure_dir);
                
                % 4. ROC曲线比较
                obj.CreateROCComparisonPlot(figure_dir);
                
                % 5. PR曲线比较
                obj.CreatePRComparisonPlot(figure_dir);
                
                % 6. 混淆矩阵热图
                obj.CreateConfusionMatrixHeatmap(figure_dir);
                
                obj.Logger.Log('info', '交叉验证可视化已创建');
                
            catch ME
                obj.Logger.LogException(ME, 'CreateCrossValidationVisualizations');
            end
        end
        
        function CreateFoldPerformancePlot(obj, figure_dir)
            % 创建折叠性能图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
                
                k = obj.Config.KFolds;
                metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
                metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
                
                for i = 1:length(metrics)
                    subplot(2, 3, i);
                    
                    metric = metrics{i};
                    values = obj.CVResults.(metric);
                    mean_val = obj.CVResults.summary.(['avg_' metric]);
                    std_val = obj.CVResults.summary.(['std_' metric]);
                    
                    % 绘制折线图
                    plot(1:k, values, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
                    hold on;
                    
                    % 绘制均值线
                    yline(mean_val, 'r--', 'LineWidth', 2);
                    
                    % 绘制标准差区间
                    fill([1:k, k:-1:1], [mean_val + std_val * ones(1, k), fliplr(mean_val - std_val * ones(1, k))], ...
                        'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
                    
                    xlabel('折数');
                    ylabel(metric_labels{i});
                    title(sprintf('%s （均值=%.3f, 标准差=%.3f）', metric_labels{i}, mean_val, std_val));
                    grid on;
                    xlim([0.5, k+0.5]);
                    
                    if i <= 5
                        ylim([0, 1]);
                    end
                end
                
                sgtitle('K折交叉验证各折性能指标', 'FontSize', 16, 'FontWeight', 'bold');
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'fold_performance.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'fold_performance.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateFoldPerformancePlot');
            end
        end
        
        function CreateMetricDistributionPlot(obj, figure_dir)
            % 创建指标分布图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 600]);
                
                metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
                metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
                
                data = zeros(obj.Config.KFolds, length(metrics));
                for i = 1:length(metrics)
                    data(:, i) = obj.CVResults.(metrics{i});
                end
                
                % 创建箱线图
                boxplot(data, 'Labels', metric_labels, 'Notch', 'on');
                
                ylabel('指标值');
                title('交叉验证指标分布');
                grid on;
                ylim([0, 1.05]);
                
                % 添加均值点
                means = mean(data);
                hold on;
                scatter(1:length(metrics), means, 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
                legend('箱线图', '均值', 'Location', 'southwest');
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'metric_distribution.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'metric_distribution.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateMetricDistributionPlot');
            end
        end
        
        function CreateCoefficientStabilityPlot(obj, figure_dir)
            % 创建系数稳定性图
            
            try
                if ~isfield(obj.CVResults, 'coefficient_stability')
                    return;
                end
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 600]);
                
                stability = obj.CVResults.coefficient_stability;
                
                % 排序变量
                [sorted_cv, idx] = sort(stability.cv_coefficients, 'descend');
                sorted_vars = stability.variable_names(idx);
                sorted_means = stability.mean_coefficients(idx);
                
                % 限制显示的变量数量
                max_vars = min(15, length(sorted_vars));
                sorted_cv = sorted_cv(1:max_vars);
                sorted_vars = sorted_vars(1:max_vars);
                sorted_means = sorted_means(1:max_vars);
                
                % 创建条形图
                bar(sorted_cv);
                hold on;
                
                % 添加阈值线
                yline(stability.unstable_threshold, 'r--', '不稳定阈值', 'LineWidth', 2);
                
                % 设置标签
                set(gca, 'XTick', 1:max_vars, 'XTickLabel', sorted_vars, 'XTickLabelRotation', 45);
                ylabel('变异系数 (CV)');
                xlabel('变量');
                title('系数稳定性分析');
                grid on;
                
                % 添加系数值标签
                for i = 1:max_vars
                    text(i, sorted_cv(i) + 0.01, sprintf('%.3f', sorted_means(i)), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                end
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'coefficient_stability.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'coefficient_stability.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateCoefficientStabilityPlot');
            end
        end
        
        function CreateROCComparisonPlot(obj, figure_dir)
            % 创建ROC曲线比较图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 800, 800]);
                
                colors = lines(obj.Config.KFolds);
                
                % 绘制每个折叠的ROC曲线
                for i = 1:obj.Config.KFolds
                    roc = obj.CVResults.roc_curves{i};
                    if ~isempty(roc) && isfield(roc, 'FPR') && isfield(roc, 'TPR')
                        plot(roc.FPR, roc.TPR, 'Color', colors(i, :), 'LineWidth', 1, ...
                            'DisplayName', sprintf('折叠 %d (AUC=%.3f)', i, roc.AUC));
                        hold on;
                    end
                end
                
                % 绘制平均ROC曲线
                avg_auc = obj.CVResults.summary.avg_auc;
                plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5, 'DisplayName', '随机分类器');
                
                xlabel('假阳性率');
                ylabel('真阳性率');
                title(sprintf('K折交叉验证ROC曲线比较 (平均AUC=%.3f±%.3f)', ...
                    avg_auc, obj.CVResults.summary.std_auc));
                legend('Location', 'southeast');
                grid on;
                axis equal;
                xlim([0, 1]);
                ylim([0, 1]);
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'roc_comparison.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'roc_comparison.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateROCComparisonPlot');
            end
        end
        
        function CreatePRComparisonPlot(obj, figure_dir)
            % 创建PR曲线比较图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 800, 800]);
                
                colors = lines(obj.Config.KFolds);
                
                % 绘制每个折叠的PR曲线
                for i = 1:obj.Config.KFolds
                    pr = obj.CVResults.precision_recall_curves{i};
                    if ~isempty(pr) && isfield(pr, 'Recall') && isfield(pr, 'Precision')
                        plot(pr.Recall, pr.Precision, 'Color', colors(i, :), 'LineWidth', 1, ...
                            'DisplayName', sprintf('折叠 %d (AUC=%.3f)', i, pr.AUC_PR));
                        hold on;
                    end
                end
                
                xlabel('召回率');
                ylabel('精确率');
                title('K折交叉验证Precision-Recall曲线比较');
                legend('Location', 'southwest');
                grid on;
                xlim([0, 1]);
                ylim([0, 1.05]);
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'pr_comparison.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'pr_comparison.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreatePRComparisonPlot');
            end
        end
        
        function CreateConfusionMatrixHeatmap(obj, figure_dir)
            % 创建混淆矩阵热图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 400]);
                
                % 计算平均混淆矩阵
                avg_cm = zeros(2, 2);
                count = 0;
                
                for i = 1:obj.Config.KFolds
                    cm = obj.CVResults.confusion_matrices{i};
                    if ~isempty(cm) && isfield(cm, 'matrix')
                        avg_cm = avg_cm + cm.matrix;
                        count = count + 1;
                    end
                end
                
                if count > 0
                    avg_cm = avg_cm / count;
                end
                
                % 创建三个子图：原始值、行归一化、列归一化
                subplot(1, 3, 1);
                heatmap(avg_cm, 'XDisplayLabels', {'预测0', '预测1'}, ...
                    'YDisplayLabels', {'实际0', '实际1'}, 'Title', '平均混淆矩阵（原始计数）');
                
                subplot(1, 3, 2);
                row_normalized = avg_cm ./ sum(avg_cm, 2);
                heatmap(row_normalized, 'XDisplayLabels', {'预测0', '预测1'}, ...
                    'YDisplayLabels', {'实际0', '实际1'}, 'Title', '行归一化混淆矩阵');
                
                subplot(1, 3, 3);
                col_normalized = avg_cm ./ sum(avg_cm, 1);
                heatmap(col_normalized, 'XDisplayLabels', {'预测0', '预测1'}, ...
                    'YDisplayLabels', {'实际0', '实际1'}, 'Title', '列归一化混淆矩阵');
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'confusion_matrix_heatmap.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'confusion_matrix_heatmap.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateConfusionMatrixHeatmap');
            end
        end
        
        function GenerateCrossValidationReport(obj)
            % 生成交叉验证报告
            
            try
                report = struct();
                report.timestamp = datetime('now');
                report.k_folds = obj.Config.KFolds;
                report.summary = obj.CVResults.summary;
                
                % 添加置信区间
                report.confidence_intervals = obj.CVResults.confidence_intervals;
                
                % 添加系数稳定性摘要
                if isfield(obj.CVResults, 'coefficient_stability')
                    report.coefficient_stability = struct();
                    report.coefficient_stability.n_variables = length(obj.CVResults.coefficient_stability.variable_names);
                    report.coefficient_stability.n_unstable = length(obj.CVResults.coefficient_stability.unstable_variables);
                    report.coefficient_stability.unstable_variables = obj.CVResults.coefficient_stability.unstable_variables;
                end
                
                % 添加统计检验结果
                if isfield(obj.CVResults, 'statistical_tests')
                    report.statistical_tests = obj.CVResults.statistical_tests;
                end
                
                % 保存报告
                obj.CVResults.final_report = report;
                
                % 记录关键发现
                obj.Logger.CreateSection('交叉验证报告摘要');
                obj.Logger.Log('info', sprintf('K值: %d', report.k_folds));
                
                % 记录主要指标
                metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc'};
                metric_labels = {'准确率', '精确率', '召回率', 'F1分数', 'AUC'};
                
                for i = 1:length(metrics)
                    metric = metrics{i};
                    avg = report.summary.(['avg_' metric]);
                    std = report.summary.(['std_' metric]);
                    ci = report.confidence_intervals.(metric);
                    
                    obj.Logger.Log('info', sprintf('%s: %.3f ± %.3f (95%% CI: [%.3f, %.3f])', ...
                        metric_labels{i}, avg, std, ci(1), ci(2)));
                end
                
                % 记录系数稳定性
                if isfield(report, 'coefficient_stability')
                    obj.Logger.Log('info', sprintf('系数稳定性: %d/%d 变量不稳定', ...
                        report.coefficient_stability.n_unstable, report.coefficient_stability.n_variables));
                end
                
            catch ME
                obj.Logger.LogException(ME, 'GenerateCrossValidationReport');
            end
        end
        
        function ExportPerformanceMetrics(obj, output_dir)
            % 导出性能指标
            
            try
                % 创建性能指标表
                metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', 'aic', 'bic'};
                k = obj.Config.KFolds;
                
                performance_table = table();
                performance_table.Fold = (1:k)';
                
                for i = 1:length(metrics)
                    metric = metrics{i};
                    performance_table.(metric) = obj.CVResults.(metric);
                end
                
                % 添加汇总行
                summary_row = table();
                summary_row.Fold = 0;  % 使用0表示汇总行
                
                for i = 1:length(metrics)
                    metric = metrics{i};
                    summary_row.(metric) = obj.CVResults.summary.(['avg_' metric]);
                end
                
                performance_table = [performance_table; summary_row];
                
                % 保存表格
                writetable(performance_table, fullfile(output_dir, 'performance_metrics.csv'));
                
                % 创建置信区间表
                ci_table = table();
                ci_fields = fieldnames(obj.CVResults.confidence_intervals);
                
                for i = 1:length(ci_fields)
                    field = ci_fields{i};
                    ci = obj.CVResults.confidence_intervals.(field);
                    ci_table.Metric{i} = field;
                    ci_table.Lower_95CI(i) = ci(1);
                    ci_table.Upper_95CI(i) = ci(2);
                end
                
                writetable(ci_table, fullfile(output_dir, 'confidence_intervals.csv'));
                
            catch ME
                obj.Logger.LogException(ME, 'ExportPerformanceMetrics');
            end
        end
        
        function ExportCoefficientAnalysis(obj, output_dir)
            % 导出系数分析结果
            
            try
                if ~isfield(obj.CVResults, 'coefficient_stability')
                    return;
                end
                
                stability = obj.CVResults.coefficient_stability;
                
                % 创建系数稳定性表
                coef_table = table();
                coef_table.Variable = stability.variable_names';
                coef_table.Mean_Coefficient = stability.mean_coefficients';
                coef_table.Std_Coefficient = stability.std_coefficients';
                coef_table.CV = stability.cv_coefficients';
                coef_table.Unstable = stability.cv_coefficients' > stability.unstable_threshold;
                
                % 按变异系数排序
                coef_table = sortrows(coef_table, 'CV', 'descend');
                
                writetable(coef_table, fullfile(output_dir, 'coefficient_stability.csv'));
                
            catch ME
                obj.Logger.LogException(ME, 'ExportCoefficientAnalysis');
            end
        end
        
        function ExportPredictions(obj, output_dir)
            % 导出预测结果
            
            try
                % 创建预测结果目录
                pred_dir = fullfile(output_dir, 'predictions');
                if ~exist(pred_dir, 'dir')
                    mkdir(pred_dir);
                end
                
                % 导出每个折叠的预测
                for i = 1:obj.Config.KFolds
                    if ~isempty(obj.CVResults.y_test{i}) && ~isempty(obj.CVResults.y_pred{i})
                        pred_table = table();
                        pred_table.True_Label = obj.CVResults.y_test{i};
                        pred_table.Predicted_Label = obj.CVResults.y_pred{i};
                        pred_table.Predicted_Probability = obj.CVResults.y_pred_prob{i};
                        
                        filename = sprintf('fold_%d_predictions.csv', i);
                        writetable(pred_table, fullfile(pred_dir, filename));
                    end
                end
                
                % 合并所有预测
                all_pred_table = table();
                all_true = [];
                all_pred = [];
                all_prob = [];
                all_fold = [];
                
                for i = 1:obj.Config.KFolds
                    if ~isempty(obj.CVResults.y_test{i})
                        all_true = [all_true; obj.CVResults.y_test{i}];
                        all_pred = [all_pred; obj.CVResults.y_pred{i}];
                        all_prob = [all_prob; obj.CVResults.y_pred_prob{i}];
                        all_fold = [all_fold; ones(length(obj.CVResults.y_test{i}), 1) * i];
                    end
                end
                
                all_pred_table.Fold = all_fold;
                all_pred_table.True_Label = all_true;
                all_pred_table.Predicted_Label = all_pred;
                all_pred_table.Predicted_Probability = all_prob;
                
                writetable(all_pred_table, fullfile(output_dir, 'all_predictions.csv'));
                
            catch ME
                obj.Logger.LogException(ME, 'ExportPredictions');
            end
        end
        
        function metrics = GetDefaultMetrics(obj)
            % 获取默认指标值
            metrics = struct();
            metrics.accuracy = 0;
            metrics.precision = 0;
            metrics.recall = 0;
            metrics.specificity = 0;
            metrics.f1_score = 0;
            metrics.auc = 0.5;
            metrics.aic = Inf;
            metrics.bic = Inf;
            metrics.log_loss = Inf;
            metrics.brier_score = 1;
        end
    end
end