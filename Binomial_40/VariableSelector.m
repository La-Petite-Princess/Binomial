classdef VariableSelector < handle
    % 变量选择器类：实现多种变量选择算法
    % 包括逐步回归、LASSO、Ridge、ElasticNet和随机森林
    
    properties (Access = private)
        Config
        Logger
        ParallelManager
        SelectionResults
    end
    
    properties (Access = public)
        Methods
        Results
    end
    
    methods (Access = public)
        function obj = VariableSelector(config, logger, parallel_manager)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.ParallelManager = parallel_manager;
            obj.Methods = config.VariableSelectionMethods;
            obj.SelectionResults = struct();
            obj.Results = struct();
        end
        
        function results = SelectVariables(obj, X, y, train_indices, test_indices, var_names)
            % 执行变量选择
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   train_indices - 训练集索引（cell数组）
            %   test_indices - 测试集索引（cell数组）
            %   var_names - 变量名称
            % 输出:
            %   results - 选择结果结构体
            
            obj.Logger.Log('info', '开始变量选择分析');
            
            try
                % 1. 准备并行任务
                tasks = obj.PrepareSelectionTasks(X, y, train_indices, test_indices, var_names);
                
                % 2. 并行执行各种方法
                obj.Logger.Log('info', '并行执行各种变量选择方法');
                method_futures = obj.ParallelManager.RunAsyncTask(@obj.ProcessMethod, tasks, 'VariableSelection');
                
                % 3. 收集结果
                method_results = obj.ParallelManager.WaitForAsyncResults(method_futures, 'VariableSelection');
                
                % 4. 处理和整合结果
                obj.IntegrateResults(method_results);
                
                % 5. 执行元分析
                obj.PerformMetaAnalysis();
                
                % 6. 创建可视化
                obj.CreateSelectionVisualizations();
                
                % 7. 生成报告
                obj.GenerateSelectionReport();
                
                results = obj.Results;
                
                obj.Logger.Log('info', '变量选择分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'VariableSelector.SelectVariables');
                rethrow(ME);
            end
        end
        
        function SaveResults(obj, output_dir)
            % 保存变量选择结果
            try
                % 创建保存目录
                var_sel_dir = fullfile(output_dir, 'variable_selection');
                if ~exist(var_sel_dir, 'dir')
                    mkdir(var_sel_dir);
                end
                
                % 保存主结果
                results = obj.Results;
                save(fullfile(var_sel_dir, 'variable_selection_results.mat'), 'results', '-v7.3');
                
                % 保存各方法的详细结果
                for i = 1:length(obj.Methods)
                    method = obj.Methods{i};
                    if isfield(obj.Results, method)
                        method_results = obj.Results.(method);
                        save(fullfile(var_sel_dir, sprintf('%s_results.mat', method)), 'method_results', '-v7.3');
                    end
                end
                
                % 导出CSV表格
                obj.ExportVariableSelectionTables(var_sel_dir);
                
                obj.Logger.Log('info', '变量选择结果已保存');
                
            catch ME
                obj.Logger.LogException(ME, 'VariableSelector.SaveResults');
            end
        end
    end
    
    methods (Access = private)
        function tasks = PrepareSelectionTasks(obj, X, y, train_indices, test_indices, var_names)
            % 准备变量选择任务
            
            tasks = cell(length(obj.Methods), 1);
            
            for i = 1:length(obj.Methods)
                task = struct();
                task.method = obj.Methods{i};
                task.X = X;
                task.y = y;
                task.train_indices = train_indices;
                task.test_indices = test_indices;
                task.var_names = var_names;
                task.config = obj.Config;
                
                tasks{i} = task;
            end
        end
        
        function result = ProcessMethod(obj, task)
            % 处理单个变量选择方法
            
            method = task.method;
            result = struct();
            result.method = method;
            
            try
                obj.Logger.Log('info', sprintf('开始处理%s方法', method));
                
                % 1. 执行变量选择
                [selected_vars, var_freq, var_combinations] = obj.SelectVariablesByMethod(task);
                
                % 2. 训练和评估模型
                [models, performance, group_performance] = obj.TrainAndEvaluateModels(task, var_combinations);
                
                % 3. 提取模型参数
                params = obj.ExtractModelParameters(models, task.var_names);
                
                % 4. 组织结果
                result.selected_vars = selected_vars;
                result.var_freq = var_freq;
                result.var_combinations = var_combinations;
                result.models = models;
                result.performance = performance;
                result.group_performance = group_performance;
                result.params = params;
                result.success = true;
                
                obj.Logger.Log('info', sprintf('%s方法处理完成', method));
                
            catch ME
                obj.Logger.LogException(ME, sprintf('ProcessMethod: %s', method));
                result.success = false;
                result.error = ME;
            end
        end
        
        function [selected_vars, var_freq, var_combinations] = SelectVariablesByMethod(obj, task)
            % 根据方法选择变量
            
            method = task.method;
            X = task.X;
            y = task.y;
            train_indices = task.train_indices;
            
            n_samples = length(train_indices);
            n_vars = size(X, 2);
            
            % 预分配
            var_combinations = cell(n_samples, 1);
            
            % 设置并行选项
            opts = obj.ParallelManager.GetParallelOptions();
            
            switch lower(method)
                case 'stepwise'
                    var_combinations = obj.SelectStepwise(X, y, train_indices, task.config, opts);
                    
                case 'lasso'
                    var_combinations = obj.SelectLASSO(X, y, train_indices, task.config, opts);
                    
                case 'ridge'
                    var_combinations = obj.SelectRidge(X, y, train_indices, task.config, opts);
                    
                case 'elasticnet'
                    var_combinations = obj.SelectElasticNet(X, y, train_indices, task.config, opts);
                    
                case 'randomforest'
                    var_combinations = obj.SelectRandomForest(X, y, train_indices, task.config, opts);
                    
                otherwise
                    error('不支持的变量选择方法: %s', method);
            end
            
            % 计算变量选择频率
            var_freq = obj.CalculateVariableFrequency(var_combinations, n_vars);
            
            % 确定最终选择的变量
            selected_vars = obj.DetermineSelectedVariables(var_combinations, var_freq, n_vars);
        end
        
        function var_combinations = SelectStepwise(obj, X, y, train_indices, config, opts)
            % 逐步回归变量选择
            
            n_samples = length(train_indices);
            var_combinations = cell(n_samples, 1);
            
            parfor i = 1:n_samples
                try
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    
                    % 执行逐步回归
                    [~, ~, ~, inmodel] = stepwisefit(X_train, y_train, ...
                        'PEnter', config.StepwisePEnter, ...
                        'PRemove', config.StepwisePRemove, ...
                        'Display', 'off');
                    
                    var_combinations{i} = find(inmodel);
                    
                catch
                    % 如果失败，使用相关性选择
                    [~, pval] = corr(X_train, y_train);
                    var_combinations{i} = find(pval < 0.05);
                end
            end
        end
        
        function var_combinations = SelectLASSO(obj, X, y, train_indices, config, opts)
            % LASSO变量选择
            
            n_samples = length(train_indices);
            var_combinations = cell(n_samples, 1);
            
            % 设置lambda范围
            lambda_range = logspace(-5, 1, 50);
            
            parfor i = 1:n_samples
                try
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    
                    % 标准化数据
                    [X_train_norm, mu, sigma] = zscore(X_train);
                    
                    % 执行LASSO
                    [B, FitInfo] = lasso(X_train_norm, y_train, ...
                        'CV', config.LassoCrossValidationFolds, ...
                        'Alpha', config.LassoAlpha, ...
                        'Lambda', lambda_range, ...
                        'Options', opts);
                    
                    % 选择最优lambda
                    lambda_min = FitInfo.LambdaMinMSE;
                    idx_min = FitInfo.Lambda == lambda_min;
                    coef = B(:, idx_min);
                    
                    var_combinations{i} = find(abs(coef) > 0);
                    
                catch
                    % 失败时的备选方法
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    [~, pval] = corr(X_train, y_train);
                    var_combinations{i} = find(pval < 0.05);
                end
                
                % 确保至少有一个变量
                if isempty(var_combinations{i})
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    [~, idx] = sort(abs(corr(X_train, y_train)), 'descend');
                    var_combinations{i} = idx(1:min(3, length(idx)));
                end
            end
        end
        
        function var_combinations = SelectRidge(obj, X, y, train_indices, config, opts)
            % Ridge回归变量选择
            
            n_samples = length(train_indices);
            var_combinations = cell(n_samples, 1);
            
            lambda_range = logspace(-5, 1, 50);
            
            parfor i = 1:n_samples
                try
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    
                    % 执行Ridge回归
                    [~, FitInfo] = lasso(X_train, y_train, ...
                        'CV', config.LassoCrossValidationFolds, ...
                        'Alpha', config.RidgeAlpha, ...
                        'Lambda', lambda_range, ...
                        'Options', opts);
                    
                    lambda_min = FitInfo.LambdaMinMSE;
                    
                    % 使用ridge函数获取系数
                    B = ridge(y_train, X_train, lambda_min, 0);
                    
                    % 选择重要变量（排除截距）
                    coef = B(2:end);
                    threshold = max(0.05, std(coef) * 0.1);
                    var_combinations{i} = find(abs(coef) > threshold);
                    
                catch
                    % 失败时的备选方法
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    mdl = fitglm(X_train, y_train, 'Distribution', 'binomial');
                    pvals = mdl.Coefficients.pValue(2:end);
                    var_combinations{i} = find(pvals < 0.05);
                end
                
                % 确保至少有一个变量
                if isempty(var_combinations{i})
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    [~, idx] = sort(abs(corr(X_train, y_train)), 'descend');
                    var_combinations{i} = idx(1:min(3, length(idx)));
                end
            end
        end
        
        function var_combinations = SelectElasticNet(obj, X, y, train_indices, config, opts)
            % ElasticNet变量选择
            
            n_samples = length(train_indices);
            var_combinations = cell(n_samples, 1);
            
            lambda_range = logspace(-5, 1, 50);
            
            parfor i = 1:n_samples
                try
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    
                    % 执行ElasticNet
                    [B, FitInfo] = lasso(X_train, y_train, ...
                        'CV', config.LassoCrossValidationFolds, ...
                        'Alpha', config.ElasticNetAlpha, ...
                        'Lambda', lambda_range, ...
                        'Options', opts);
                    
                    lambda_min = FitInfo.LambdaMinMSE;
                    idx_min = FitInfo.Lambda == lambda_min;
                    coef = B(:, idx_min);
                    
                    var_combinations{i} = find(abs(coef) > 0);
                    
                catch
                    % 失败时的备选方法
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    [~, pval] = corr(X_train, y_train);
                    var_combinations{i} = find(pval < 0.05);
                end
                
                % 确保至少有一个变量
                if isempty(var_combinations{i})
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    [~, idx] = sort(abs(corr(X_train, y_train)), 'descend');
                    var_combinations{i} = idx(1:min(3, length(idx)));
                end
            end
        end
        
        function var_combinations = SelectRandomForest(obj, X, y, train_indices, config, opts)
            % 随机森林变量选择
            
            n_samples = length(train_indices);
            var_combinations = cell(n_samples, 1);
            
            parfor i = 1:n_samples
                try
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    
                    if exist('TreeBagger', 'file')
                        % 创建随机森林
                        rf_model = TreeBagger(config.RandomForestNumTrees, X_train, y_train, ...
                            'Method', 'classification', ...
                            'OOBPrediction', 'on', ...
                            'OOBPredictorImportance', 'on', ...
                            'MinLeafSize', max(1, floor(size(X_train, 1) / 50)), ...
                            'NumPredictorsToSample', max(1, floor(sqrt(size(X_train, 2)))), ...
                            'Options', opts, ...
                            'PredictorSelection', 'curvature', ...
                            'MaxNumSplits', 1e4, ...
                            'Surrogate', 'off');
                        
                        % 获取特征重要性
                        importance = rf_model.OOBPermutedPredictorDeltaError;
                        
                        % 选择重要变量
                        var_combinations{i} = find(importance > mean(importance));
                        
                        % 移除clear语句，让变量自然地在迭代结束时被清理
                        % 不使用 clear rf; 
                    else
                        % 备选方法：相关性选择
                        [~, pval] = corr(X_train, y_train);
                        var_combinations{i} = find(pval < 0.05);
                    end
                    
                    % 确保至少有一个变量
                    if isempty(var_combinations{i})
                        [~, idx] = sort(abs(corr(X_train, y_train)), 'descend');
                        var_combinations{i} = idx(1:min(3, length(idx)));
                    end
                catch
                    % 失败时的备选方法
                    X_train = X(train_indices{i}, :);
                    y_train = y(train_indices{i});
                    [~, pval] = corr(X_train, y_train);
                    var_combinations{i} = find(pval < 0.05);
                    
                    % 确保至少有一个变量
                    if isempty(var_combinations{i})
                        [~, idx] = sort(abs(corr(X_train, y_train)), 'descend');
                        var_combinations{i} = idx(1:min(3, length(idx)));
                    end
                end
            end
        end
        
        function var_freq = CalculateVariableFrequency(obj, var_combinations, n_vars)
            % 计算变量选择频率
            
            n_samples = length(var_combinations);
            var_freq = zeros(n_vars, 1);
            
            for i = 1:n_samples
                if ~isempty(var_combinations{i})
                    selected = false(n_vars, 1);
                    selected(var_combinations{i}) = true;
                    var_freq = var_freq + selected;
                end
            end
            
            var_freq = var_freq / n_samples;
        end
        
        function selected_vars = DetermineSelectedVariables(obj, var_combinations, var_freq, n_vars)
            % 确定最终选择的变量
            
            % 找出最常见的变量组合
            combo_strings = cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false);
            [unique_combos, ~, ic] = unique(combo_strings);
            combo_counts = accumarray(ic, 1);
            
            % 使用最频繁的组合
            [~, max_idx] = max(combo_counts);
            most_frequent_combo = unique_combos{max_idx};
            
            % 从字符串中提取变量索引
            combo_indices = str2num(['[' most_frequent_combo(1:end-1) ']']);
            
            % 创建选择向量
            selected_vars = false(n_vars, 1);
            if ~isempty(combo_indices)
                selected_vars(combo_indices) = true;
            end
            
            obj.Logger.Log('info', sprintf('最常见的变量组合出现%d次，占比%.2f%%', ...
                combo_counts(max_idx), 100*combo_counts(max_idx)/length(var_combinations)));
        end
        
        function [models, performance, group_performance] = TrainAndEvaluateModels(obj, task, var_combinations)
            % 训练和评估模型
            
            X = task.X;
            y = task.y;
            train_indices = task.train_indices;
            test_indices = task.test_indices;
            var_names = task.var_names;
            method = task.method;
            
            n_samples = length(train_indices);
            
            % 预分配
            models = cell(n_samples, 1);
            metrics_names = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc', 'aic', 'bic'};
            
            % 预测结果存储
            y_pred_all = cell(n_samples, 1);
            y_test_all = cell(n_samples, 1);
            y_pred_prob_all = cell(n_samples, 1);
            all_coefs = cell(n_samples, 1);
            
            % 创建临时数组来存储性能指标，而不是直接修改结构体
            temp_metrics = zeros(n_samples, length(metrics_names));
            
            % 组合性能初始化
            perf_template = struct(...
                'accuracy', 0, 'sensitivity', 0, 'specificity', 0, ...
                'precision', 0, 'f1_score', 0, 'auc', 0, ...
                'aic', 0, 'bic', 0, 'count', 0, 'variables', {{}});
            
            combo_keys = cell(n_samples, 1);
            perf_structs = cell(n_samples, 1);
            
            % 并行训练和评估
            parfor i = 1:n_samples
                try
                    % 获取训练和测试集
                    train_idx = train_indices{i};
                    test_idx = test_indices{i};
                    selected_vars = var_combinations{i};
                    
                    if isempty(selected_vars)
                        models{i} = [];
                        continue;
                    end
                    
                    X_selected = X(:, selected_vars);
                    
                    % 训练模型
                    local_mdl = obj.TrainModel(X_selected(train_idx, :), y(train_idx), method);
                    models{i} = local_mdl;
                    
                    if ~isempty(local_mdl)
                        % 预测
                        [y_pred, y_pred_prob] = obj.PredictModel(local_mdl, X_selected(test_idx, :), method);
                        
                        % 计算性能指标
                        [metrics, coefs] = obj.EvaluateModel(y(test_idx), y_pred, y_pred_prob, local_mdl, method, ...
                            X_selected(train_idx, :), y(train_idx));
                        
                        % 存储结果到临时数组，而不是直接修改性能结构体
                        for j = 1:length(metrics_names)
                            metric = metrics_names{j};
                            if isfield(metrics, metric)
                                temp_metrics(i, j) = metrics.(metric);
                            end
                        end
                        
                        y_pred_all{i} = y_pred;
                        y_test_all{i} = y(test_idx);
                        y_pred_prob_all{i} = y_pred_prob;
                        all_coefs{i} = coefs;
                        
                        % 组合性能
                        combo_key = sprintf('%s', mat2str(sort(selected_vars)));
                        combo_keys{i} = combo_key;
                        
                        perf = perf_template;
                        for field = fieldnames(metrics)'
                            if isfield(perf, field{1})
                                perf.(field{1}) = metrics.(field{1});
                            end
                        end
                        perf.count = 1;
                        perf.variables = var_names(selected_vars);
                        perf_structs{i} = perf;
                    end
                    
                catch ME
                    obj.Logger.Log('warning', sprintf('模型 %d 失败: %s', i, ME.message));
                    models{i} = [];
                end
            end
            
            % 处理组合性能
            group_performance = obj.ProcessGroupPerformance(combo_keys, perf_structs);
            
            % 将临时数组转换回结构体
            performance = struct();
            for j = 1:length(metrics_names)
                metric = metrics_names{j};
                performance.(metric) = temp_metrics(:, j);
            end
            
            % 处理总体性能
            performance = obj.ProcessOverallPerformance(performance, y_pred_all, y_test_all, y_pred_prob_all, all_coefs);
        end
        
        function model = TrainModel(obj, X_train, y_train, method)
            % 训练单个模型
            
            try
                switch lower(method)
                    case 'randomforest'
                        if exist('TreeBagger', 'file')
                            model = TreeBagger(100, X_train, y_train, ...
                                'Method', 'classification', ...
                                'OOBPrediction', 'off', ...
                                'Options', obj.ParallelManager.GetParallelOptions());
                        else
                            model = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');
                        end
                        
                    otherwise
                        model = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');
                end
            catch
                model = [];
            end
        end
        
        function [y_pred, y_pred_prob] = PredictModel(obj, model, X_test, method)
            % 使用模型进行预测
            
            try
                switch lower(method)
                    case 'randomforest'
                        if isa(model, 'TreeBagger')
                            [y_pred_class, y_pred_scores] = predict(model, X_test);
                            y_pred = str2double(y_pred_class) > 0.5;
                            y_pred_prob = y_pred_scores(:, 2);
                        else
                            y_pred_prob = predict(model, X_test);
                            y_pred = y_pred_prob > 0.5;
                        end
                        
                    otherwise
                        y_pred_prob = predict(model, X_test);
                        y_pred = y_pred_prob > 0.5;
                end
            catch
                y_pred = rand(size(X_test, 1), 1) > 0.5;
                y_pred_prob = rand(size(X_test, 1), 1);
            end
        end
        
        function [metrics, coefs] = EvaluateModel(obj, y_test, y_pred, y_pred_prob, model, method, X_train, y_train)
            % 评估模型性能
            
            metrics = struct();
            coefs = [];
            
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
                    [metrics.aic, metrics.bic] = obj.CalculateModelSelectionMetrics(model, method, X_train, y_train);
                else
                    metrics.aic = NaN;
                    metrics.bic = NaN;
                end
                
                % 提取系数
                coefs = obj.ExtractCoefficients(model, method);
                
            catch
                % 设置默认值
                metrics.accuracy = 0;
                metrics.precision = 0;
                metrics.recall = 0;
                metrics.specificity = 0;
                metrics.f1_score = 0;
                metrics.auc = 0.5;
                metrics.aic = Inf;
                metrics.bic = Inf;
            end
        end
        
        function [accuracy, precision, recall, specificity, f1_score] = CalculateClassificationMetrics(obj, y_test, y_pred)
            % 计算分类指标
            
            TP = sum(y_pred == 1 & y_test == 1);
            TN = sum(y_pred == 0 & y_test == 0);
            FP = sum(y_pred == 1 & y_test == 0);
            FN = sum(y_pred == 0 & y_test == 1);
            
            accuracy = (TP + TN) / length(y_test);
            sensitivity = TP / max(1, (TP + FN));
            specificity = TN / max(1, (TN + FP));
            precision = TP / max(1, (TP + FP));
            f1_score = 2 * (precision * sensitivity) / max(1, (precision + sensitivity));
            
            recall = sensitivity;  % 召回率就是敏感性
        end
        
        function [aic, bic] = CalculateModelSelectionMetrics(obj, model, method, X_train, y_train)
            % 计算模型选择指标
            
            try
                switch lower(method)
                    case 'randomforest'
                        if isa(model, 'TreeBagger')
                            % 使用OOB误差计算
                            oob_err_vec = oobError(model);
                            oob_error = oob_err_vec(end);
                            n_trees = model.NumTrees;
                            n_predictors = size(X_train, 2);
                            
                            aic = oob_error * length(y_train) + 2 * (n_trees + n_predictors);
                            bic = oob_error * length(y_train) + log(length(y_train)) * (n_trees + n_predictors);
                        else
                            aic = model.ModelCriterion.AIC;
                            bic = model.ModelCriterion.BIC;
                        end
                        
                    otherwise
                        aic = model.ModelCriterion.AIC;
                        bic = model.ModelCriterion.BIC;
                end
            catch
                aic = Inf;
                bic = Inf;
            end
        end
        
        function coefs = ExtractCoefficients(obj, model, method)
            % 提取模型系数
            
            try
                switch lower(method)
                    case 'randomforest'
                        if isa(model, 'TreeBagger')
                            coefs = model.OOBPermutedPredictorDeltaError;
                        else
                            coefs = model.Coefficients.Estimate;
                        end
                        
                    otherwise
                        coefs = model.Coefficients.Estimate;
                end
            catch
                coefs = [];
            end
        end
        
        function group_performance = ProcessGroupPerformance(obj, combo_keys, perf_structs)
            % 处理组合性能
            
            [unique_keys, ~, ic] = unique(combo_keys);
            n_unique_combos = length(unique_keys);
            
            perf_template = struct(...
                'accuracy', 0, 'sensitivity', 0, 'specificity', 0, ...
                'precision', 0, 'f1_score', 0, 'auc', 0, ...
                'aic', 0, 'bic', 0, 'count', 0, 'variables', {{}});
            
            group_performance = repmat(perf_template, n_unique_combos, 1);
            
            for i = 1:n_unique_combos
                combo_indices = find(ic == i);
                first_idx = combo_indices(1);
                
                if ~isempty(perf_structs{first_idx})
                    group_performance(i).variables = perf_structs{first_idx}.variables;
                    group_performance(i).count = length(combo_indices);
                    
                    % 计算平均性能
                    metrics = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc', 'aic', 'bic'};
                    for metric = metrics
                        sum_val = 0;
                        valid_count = 0;
                        for j = 1:length(combo_indices)
                            idx = combo_indices(j);
                            if ~isempty(perf_structs{idx}) && isfield(perf_structs{idx}, metric{1})
                                sum_val = sum_val + perf_structs{idx}.(metric{1});
                                valid_count = valid_count + 1;
                            end
                        end
                        if valid_count > 0
                            group_performance(i).(metric{1}) = sum_val / valid_count;
                        end
                    end
                end
            end
            
            % 按出现次数排序
            [~, idx] = sort([group_performance.count], 'descend');
            group_performance = group_performance(idx);
        end
        
        function performance = ProcessOverallPerformance(obj, performance, y_pred_all, y_test_all, y_pred_prob_all, all_coefs)
            % 处理总体性能
            
            % 计算平均值和标准差
            metrics = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc', 'aic', 'bic'};
            
            for metric = metrics
                values = performance.(metric{1});
                performance.(['avg_' metric{1}]) = mean(values, 'omitnan');
                performance.(['std_' metric{1}]) = std(values, 'omitnan');
            end
            
            % 添加预测结果
            performance.y_pred = y_pred_all;
            performance.y_test = y_test_all;
            performance.y_pred_prob = y_pred_prob_all;
            performance.all_coefs = all_coefs;
        end
        
        function params = ExtractModelParameters(obj, models, var_names)
            % 提取模型参数
            
            n_models = length(models);
            coef_cell = cell(n_models, 1);
            pval_cell = cell(n_models, 1);
            var_cell = cell(n_models, 1);
            
            parfor i = 1:n_models
                mdl = models{i};
                
                if ~isempty(mdl)
                    if isa(mdl, 'TreeBagger')
                        % 随机森林特征重要性
                        local_imp = mdl.OOBPermutedPredictorDeltaError;
                        local_coef = local_imp;
                        local_pval = nan(size(local_imp));
                        local_vars = cellstr(mdl.PredictorNames);
                    else
                        % 逻辑回归模型
                        try
                            coef_table = mdl.Coefficients;
                            local_coef = coef_table.Estimate';
                            local_pval = coef_table.pValue';
                            local_vars = coef_table.Row';
                        catch
                            local_coef = [];
                            local_pval = [];
                            local_vars = {};
                        end
                    end
                else
                    local_coef = [];
                    local_pval = [];
                    local_vars = {};
                end
                
                coef_cell{i} = local_coef;
                pval_cell{i} = local_pval;
                var_cell{i} = local_vars;
            end
            
            params = struct();
            params.coef_cell = coef_cell;
            params.pval_cell = pval_cell;
            params.var_cell = var_cell;
        end
        
        function IntegrateResults(obj, method_results)
            % 整合所有方法的结果
            
            obj.Results = struct();
            
            for i = 1:length(method_results)
                result = method_results{i};
                if result.success
                    method = result.method;
                    obj.Results.(method) = rmfield(result, {'method', 'success'});
                else
                    obj.Logger.Log('warning', sprintf('%s方法失败，跳过', result.method));
                end
            end
        end
        
        function PerformMetaAnalysis(obj)
            % 执行元分析
            
            obj.Logger.Log('info', '开始元分析');
            
            try
                % 1. 跨方法变量选择一致性分析
                obj.AnalyzeCrossMethodConsistency();
                
                % 2. 性能比较分析
                obj.AnalyzePerformanceComparison();
                
                % 3. 最佳变量组合推荐
                obj.RecommendBestVariableSet();
                
                % 4. 稳定性分析
                obj.AnalyzeSelectionStability();
                
                obj.Logger.Log('info', '元分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'PerformMetaAnalysis');
            end
        end
        
        function AnalyzeCrossMethodConsistency(obj)
            % 跨方法变量选择一致性分析
            
            try
                methods = fieldnames(obj.Results);
                n_methods = length(methods);
                
                if n_methods < 2
                    return;
                end
                
                % 收集所有方法的变量选择
                all_frequencies = [];
                method_names = {};
                
                for i = 1:n_methods
                    method = methods{i};
                    if isfield(obj.Results.(method), 'var_freq')
                        all_frequencies = [all_frequencies, obj.Results.(method).var_freq];
                        method_names{end+1} = method;
                    end
                end
                
                if isempty(all_frequencies)
                    return;
                end
                
                % 计算一致性指标
                consistency = struct();
                consistency.method_agreement_matrix = corr(all_frequencies);
                consistency.average_frequency = mean(all_frequencies, 2);
                consistency.frequency_variance = var(all_frequencies, 0, 2);
                consistency.consistency_score = mean(consistency.method_agreement_matrix(:));
                
                % 识别一致选择的变量
                threshold = 0.5;
                consistency.consistently_selected = find(consistency.average_frequency > threshold);
                consistency.variable_consensus = consistency.average_frequency(consistency.consistently_selected);
                
                obj.Results.meta_analysis.cross_method_consistency = consistency;
                
                obj.Logger.Log('info', sprintf('方法间一致性分析完成，一致性得分: %.3f', consistency.consistency_score));
                
            catch ME
                obj.Logger.LogException(ME, 'AnalyzeCrossMethodConsistency');
            end
        end
        
        function AnalyzePerformanceComparison(obj)
            % 性能比较分析
            
            try
                methods = fieldnames(obj.Results);
                performance_comparison = struct();
                
                % 收集性能指标
                metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc'};
                
                for metric = metrics
                    comparison_data = table();
                    
                    for i = 1:length(methods)
                        method = methods{i};
                        if isfield(obj.Results.(method), 'performance') && ...
                           isfield(obj.Results.(method).performance, ['avg_' metric{1}])
                            
                            row = table();
                            row.Method = {method};
                            row.Mean = obj.Results.(method).performance.(['avg_' metric{1}]);
                            row.Std = obj.Results.(method).performance.(['std_' metric{1}]);
                            row.Metric = {metric{1}};
                            
                            comparison_data = [comparison_data; row];
                        end
                    end
                    
                    if height(comparison_data) > 0
                        % 排序
                        comparison_data = sortrows(comparison_data, 'Mean', 'descend');
                        performance_comparison.(metric{1}) = comparison_data;
                        
                        % 统计检验
                        if height(comparison_data) >= 2
                            values = comparison_data.Mean;
                            [~, p_value] = ttest(values);
                            performance_comparison.([metric{1} '_test']) = struct('p_value', p_value);
                        end
                    end
                end
                
                obj.Results.meta_analysis.performance_comparison = performance_comparison;
                
                obj.Logger.Log('info', '性能比较分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'AnalyzePerformanceComparison');
            end
        end
        
        function RecommendBestVariableSet(obj)
            % 推荐最佳变量组合
            
            try
                methods = fieldnames(obj.Results);
                recommendations = struct();
                
                % 基于性能推荐
                best_performance = struct();
                for metric = {'f1_score', 'auc', 'accuracy'}
                    best_method = '';
                    best_value = -Inf;
                    
                    for i = 1:length(methods)
                        method = methods{i};
                        if isfield(obj.Results.(method), 'performance') && ...
                           isfield(obj.Results.(method).performance, ['avg_' metric{1}])
                            value = obj.Results.(method).performance.(['avg_' metric{1}]);
                            if value > best_value
                                best_value = value;
                                best_method = method;
                            end
                        end
                    end
                    
                    if ~isempty(best_method)
                        best_performance.(metric{1}) = struct('method', best_method, 'value', best_value);
                        
                        % 获取变量组合
                        if isfield(obj.Results.(best_method), 'group_performance') && ...
                           ~isempty(obj.Results.(best_method).group_performance)
                            top_combo = obj.Results.(best_method).group_performance(1);
                            best_performance.(metric{1}).variables = top_combo.variables;
                        end
                    end
                end
                
                recommendations.by_performance = best_performance;
                
                % 基于一致性推荐
                if isfield(obj.Results, 'meta_analysis') && ...
                   isfield(obj.Results.meta_analysis, 'cross_method_consistency')
                    consistency = obj.Results.meta_analysis.cross_method_consistency;
                    
                    if ~isempty(consistency.consistently_selected)
                        % 获取变量名
                        all_var_names = obj.GetVariableNames();
                        if ~isempty(all_var_names)
                            recommended_vars = all_var_names(consistency.consistently_selected);
                            recommendations.by_consensus = struct();
                            recommendations.by_consensus.variables = recommended_vars;
                            recommendations.by_consensus.frequencies = consistency.variable_consensus;
                        end
                    end
                end
                
                obj.Results.meta_analysis.recommendations = recommendations;
                
                obj.Logger.Log('info', '变量组合推荐完成');
                
            catch ME
                obj.Logger.LogException(ME, 'RecommendBestVariableSet');
            end
        end
        
        function AnalyzeSelectionStability(obj)
            % 分析变量选择的稳定性
            
            try
                methods = fieldnames(obj.Results);
                stability_analysis = struct();
                
                for i = 1:length(methods)
                    method = methods{i};
                    
                    if isfield(obj.Results.(method), 'var_combinations')
                        var_combinations = obj.Results.(method).var_combinations;
                        
                        % 计算选择稳定性
                        stability = struct();
                        
                        % 1. Shannon熵
                        var_freq = obj.Results.(method).var_freq;
                        stability.shannon_entropy = obj.CalculateShannonEntropy(var_freq);
                        
                        % 2. Jaccard相似度
                        stability.jaccard_similarity = obj.CalculateJaccardSimilarity(var_combinations);
                        
                        % 3. 变异系数
                        combo_sizes = cellfun(@length, var_combinations);
                        stability.size_cv = std(combo_sizes) / mean(combo_sizes);
                        
                        stability_analysis.(method) = stability;
                    end
                end
                
                obj.Results.meta_analysis.stability_analysis = stability_analysis;
                
                obj.Logger.Log('info', '稳定性分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'AnalyzeSelectionStability');
            end
        end
        
        function entropy = CalculateShannonEntropy(obj, var_freq)
            % 计算Shannon熵
            
            % 避免log(0)
            non_zero_freq = var_freq(var_freq > 0);
            
            if isempty(non_zero_freq)
                entropy = 0;
            else
                entropy = -sum(non_zero_freq .* log2(non_zero_freq));
            end
        end
        
        function similarity = CalculateJaccardSimilarity(obj, var_combinations)
            % 计算Jaccard相似度
            
            n_combinations = length(var_combinations);
            similarities = [];
            
            for i = 1:n_combinations-1
                for j = i+1:n_combinations
                    set_i = var_combinations{i};
                    set_j = var_combinations{j};
                    
                    if isempty(set_i) && isempty(set_j)
                        sim = 1;
                    elseif isempty(set_i) || isempty(set_j)
                        sim = 0;
                    else
                        intersection = length(intersect(set_i, set_j));
                        union = length(union(set_i, set_j));
                        sim = intersection / union;
                    end
                    
                    similarities = [similarities; sim];
                end
            end
            
            if isempty(similarities)
                similarity = 1;
            else
                similarity = mean(similarities);
            end
        end
        
        function var_names = GetVariableNames(obj)
            % 获取变量名称
            
            var_names = [];
            methods = fieldnames(obj.Results);
            
            for i = 1:length(methods)
                method = methods{i};
                if isfield(obj.Results.(method), 'group_performance') && ...
                   ~isempty(obj.Results.(method).group_performance)
                    
                    for j = 1:length(obj.Results.(method).group_performance)
                        if isfield(obj.Results.(method).group_performance(j), 'variables')
                            vars = obj.Results.(method).group_performance(j).variables;
                            if ~isempty(vars)
                                var_names = vars;
                                return;
                            end
                        end
                    end
                end
            end
        end
        
        function CreateSelectionVisualizations(obj)
            % 创建变量选择可视化
            
            try
                figure_dir = fullfile(obj.Config.OutputDirectory, 'figures', 'variable_selection');
                if ~exist(figure_dir, 'dir')
                    mkdir(figure_dir);
                end
                
                % 1. 变量选择频率图
                obj.CreateVariableFrequencyPlot(figure_dir);
                
                % 2. 方法性能比较图
                obj.CreateMethodPerformanceComparison(figure_dir);
                
                % 3. 变量组合性能图
                obj.CreateCombinationPerformancePlot(figure_dir);
                
                % 4. 一致性分析图
                obj.CreateConsistencyAnalysisPlot(figure_dir);
                
                % 5. 稳定性分析图
                obj.CreateStabilityAnalysisPlot(figure_dir);
                
                obj.Logger.Log('info', '变量选择可视化已创建');
                
            catch ME
                obj.Logger.LogException(ME, 'CreateSelectionVisualizations');
            end
        end
        
        function CreateVariableFrequencyPlot(obj, figure_dir)
            % 创建变量选择频率图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
                
                methods = fieldnames(obj.Results);
                n_methods = length(methods);
                
                % 收集数据
                all_frequencies = [];
                var_names = obj.GetVariableNames();
                
                for i = 1:n_methods
                    method = methods{i};
                    if isfield(obj.Results.(method), 'var_freq')
                        all_frequencies = [all_frequencies, obj.Results.(method).var_freq];
                    end
                end
                
                if isempty(all_frequencies)
                    close(fig);
                    return;
                end
                
                % 创建分组条形图
                bar(all_frequencies, 'grouped');
                
                % 设置标签
                xlabel('变量');
                ylabel('选择频率');
                title('不同方法的变量选择频率比较');
                legend(methods, 'Location', 'best');
                grid on;
                
                % 设置X轴标签
                if ~isempty(var_names) && length(var_names) == size(all_frequencies, 1)
                    set(gca, 'XTick', 1:length(var_names), 'XTickLabel', var_names, 'XTickLabelRotation', 45);
                end
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'variable_frequency_comparison.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'variable_frequency_comparison.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateVariableFrequencyPlot');
            end
        end
        
        function CreateMethodPerformanceComparison(obj, figure_dir)
            % 创建方法性能比较图
            
            try
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
                
                methods = fieldnames(obj.Results);
                metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc'};
                metric_labels = {'准确率', '精确率', '召回率', 'F1分数', 'AUC'};
                
                n_metrics = length(metrics);
                
                for i = 1:n_metrics
                    subplot(2, 3, i);
                    
                    metric = metrics{i};
                    means = [];
                    stds = [];
                    method_names = {};
                    
                    for j = 1:length(methods)
                        method = methods{j};
                        if isfield(obj.Results.(method), 'performance') && ...
                           isfield(obj.Results.(method).performance, ['avg_' metric])
                            
                            means = [means; obj.Results.(method).performance.(['avg_' metric])];
                            stds = [stds; obj.Results.(method).performance.(['std_' metric])];
                            method_names{end+1} = method;
                        end
                    end
                    
                    if ~isempty(means)
                        % 创建条形图
                        bar_h = bar(means);
                        hold on;
                        errorbar(1:length(means), means, stds, '.k');
                        
                        % 设置标签
                        set(gca, 'XTick', 1:length(method_names), 'XTickLabel', method_names, 'XTickLabelRotation', 45);
                        ylabel(metric_labels{i});
                        title(sprintf('%s比较', metric_labels{i}));
                        grid on;
                        
                        if i <= 5
                            ylim([0, 1.05]);
                        end
                    end
                end
                
                sgtitle('不同变量选择方法的性能比较', 'FontSize', 16, 'FontWeight', 'bold');
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'method_performance_comparison.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'method_performance_comparison.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateMethodPerformanceComparison');
            end
        end
        
        function CreateCombinationPerformancePlot(obj, figure_dir)
            % 创建变量组合性能图
            
            try
                methods = fieldnames(obj.Results);
                
                for i = 1:length(methods)
                    method = methods{i};
                    
                    if isfield(obj.Results.(method), 'group_performance')
                        group_perf = obj.Results.(method).group_performance;
                        
                        if length(group_perf) >= 3
                            fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
                            
                            % 取前8个组合
                            top_n = min(8, length(group_perf));
                            
                            % 提取数据
                            combo_labels = cell(top_n, 1);
                            f1_scores = zeros(top_n, 1);
                            aucs = zeros(top_n, 1);
                            counts = zeros(top_n, 1);
                            
                            for j = 1:top_n
                                combo_labels{j} = sprintf('组合 %d', j);
                                f1_scores(j) = group_perf(j).f1_score;
                                aucs(j) = group_perf(j).auc;
                                counts(j) = group_perf(j).count;
                            end
                            
                            % 创建两个子图
                            subplot(2, 1, 1);
                            bar([f1_scores, aucs], 'grouped');
                            set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45);
                            ylabel('性能指标');
                            legend({'F1分数', 'AUC'});
                            title(sprintf('%s方法：变量组合性能', method));
                            grid on;
                            ylim([0, 1.05]);
                            
                            subplot(2, 1, 2);
                            bar(counts);
                            set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45);
                            ylabel('出现次数');
                            title(sprintf('%s方法：变量组合频率', method));
                            grid on;
                            
                            % 保存图形
                            filename = sprintf('%s_combination_performance', method);
                            saveas(fig, fullfile(figure_dir, [filename '.svg']), 'svg');
                            saveas(fig, fullfile(figure_dir, [filename '.png']), 'png');
                            close(fig);
                        end
                    end
                end
                
            catch ME
                obj.Logger.LogException(ME, 'CreateCombinationPerformancePlot');
            end
        end
        
        function CreateConsistencyAnalysisPlot(obj, figure_dir)
            % 创建一致性分析图
            
            try
                if ~isfield(obj.Results, 'meta_analysis') || ...
                   ~isfield(obj.Results.meta_analysis, 'cross_method_consistency')
                    return;
                end
                
                consistency = obj.Results.meta_analysis.cross_method_consistency;
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 800]);
                
                % 绘制方法间相关性矩阵
                imagesc(consistency.method_agreement_matrix);
                colorbar;
                colormap('RdBu_r');
                
                % 设置标签
                methods = fieldnames(obj.Results);
                set(gca, 'XTick', 1:length(methods), 'XTickLabel', methods, 'XTickLabelRotation', 45);
                set(gca, 'YTick', 1:length(methods), 'YTickLabel', methods);
                
                title('方法间变量选择一致性');
                xlabel('方法');
                ylabel('方法');
                
                % 添加数值标签
                for i = 1:size(consistency.method_agreement_matrix, 1)
                    for j = 1:size(consistency.method_agreement_matrix, 2)
                        text(j, i, sprintf('%.3f', consistency.method_agreement_matrix(i, j)), ...
                            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
                    end
                end
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'consistency_analysis.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'consistency_analysis.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateConsistencyAnalysisPlot');
            end
        end
        
        function CreateStabilityAnalysisPlot(obj, figure_dir)
            % 创建稳定性分析图
            
            try
                if ~isfield(obj.Results, 'meta_analysis') || ...
                   ~isfield(obj.Results.meta_analysis, 'stability_analysis')
                    return;
                end
                
                stability = obj.Results.meta_analysis.stability_analysis;
                methods = fieldnames(stability);
                
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 600]);
                
                % 提取数据
                shannon_entropies = zeros(length(methods), 1);
                jaccard_similarities = zeros(length(methods), 1);
                size_cvs = zeros(length(methods), 1);
                
                for i = 1:length(methods)
                    method = methods{i};
                    shannon_entropies(i) = stability.(method).shannon_entropy;
                    jaccard_similarities(i) = stability.(method).jaccard_similarity;
                    size_cvs(i) = stability.(method).size_cv;
                end
                
                % 创建三个子图
                subplot(1, 3, 1);
                bar(shannon_entropies);
                set(gca, 'XTick', 1:length(methods), 'XTickLabel', methods, 'XTickLabelRotation', 45);
                ylabel('Shannon熵');
                title('变量选择熵');
                grid on;
                
                subplot(1, 3, 2);
                bar(jaccard_similarities);
                set(gca, 'XTick', 1:length(methods), 'XTickLabel', methods, 'XTickLabelRotation', 45);
                ylabel('Jaccard相似度');
                title('选择一致性');
                grid on;
                ylim([0, 1]);
                
                subplot(1, 3, 3);
                bar(size_cvs);
                set(gca, 'XTick', 1:length(methods), 'XTickLabel', methods, 'XTickLabelRotation', 45);
                ylabel('变异系数');
                title('选择大小稳定性');
                grid on;
                
                sgtitle('变量选择稳定性分析', 'FontSize', 16, 'FontWeight', 'bold');
                
                % 保存图形
                saveas(fig, fullfile(figure_dir, 'stability_analysis.svg'), 'svg');
                saveas(fig, fullfile(figure_dir, 'stability_analysis.png'), 'png');
                close(fig);
                
            catch ME
                obj.Logger.LogException(ME, 'CreateStabilityAnalysisPlot');
            end
        end
        
        function GenerateSelectionReport(obj)
            % 生成变量选择报告
            
            try
                report = struct();
                report.timestamp = datetime('now');
                report.methods_analyzed = length(fieldnames(obj.Results));
                
                % 性能总结
                report.performance_summary = struct();
                
                methods = fieldnames(obj.Results);
                for method = methods'
                    if isfield(obj.Results.(method{1}), 'performance')
                        perf = obj.Results.(method{1}).performance;
                        summary = struct();
                        
                        metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc'};
                        for metric = metrics
                            if isfield(perf, ['avg_' metric{1}])
                                summary.(metric{1}) = struct();
                                summary.(metric{1}).mean = perf.(['avg_' metric{1}]);
                                summary.(metric{1}).std = perf.(['std_' metric{1}]);
                            end
                        end
                        
                        report.performance_summary.(method{1}) = summary;
                    end
                end
                
                % 推荐总结
                if isfield(obj.Results, 'meta_analysis') && ...
                   isfield(obj.Results.meta_analysis, 'recommendations')
                    report.recommendations = obj.Results.meta_analysis.recommendations;
                end
                
                % 稳定性总结
                if isfield(obj.Results, 'meta_analysis') && ...
                   isfield(obj.Results.meta_analysis, 'stability_analysis')
                    report.stability_summary = struct();
                    
                    for method = methods'
                        if isfield(obj.Results.meta_analysis.stability_analysis, method{1})
                            stability = obj.Results.meta_analysis.stability_analysis.(method{1});
                            report.stability_summary.(method{1}) = stability;
                        end
                    end
                end
                
                % 保存报告
                obj.Results.final_report = report;
                
                % 记录关键发现
                obj.Logger.CreateSection('变量选择报告总结');
                obj.Logger.Log('info', sprintf('分析方法数: %d', report.methods_analyzed));
                
                % 记录最佳性能
                best_f1 = -Inf;
                best_method = '';
                
                for method = methods'
                    if isfield(report.performance_summary, method{1}) && ...
                       isfield(report.performance_summary.(method{1}), 'f1_score')
                        f1 = report.performance_summary.(method{1}).f1_score.mean;
                        if f1 > best_f1
                            best_f1 = f1;
                            best_method = method{1};
                        end
                    end
                end
                
                if ~isempty(best_method)
                    obj.Logger.Log('info', sprintf('最佳性能方法: %s (F1=%.3f)', best_method, best_f1));
                end
                
                % 记录推荐
                if isfield(report, 'recommendations') && isfield(report.recommendations, 'by_consensus')
                    consensus_vars = report.recommendations.by_consensus.variables;
                    obj.Logger.Log('info', sprintf('一致推荐的变量: %s', strjoin(consensus_vars, ', ')));
                end
                
            catch ME
                obj.Logger.LogException(ME, 'GenerateSelectionReport');
            end
        end
        
        function ExportVariableSelectionTables(obj, output_dir)
            % 导出变量选择表格
            
            try
                % 1. 导出变量选择频率表
                methods = fieldnames(obj.Results);
                var_names = obj.GetVariableNames();
                
                if ~isempty(var_names)
                    freq_table = table();
                    freq_table.Variable = var_names';
                    
                    for i = 1:length(methods)
                        method = methods{i};
                        if isfield(obj.Results.(method), 'var_freq')
                            freq_table.(method) = obj.Results.(method).var_freq;
                        end
                    end
                    
                    % 添加平均频率
                    freq_data = table2array(freq_table(:, 2:end));
                    freq_table.Average = mean(freq_data, 2);
                    
                    % 按平均频率排序
                    freq_table = sortrows(freq_table, 'Average', 'descend');
                    
                    writetable(freq_table, fullfile(output_dir, 'variable_selection_frequencies.csv'));
                end
                
                % 2. 导出性能比较表
                perf_table = table();
                
                metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc'};
                for metric = metrics
                    method_names = {};
                    means = [];
                    stds = [];
                    
                    for i = 1:length(methods)
                        method = methods{i};
                        if isfield(obj.Results.(method), 'performance') && ...
                           isfield(obj.Results.(method).performance, ['avg_' metric{1}])
                            
                            method_names{end+1} = method;
                            means = [means; obj.Results.(method).performance.(['avg_' metric{1}])];
                            stds = [stds; obj.Results.(method).performance.(['std_' metric{1}])];
                        end
                    end
                    
                    if ~isempty(method_names)
                        metric_table = table();
                        metric_table.Method = method_names';
                        metric_table.Mean = means;
                        metric_table.Std = stds;
                        metric_table.Metric = repmat({metric{1}}, length(method_names), 1);
                        
                        perf_table = [perf_table; metric_table];
                    end
                end
                
                writetable(perf_table, fullfile(output_dir, 'method_performance_comparison.csv'));
                
                % 3. 导出变量组合表
                for i = 1:length(methods)
                    method = methods{i};
                    if isfield(obj.Results.(method), 'group_performance')
                        group_perf = obj.Results.(method).group_performance;
                        
                        combo_table = table();
                        
                        for j = 1:length(group_perf)
                            row = table();
                            row.Combination = j;
                            row.Count = group_perf(j).count;
                            row.Accuracy = group_perf(j).accuracy;
                            row.F1_Score = group_perf(j).f1_score;
                            row.AUC = group_perf(j).auc;
                            row.Variables = {strjoin(group_perf(j).variables, ', ')};
                            
                            combo_table = [combo_table; row];
                        end
                        
                        filename = sprintf('%s_variable_combinations.csv', method);
                        writetable(combo_table, fullfile(output_dir, filename));
                    end
                end
                
                obj.Logger.Log('info', '变量选择表格已导出');
                
            catch ME
                obj.Logger.LogException(ME, 'ExportVariableSelectionTables');
            end
        end
    end
end