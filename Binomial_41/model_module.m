%% model_module.m - 模型训练与评估模块
classdef model_module
    methods(Static)
        function results = perform_variable_selection_and_modeling(X, y, train_indices, test_indices, methods, var_names)
            % 使用不同方法进行变量选择和模型训练
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量向量
            %   train_indices - 训练集索引
            %   test_indices - 测试集索引
            %   methods - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   results - 结果结构体
            
            t_start = toc;
            results = struct();
            
            % 创建函数句柄数组
            futures = cell(length(methods), 1);
            
            % 并行启动所有方法
            for i = 1:length(methods)
                method = methods{i};
                logger.log_message('info', sprintf('开始并行使用%s方法筛选变量', method));
                
                % 使用parfeval异步执行
                futures{i} = parfeval(@model_module.process_method, 1, X, y, train_indices, test_indices, method, var_names);
            end
            
            % 收集结果
            for i = 1:length(methods)
                method = methods{i};
                [methodResult] = fetchOutputs(futures{i});
                results.(method) = methodResult;
                logger.log_message('info', sprintf('%s方法完成', method));
            end
            
            t_end = toc;
            logger.log_message('info', sprintf('所有变量选择方法完成，耗时：%.2f秒', t_end - t_start));
            
            return;
        end
        
        function result = process_method(X_final, y, train_indices, test_indices, method, var_names)
            % 处理单个变量选择方法
            % 输入:
            %   X_final - 自变量矩阵
            %   y - 因变量
            %   train_indices - 训练集索引
            %   test_indices - 测试集索引
            %   method - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   result - 方法结果结构
            
            % 变量筛选
            logger.log_message('info', sprintf('%s: 开始变量筛选', method));
            [selected_vars, var_freq, var_combinations] = model_module.select_variables(X_final, y, train_indices, method);
            logger.log_message('info', sprintf('%s: 变量筛选完成', method));
            
            % 训练和评估模型
            logger.log_message('info', sprintf('%s: 开始模型训练与评估', method));
            [models, performance, group_performance] = model_module.train_and_evaluate_models_with_groups(X_final, y, train_indices, test_indices, var_combinations, method, var_names);
            logger.log_message('info', sprintf('%s: 模型训练与评估完成', method));
            
            % 获取模型参数
            logger.log_message('info', sprintf('%s: 提取模型参数', method));
            params = model_module.get_model_parameters(models, var_names);
            logger.log_message('info', sprintf('%s: 参数提取完成', method));
            
            % 组织结果
            result = struct();
            result.selected_vars = selected_vars;
            result.var_freq = var_freq;
            result.var_combinations = var_combinations;
            result.models = models;
            result.performance = performance;
            result.group_performance = group_performance;
            result.params = params;
        end

        function [selected_vars, var_freq, var_combinations] = select_variables(X, y, train_indices, method)
            % 使用不同方法进行变量选择，并保存每次选择的变量组合
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   train_indices - 训练集索引
            %   method - 方法名称
            % 输出:
            %   selected_vars - 总体选中的变量
            %   var_freq - 变量选择频率
            %   var_combinations - 每次迭代选择的变量组合
            
            n_samples = length(train_indices);
            n_vars = size(X, 2);
            
            % 初始化变量选择频率计数
            var_selection_count = zeros(n_vars, 1);
            
            % 初始化变量组合存储
            var_combinations = cell(n_samples, 1);
            
            % 优化并行设置 - 适合i9处理器
            UseParallel = true;
            opts = statset('UseParallel', UseParallel, 'Display', 'off');
            
            % 更密集的lambda范围提高模型质量
            lambda_range = logspace(-5, 1, 50);  
            
            % 使用parfor并行处理
            parfor i = 1:n_samples
                % 获取当前训练集
                X_train = X(train_indices{i}, :);
                y_train = y(train_indices{i});
                
                % 根据方法选择变量
                selected = false(1, n_vars);
                
                switch lower(method)
                    case 'stepwise'
                        % 逐步回归 - 优化P值阈值
                        [~, ~, ~, inmodel] = stepwisefit(X_train, y_train, 'PEnter', 0.15, 'PRemove', 0.20, 'Display', 'off');
                        selected = inmodel;
                        
                    case 'lasso'
                        % LASSO回归 - 增加交叉验证折数
                        [B, FitInfo] = lasso(X_train, y_train, 'CV', 10, ...
                                            'Options', opts, 'Alpha', 1, 'Lambda', lambda_range);
                        lambda_min = FitInfo.LambdaMinMSE;
                        coef = B(:, FitInfo.Lambda == lambda_min);
                        selected = abs(coef) > 0;
                        
                    case 'ridge'
                        % Ridge回归 - 针对i9优化alpha值
                        [~, FitInfo] = lasso(X_train, y_train, 'CV', 10, 'Alpha', 0.001, 'Lambda', lambda_range, 'Options', opts);
                        lambda_min = FitInfo.LambdaMinMSE;
                        B = ridge(y_train, X_train, lambda_min, 0);
                        % 使用自适应阈值
                        threshold = max(0.05, std(B(2:end)) * 0.1);
                        selected = abs(B(2:end)) > threshold;
                        
                    case 'elasticnet'
                        % Elastic Net - 使用中等alpha值
                        [B, FitInfo] = lasso(X_train, y_train, 'CV', 10, ...
                                            'Options', opts, 'Alpha', 0.5, 'Lambda', lambda_range);
                        lambda_min = FitInfo.LambdaMinMSE;
                        coef = B(:, FitInfo.Lambda == lambda_min);
                        selected = abs(coef) > 0;
                        
                    case 'randomforest'
                        % Random Forest - 增加并行控制参数和性能优化
                        if exist('TreeBagger', 'file')
                            % 创建高级并行选项 - 移除 CompileOptions 参数
                            parallelOptions = statset('UseParallel', true, 'UseSubstreams', true);
                            
                            % 使用更多树和更高效的参数配置
                            mdl = TreeBagger(200, X_train, y_train, 'Method', 'classification', ...
                                'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', ...
                                'MinLeafSize', max(1, floor(size(X_train,1)/50)), ...
                                'NumPredictorsToSample', max(1, floor(sqrt(size(X_train,2)))), ...
                                'Options', parallelOptions, ...
                                'PredictorSelection', 'curvature', ...  % 添加曲率测试提高变量选择效率
                                'MaxNumSplits', 1e4, ... % 限制分裂数量提高速度
                                'Surrogate', 'off'); % 关闭替代分裂以提高速度
                                
                            imp = mdl.OOBPermutedPredictorDeltaError;
                            selected = imp > mean(imp);
                            
                            % 使用后立即清除大对象
                            mdl = [];
                        else
                            % 如果没有TreeBagger，使用简单的相关系数筛选
                            [~, pval] = corr(X_train, y_train);
                            selected = pval < 0.05;
                        end
                    otherwise
                        error('不支持的变量选择方法: %s', method);
                end
                
                % 保存当前迭代选择的变量组合
                var_combinations{i} = find(selected);
                
                % 如果没有变量被选中，选择相关性最高的3个变量
                if isempty(var_combinations{i})
                    [~, idx] = sort(abs(corr(X_train, y_train)), 'descend');
                    var_combinations{i} = idx(1:min(3, length(idx)));
                end
            end
            
            % 统计选择频率
            for i = 1:n_samples
                selected = false(n_vars, 1);
                selected(var_combinations{i}) = true;
                var_selection_count = var_selection_count + selected;
            end
            
            % 计算变量选择频率
            var_freq = var_selection_count / n_samples;
            
            % 确保var_freq的长度与n_vars一致
            if length(var_freq) ~= n_vars
                logger.log_message('warning', sprintf('变量频率长度(%d)与变量数量(%d)不匹配，进行调整', length(var_freq), n_vars));
                if length(var_freq) < n_vars
                    tmp = zeros(n_vars, 1);
                    tmp(1:length(var_freq)) = var_freq;
                    var_freq = tmp;
                else
                    var_freq = var_freq(1:n_vars);
                end
            end
            
            % 将变量组合转换为字符串表示，以便统计频率
            combo_strings = cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false);
            [unique_combos, ~, ic] = unique(combo_strings);
            combo_counts = accumarray(ic, 1);
            
            % 找出出现频率最高的变量组合
            [~, max_idx] = max(combo_counts);
            most_frequent_combo = unique_combos{max_idx};
            
            % 从字符串中提取变量索引
            combo_indices = str2num(['[' most_frequent_combo(1:end-1) ']']);
            
            % 使用最频繁的变量组合作为总体选择的变量
            selected_vars = false(n_vars, 1);
            selected_vars(combo_indices) = true;
            
            % 记录最频繁的变量组合
            logger.log_message('info', sprintf('使用出现频率最高的变量组合作为总体变量 (出现%d次，占比%.2f%%)', ...
                combo_counts(max_idx), 100*combo_counts(max_idx)/n_samples));
            logger.log_message('info', sprintf('选择的变量索引: %s', mat2str(combo_indices)));
            
            % 删除重复的变量组合并统计每种组合的出现次数
            [unique_combinations, ~, ic] = unique(cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false));
            combination_counts = accumarray(ic, 1);
            
            % 按出现次数排序
            [sorted_counts, idx] = sort(combination_counts, 'descend');
            sorted_combinations = unique_combinations(idx);
            
            % 打印前5个最常见的变量组合
            logger.log_message('info', sprintf('前%d个最常见的变量组合:', min(5, length(sorted_combinations))));
            for i = 1:min(5, length(sorted_combinations))
                logger.log_message('info', sprintf('组合 #%d (出现%d次): %s', i, sorted_counts(i), sorted_combinations{i}));
            end
        end
        
        function [models, overall_performance, group_performance] = train_and_evaluate_models_with_groups(X, y, train_indices, test_indices, var_combinations, method, var_names)
            % 训练和评估模型，支持变量组合分析
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   train_indices - 训练集索引
            %   test_indices - 测试集索引
            %   var_combinations - 变量组合
            %   method - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   models - 训练的模型
            %   overall_performance - 总体性能
            %   group_performance - 组性能
            
            n_samples = length(train_indices);
        
            % 初始化结果
            models = cell(n_samples, 1);
            perf_template = struct(...
                'accuracy', 0, ...
                'sensitivity', 0, ...
                'specificity', 0, ...
                'precision', 0, ...
                'f1_score', 0, ...
                'auc', 0, ...
                'aic', 0, ...
                'bic', 0, ...
                'count', 0, ...
                'variables', {{}});
        
            % 初始化性能指标数组
            accuracy_values = zeros(n_samples, 1);
            sensitivity_values = zeros(n_samples, 1);
            specificity_values = zeros(n_samples, 1);
            precision_values = zeros(n_samples, 1);
            f1_score_values = zeros(n_samples, 1);
            auc_values = zeros(n_samples, 1);
            aic_values = zeros(n_samples, 1);
            bic_values = zeros(n_samples, 1);
        
            % 初始化预测结果存储
            y_pred_all = cell(n_samples, 1);
            y_test_all = cell(n_samples, 1);
            y_pred_prob_all = cell(n_samples, 1);
        
            % 预分配数组以存储组合键和性能
            combo_keys = cell(n_samples, 1);
            perf_structs = cell(n_samples, 1);
            all_coefs = cell(n_samples, 1);
        
            % 使用parfor并行处理
            parfor i = 1:n_samples
                % 初始化临时变量，避免警告
                y_pred_prob = [];
                y_pred = [];
                coefs = [];
                aic = NaN;  % 初始化为NaN
                bic = NaN;  % 初始化为NaN
        
                % 获取当前训练集和测试集
                train_idx = train_indices{i};
                test_idx = test_indices{i};
        
                % 获取当前迭代的变量组合
                selected_vars = var_combinations{i};
                X_selected = X(:, selected_vars);
        
                % 训练模型
                local_mdl = [];
                switch lower(method)
                    case {'stepwise', 'lasso', 'ridge', 'elasticnet'}
                        local_mdl = fitglm(X_selected(train_idx, :), y(train_idx), ...
                            'Distribution', 'binomial', 'Link', 'logit', ...
                            'Intercept', true, 'PredictorVars', 1:size(X_selected, 2));
                        
                        coefs = local_mdl.Coefficients.Estimate;
                        y_pred_prob = predict(local_mdl, X_selected(test_idx, :));
                        y_pred = y_pred_prob > 0.5;
                        
                        deviance = local_mdl.Deviance;
                        n_params = length(coefs);
                        n_samples_iter = length(train_idx);
                        aic = deviance + 2 * n_params;
                        bic = deviance + log(n_samples_iter) * n_params;
        
                    case 'randomforest'
                        parallelOptions = statset('UseParallel', true);
                        
                        if exist('TreeBagger', 'file')
                            local_mdl = TreeBagger(250, X_selected(train_idx, :), y(train_idx), ...
                                'Method', 'classification', ...
                                'OOBPrediction', 'on', ...
                                'OOBPredictorImportance', 'on', ...
                                'MinLeafSize', max(1, floor(length(train_idx)/50)), ...
                                'NumPredictorsToSample', max(1, floor(sqrt(size(X_selected, 2)))), ...
                                'Options', parallelOptions, ...
                                'PredictorSelection', 'curvature', ...
                                'SplitCriterion', 'gdi', ...
                                'MaxNumSplits', 1e4, ...
                                'Surrogate', 'off');
                            
                            coefs = local_mdl.OOBPermutedPredictorDeltaError;
                            [y_pred_class, y_pred_scores] = predict(local_mdl, X_selected(test_idx, :));
                            y_pred = str2double(y_pred_class) > 0.5;
                            y_pred_prob = y_pred_scores(:, 2);
                            
                            % 修改点：使用oobError方法计算OOB误差
                            oob_err_vec = oobError(local_mdl);
                            oob_error = oob_err_vec(end);
                            n_trees = local_mdl.NumTrees;
                            n_predictors = size(X_selected, 2);
                            
                            aic = oob_error * length(train_idx) + 2 * (n_trees + n_predictors);
                            bic = oob_error * length(train_idx) + log(length(train_idx)) * (n_trees + n_predictors);
                        else
                            local_mdl = fitglm(X_selected(train_idx, :), y(train_idx), 'Distribution', 'binomial', 'Link', 'logit');
                            coefs = local_mdl.Coefficients.Estimate;
                            y_pred_prob = predict(local_mdl, X_selected(test_idx, :));
                            y_pred = y_pred_prob > 0.5;
                            
                            deviance = local_mdl.Deviance;
                            n_params = length(coefs);
                            n_samples_iter = length(train_idx);
                            aic = deviance + 2 * n_params;
                            bic = deviance + log(n_samples_iter) * n_params;
                        end
                end
        
                % 存储模型和系数
                models{i} = local_mdl;
                all_coefs{i} = coefs;
        
                % 存储预测结果
                y_pred_all{i} = y_pred;
                y_test_all{i} = y(test_idx);
                y_pred_prob_all{i} = y_pred_prob;
        
                % 计算性能指标
                y_test = y(test_idx);
                accuracy = sum(y_pred == y_test) / length(y_test);
                
                TP = sum(y_pred == 1 & y_test == 1);
                TN = sum(y_pred == 0 & y_test == 0);
                FP = sum(y_pred == 1 & y_test == 0);
                FN = sum(y_pred == 0 & y_test == 1);
                
                sensitivity = TP / max(1, (TP + FN));
                specificity = TN / max(1, (TN + FP));
                precision = TP / max(1, (TP + FP));
                f1_score = 2 * (precision * sensitivity) / max(1, (precision + sensitivity));
                
                auc = 0.5;
                if length(unique(y_test)) > 1
                    try
                        [~, ~, ~, auc] = perfcurve(y_test, y_pred_prob, 1);
                    catch
                        auc = 0.5;
                    end
                end
        
                % 存储性能指标
                accuracy_values(i) = accuracy;
                sensitivity_values(i) = sensitivity;
                specificity_values(i) = specificity;
                precision_values(i) = precision;
                f1_score_values(i) = f1_score;
                auc_values(i) = auc;
                aic_values(i) = aic;
                bic_values(i) = bic;
        
                % 创建变量组合的唯一标识符
                combo_key = sprintf('%s', mat2str(sort(selected_vars)));
                combo_keys{i} = combo_key;
        
                % 创建性能结构
                perf = perf_template;
                perf.accuracy = accuracy;
                perf.sensitivity = sensitivity;
                perf.specificity = specificity;
                perf.precision = precision;
                perf.f1_score = f1_score;
                perf.auc = auc;
                perf.aic = aic;
                perf.bic = bic;
                perf.count = 1;
                perf.variables = var_names(selected_vars);
                perf_structs{i} = perf;
            end
        
            % 构建整体性能结构
            overall_performance = struct();
            overall_performance.accuracy = accuracy_values;
            overall_performance.sensitivity = sensitivity_values;
            overall_performance.specificity = specificity_values;
            overall_performance.precision = precision_values;
            overall_performance.f1_score = f1_score_values;
            overall_performance.auc = auc_values;
            overall_performance.aic = aic_values;
            overall_performance.bic = bic_values;
            overall_performance.avg_accuracy = mean(accuracy_values);
            overall_performance.avg_sensitivity = mean(sensitivity_values);
            overall_performance.avg_specificity = mean(specificity_values);
            overall_performance.avg_precision = mean(precision_values);
            overall_performance.avg_f1_score = mean(f1_score_values);
            overall_performance.avg_auc = nanmean(auc_values);
            overall_performance.avg_aic = nanmean(aic_values);
            overall_performance.avg_bic = nanmean(bic_values);
            overall_performance.std_accuracy = std(accuracy_values);
            overall_performance.std_sensitivity = std(sensitivity_values);
            overall_performance.std_specificity = std(specificity_values);
            overall_performance.std_precision = std(precision_values);
            overall_performance.std_f1_score = std(f1_score_values);
            overall_performance.std_auc = nanstd(auc_values);
            overall_performance.std_aic = nanstd(aic_values);
            overall_performance.std_bic = nanstd(bic_values);
            overall_performance.all_coefs = all_coefs;
            overall_performance.y_pred = y_pred_all;
            overall_performance.y_test = y_test_all;
            overall_performance.y_pred_prob = y_pred_prob_all;
        
            % 合并组合性能
            [unique_keys, ~, ic] = unique(combo_keys);
            n_unique_combos = length(unique_keys);
            group_performance = repmat(perf_template, n_unique_combos, 1);
        
            for i = 1:n_unique_combos
                combo_indices = find(ic == i);
                first_idx = combo_indices(1);
                group_performance(i).variables = perf_structs{first_idx}.variables;
                group_performance(i).count = length(combo_indices);
                
                acc_sum = 0; sens_sum = 0; spec_sum = 0; prec_sum = 0; f1_sum = 0; auc_sum = 0; aic_sum = 0; bic_sum = 0;
                for j = 1:length(combo_indices)
                    idx = combo_indices(j);
                    acc_sum = acc_sum + perf_structs{idx}.accuracy;
                    sens_sum = sens_sum + perf_structs{idx}.sensitivity;
                    spec_sum = spec_sum + perf_structs{idx}.specificity;
                    prec_sum = prec_sum + perf_structs{idx}.precision;
                    f1_sum = f1_sum + perf_structs{idx}.f1_score;
                    auc_sum = auc_sum + perf_structs{idx}.auc;
                    aic_sum = aic_sum + perf_structs{idx}.aic;
                    bic_sum = bic_sum + perf_structs{idx}.bic;
                end
                
                group_performance(i).accuracy = acc_sum / length(combo_indices);
                group_performance(i).sensitivity = sens_sum / length(combo_indices);
                group_performance(i).specificity = spec_sum / length(combo_indices);
                group_performance(i).precision = prec_sum / length(combo_indices);
                group_performance(i).f1_score = f1_sum / length(combo_indices);
                group_performance(i).auc = auc_sum / length(combo_indices);
                group_performance(i).aic = aic_sum / length(combo_indices);
                group_performance(i).bic = bic_sum / length(combo_indices);
            end
        
            % 按出现次数排序
            [~, idx] = sort([group_performance.count], 'descend');
            group_performance = group_performance(idx);
        
            % 记录最常见的变量组合
            top_n = min(5, length(group_performance));
            logger.log_message('info', sprintf('前%d个最常见的变量组合的性能:', top_n));
            for i = 1:top_n
                combo = group_performance(i);
                var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
                logger.log_message('info', sprintf('组合 #%d (出现%d次, AUC=%.3f, F1=%.3f, AIC=%.1f, BIC=%.1f): %s', ...
                    i, combo.count, combo.auc, combo.f1_score, combo.aic, combo.bic, var_str));
            end
        end

        function cv_results = k_fold_cross_validation(X, y, k, var_names)
            % 执行K折交叉验证来评估模型稳定性
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   k - 折数
            %   var_names - 变量名称（新增）
            % 输出:
            %   results - 交叉验证结果
            
            t_start = toc;
            
            % 确认K值有效
            if k < 2
                error('K值必须大于等于2');
            end
            
            % 获取总样本数
            n = length(y);
            
            % 如果K大于样本数，调整K
            if k > n
                logger.log_message('warning', sprintf('K值(%d)大于样本数(%d)，调整为%d', k, n, n));
                k = n;
            end
            
            % 创建交叉验证分组
            cv = cvpartition(y, 'KFold', k);
            
            % 初始化性能指标结果
            cv_results = struct();
            cv_results.accuracy = zeros(k, 1);
            cv_results.precision = zeros(k, 1);
            cv_results.recall = zeros(k, 1);
            cv_results.specificity = zeros(k, 1);
            cv_results.f1_score = zeros(k, 1);
            cv_results.auc = zeros(k, 1);
            cv_results.aic = zeros(k, 1);     % 新增
            cv_results.bic = zeros(k, 1);     % 新增
            cv_results.coefs = cell(k, 1);
            cv_results.fold_indices = cell(k, 1);
            cv_results.y_pred = cell(k, 1);       % 新增
            cv_results.y_test = cell(k, 1);       % 新增
            cv_results.y_pred_prob = cell(k, 1);  % 新增
            
            % 记录每个模型的变量系数分布
            n_vars = size(X, 2);
            all_coefs = zeros(k, n_vars+1); % +1是因为有截距项
            
            % 对每个折执行训练和评估
            for i = 1:k
                % 获取当前折的训练集和测试集
                train_idx = cv.training(i);
                test_idx = cv.test(i);
                
                % 存储当前折的索引
                cv_results.fold_indices{i} = struct('train', find(train_idx), 'test', find(test_idx));
                
                % 使用训练集训练逻辑回归模型
                try
                    % 使用更强大的fitglm函数
                    mdl = fitglm(X(train_idx, :), y(train_idx), 'Distribution', 'binomial', 'Link', 'logit');
                    
                    % 存储模型系数
                    coefs = mdl.Coefficients.Estimate;
                    cv_results.coefs{i} = coefs;
                    all_coefs(i, :) = coefs';
                    
                    % 使用测试集预测
                    y_pred_prob = predict(mdl, X(test_idx, :));
                    y_pred = y_pred_prob > 0.5;
                    
                    % 保存预测结果
                    cv_results.y_pred{i} = y_pred;
                    cv_results.y_test{i} = y(test_idx);
                    cv_results.y_pred_prob{i} = y_pred_prob;
                    
                    % 计算评估指标
                    y_test = y(test_idx);
                    
                    % 准确率
                    cv_results.accuracy(i) = sum(y_pred == y_test) / length(y_test);
                    
                    % 计算混淆矩阵
                    TP = sum(y_pred == 1 & y_test == 1);
                    TN = sum(y_pred == 0 & y_test == 0);
                    FP = sum(y_pred == 1 & y_test == 0);
                    FN = sum(y_pred == 0 & y_test == 1);
                    
                    % 精确率
                    if (TP + FP) > 0
                        cv_results.precision(i) = TP / (TP + FP);
                    else
                        cv_results.precision(i) = 0;
                    end
                    
                    % 召回率/敏感性
                    if (TP + FN) > 0
                        cv_results.recall(i) = TP / (TP + FN);
                    else
                        cv_results.recall(i) = 0;
                    end
                    
                    % 特异性
                    if (TN + FP) > 0
                        cv_results.specificity(i) = TN / (TN + FP);
                    else
                        cv_results.specificity(i) = 0;
                    end
                    
                    % F1分数
                    if (cv_results.precision(i) + cv_results.recall(i)) > 0
                        cv_results.f1_score(i) = 2 * (cv_results.precision(i) * cv_results.recall(i)) / (cv_results.precision(i) + cv_results.recall(i));
                    else
                        cv_results.f1_score(i) = 0;
                    end
                    
                    % AUC
                    if length(unique(y_test)) > 1 % 确保正负样本都有
                        [~, ~, ~, auc] = perfcurve(y_test, y_pred_prob, 1);
                        cv_results.auc(i) = auc;
                    else
                        cv_results.auc(i) = NaN;
                    end
                    
                    % 计算AIC和BIC - 新增
                    deviance = mdl.Deviance;
                    n_params = length(coefs);
                    n_samples = sum(train_idx);
                    
                    cv_results.aic(i) = deviance + 2 * n_params;
                    cv_results.bic(i) = deviance + log(n_samples) * n_params;
                    
                catch ME
                    logger.log_message('warning', sprintf('第%d折交叉验证失败: %s', i, ME.message));
                    cv_results.accuracy(i) = NaN;
                    cv_results.precision(i) = NaN;
                    cv_results.recall(i) = NaN;
                    cv_results.specificity(i) = NaN;
                    cv_results.f1_score(i) = NaN;
                    cv_results.auc(i) = NaN;
                    cv_results.aic(i) = NaN;
                    cv_results.bic(i) = NaN;
                    cv_results.coefs{i} = NaN(n_vars+1, 1);
                    all_coefs(i, :) = NaN(1, n_vars+1);
                end
            end
            
            % 计算每个指标的平均值和标准差
            fields = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', 'aic', 'bic'};
            for j = 1:length(fields)
                field = fields{j};
                cv_results.(['avg_' field]) = mean(cv_results.(field), 'omitnan');
                cv_results.(['std_' field]) = std(cv_results.(field), 'omitnan');
                cv_results.(['cv_' field]) = cv_results.(['std_' field]) / cv_results.(['avg_' field]); % 新增：变异系数
            end
            
            % 系数稳定性分析
            cv_results.coef_mean = mean(all_coefs, 'omitnan');
            cv_results.coef_std = std(all_coefs, 'omitnan');
            cv_results.coef_cv = abs(cv_results.coef_std ./ cv_results.coef_mean); % 变异系数
            cv_results.all_coefs = all_coefs;
            
            % 添加变量名字段
            cv_results.variables = cell(n_vars + 1, 1); % +1是因为有截距项
            cv_results.variables{1} = 'Intercept';
            for i = 1:n_vars
                if nargin > 3 && i <= length(var_names) % 检查是否传入了变量名
                    cv_results.variables{i+1} = var_names{i};
                else
                    cv_results.variables{i+1} = sprintf('Var%d', i);
                end
            end
            
            % 记录K折验证的整体情况
            logger.log_message('info', sprintf('K折交叉验证指标: 准确率=%.3f(±%.3f), 精确率=%.3f(±%.3f), 召回率=%.3f(±%.3f), F1=%.3f(±%.3f), AUC=%.3f(±%.3f), AIC=%.1f(±%.1f), BIC=%.1f(±%.1f)', ...
                cv_results.avg_accuracy, cv_results.std_accuracy, ...
                cv_results.avg_precision, cv_results.std_precision, ...
                cv_results.avg_recall, cv_results.std_recall, ...
                cv_results.avg_f1_score, cv_results.std_f1_score, ...
                cv_results.avg_auc, cv_results.std_auc, ...
                cv_results.avg_aic, cv_results.std_aic, ...
                cv_results.avg_bic, cv_results.std_bic));
                
            t_end = toc;
            logger.log_message('info', sprintf('K折交叉验证完成(K=%d)，耗时：%.2f秒', k, t_end - t_start));
        end
        
        function create_residual_analysis(results, methods, figure_dir)
            % 创建残差分析和可视化
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   figure_dir - 图形保存目录
            
            t_start = toc;
            
            % 对每种方法执行残差分析
            for i = 1:length(methods)
                method = methods{i};
                
                % 检查该方法是否适合残差分析（逻辑回归模型）
                if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                    
                    % 获取预测结果
                    if isfield(results.(method).performance, 'y_pred_prob') && ...
                       isfield(results.(method).performance, 'y_test')
                        
                        y_pred_prob = results.(method).performance.y_pred_prob;
                        y_test = results.(method).performance.y_test;
                        
                        % 合并所有Bootstrap样本的数据
                        all_probs = [];
                        all_labels = [];
                        
                        for j = 1:length(y_pred_prob)
                            if ~isempty(y_pred_prob{j}) && ~isempty(y_test{j})
                                all_probs = [all_probs; y_pred_prob{j}];
                                all_labels = [all_labels; y_test{j}];
                            end
                        end
                        
                        if ~isempty(all_probs) && ~isempty(all_labels)
                            % 计算对数几率
                            logodds = log(all_probs ./ (1 - all_probs));
                            
                            % 计算残差（皮尔森残差）
                            residuals = (all_labels - all_probs) ./ sqrt(all_probs .* (1 - all_probs));
                            
                            % 计算Deviance残差
                            deviance_residuals = sign(all_labels - all_probs) .* ...
                                sqrt(-2 * (all_labels .* log(all_probs) + (1 - all_labels) .* log(1 - all_probs)));
                            
                            % 创建残差分析图
                            visualization_module.create_residual_analysis_plot(method, all_probs, residuals, deviance_residuals, all_labels, figure_dir);
                            
                            % 创建残差汇总统计表
                            stats = struct();
                            stats.method = method;
                            stats.pearson_mean = mean(residuals);
                            stats.pearson_std = std(residuals);
                            stats.pearson_min = min(residuals);
                            stats.pearson_max = max(residuals);
                            stats.pearson_skewness = skewness(residuals);
                            stats.pearson_kurtosis = kurtosis(residuals);
                            stats.deviance_mean = mean(deviance_residuals);
                            stats.deviance_std = std(deviance_residuals);
                            stats.deviance_min = min(deviance_residuals);  % 新增
                            stats.deviance_max = max(deviance_residuals);  % 新增
                            stats.deviance_skewness = skewness(deviance_residuals);  % 新增
                            stats.deviance_kurtosis = kurtosis(deviance_residuals);  % 新增
                            
                            % 输出残差统计信息
                            logger.log_message('info', sprintf('%s方法残差统计:', method));
                            logger.log_message('info', sprintf('皮尔森残差: 均值=%.3f, 标准差=%.3f, 偏度=%.3f, 峰度=%.3f', ...
                                stats.pearson_mean, stats.pearson_std, stats.pearson_skewness, stats.pearson_kurtosis));
                            logger.log_message('info', sprintf('Deviance残差: 均值=%.3f, 标准差=%.3f, 偏度=%.3f, 峰度=%.3f', ...
                                stats.deviance_mean, stats.deviance_std, stats.deviance_skewness, stats.deviance_kurtosis));
                            
                            % 检测潜在的异常点
                            pearson_outliers = abs(residuals) > 2.5;
                            deviance_outliers = abs(deviance_residuals) > 2.5;
                            
                            if any(pearson_outliers)
                                logger.log_message('info', sprintf('%s方法检测到%d个皮尔森残差异常点(|残差|>2.5)', ...
                                    method, sum(pearson_outliers)));
                            end
                            
                            if any(deviance_outliers)
                                logger.log_message('info', sprintf('%s方法检测到%d个Deviance残差异常点(|残差|>2.5)', ...
                                    method, sum(deviance_outliers)));
                            end
                        else
                            logger.log_message('warning', sprintf('%s方法没有足够的预测数据进行残差分析', method));
                        end
                    else
                        logger.log_message('warning', sprintf('%s方法缺少预测概率或真实标签数据，无法进行残差分析', method));
                    end
                else
                    logger.log_message('info', sprintf('%s方法不适用于传统残差分析', method));
                end
            end
            
            % 创建所有方法的残差比较图
            model_module.create_residual_comparison(results, methods, figure_dir);
            
            t_end = toc;
            logger.log_message('info', sprintf('残差分析完成，耗时：%.2f秒', t_end - t_start));
        end
        
        function create_residual_comparison(results, methods, figure_dir)
            % 创建所有方法的残差比较图
            try
                % 收集所有方法的残差数据
                methods_with_residuals = {};
                all_pearson_residuals = {};
                all_deviance_residuals = {};  % 新增
                
                for i = 1:length(methods)
                    method = methods{i};
                    
                    % 检查该方法是否适合残差分析（逻辑回归模型）
                    if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                        
                        % 获取预测结果
                        if isfield(results.(method).performance, 'y_pred_prob') && ...
                           isfield(results.(method).performance, 'y_test')
                            
                            y_pred_prob = results.(method).performance.y_pred_prob;
                            y_test = results.(method).performance.y_test;
                            
                            % 合并所有Bootstrap样本的数据
                            all_probs = [];
                            all_labels = [];
                            
                            for j = 1:length(y_pred_prob)
                                if ~isempty(y_pred_prob{j}) && ~isempty(y_test{j})
                                    all_probs = [all_probs; y_pred_prob{j}];
                                    all_labels = [all_labels; y_test{j}];
                                end
                            end
                            
                            if ~isempty(all_probs) && ~isempty(all_labels)
                                % 计算皮尔森残差
                                pearson_residuals = (all_labels - all_probs) ./ sqrt(all_probs .* (1 - all_probs));
                                
                                % 计算Deviance残差
                                deviance_residuals = sign(all_labels - all_probs) .* ...
                                    sqrt(-2 * (all_labels .* log(all_probs) + (1 - all_labels) .* log(1 - all_probs)));
                                
                                % 保存方法名和残差
                                methods_with_residuals{end+1} = method;
                                all_pearson_residuals{end+1} = pearson_residuals;
                                all_deviance_residuals{end+1} = deviance_residuals;  % 新增
                            end
                        end
                    end
                end
                
                if length(methods_with_residuals) >= 2
                    visualization_module.create_residual_comparison_plots(methods_with_residuals, all_pearson_residuals, all_deviance_residuals, figure_dir);
                end
            catch ME
                logger.log_message('warning', sprintf('创建残差比较图失败: %s', ME.message));
            end
        end
        
        function coef_stability = monitor_coefficient_stability(results, methods, var_names)
            % 监控模型系数的稳定性
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   coef_stability - 系数稳定性分析结果
            
            t_start = toc;
            coef_stability = struct();
            
            for m = 1:length(methods)
                method = methods{m};
                
                % 只针对回归类模型进行系数稳定性分析
                if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                    % 获取该方法的所有模型系数
                    all_coefs = results.(method).performance.all_coefs;
                    n_models = length(all_coefs);
                    
                    % 确定最常见的变量组合
                    var_combinations = results.(method).var_combinations;
                    combo_strings = cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false);
                    [unique_combos, ~, ic] = unique(combo_strings);
                    combo_counts = accumarray(ic, 1);
                    [~, max_idx] = max(combo_counts);
                    most_frequent_combo = unique_combos{max_idx};
                    combo_indices = str2num(['[' most_frequent_combo(1:end-1) ']']);
                    
                    % 找出使用了最常见组合的模型索引
                    common_combo_indices = find(ic == max_idx);
                    
                    % 降低模型数量要求
                    if length(common_combo_indices) >= 5
                        % 提取这些模型的系数
                        % 确保common_coefs是一个有效的数值矩阵
                        common_coefs = [];
                        valid_coefs_count = 0;
                        
                        for cidx = 1:length(common_combo_indices)
                            model_idx = common_combo_indices(cidx);
                            coef = all_coefs{model_idx};
                            
                            % 检查coef是有效的数值向量
                            if isnumeric(coef) && ~isempty(coef) && ~any(isnan(coef))
                                if isempty(common_coefs)
                                    common_coefs = coef';  % 转置为行向量
                                    valid_coefs_count = 1;
                                else
                                    % 确保长度匹配
                                    if length(coef) == size(common_coefs, 2)
                                        common_coefs = [common_coefs; coef'];
                                        valid_coefs_count = valid_coefs_count + 1;
                                    end
                                end
                            end
                        end
                        
                        % 如果没有有效系数，则跳过
                        if valid_coefs_count < 2
                            logger.log_message('warning', sprintf('%s方法没有足够的有效系数，跳过系数稳定性分析', method));
                            coef_stability.(method) = struct('status', 'insufficient_valid_data');
                            continue;
                        end
                        
                        % 检查系数维度是否与变量组合匹配
                        expected_dim = length(combo_indices) + 1; % +1是截距项
                        actual_dim = size(common_coefs, 2);
                        
                        logger.log_message('info', sprintf('%s方法的系数维度=%d，变量组合维度=%d', ...
                            method, actual_dim, expected_dim));
                            
                        % 如果维度不匹配，尝试调整
                        if actual_dim ~= expected_dim
                            logger.log_message('warning', sprintf('%s方法的系数维度(%d)与变量组合维度(%d)不匹配，尝试调整', ...
                                method, actual_dim, expected_dim));
                            
                            % 根据实际情况调整
                            if actual_dim > expected_dim
                                % 如果系数数量多于变量数量，取前面的部分（截距项和选择的变量）
                                logger.log_message('info', sprintf('截取系数维度从%d到%d', actual_dim, expected_dim));
                                common_coefs = common_coefs(:, 1:expected_dim);
                            elseif actual_dim < expected_dim && actual_dim > 0
                                % 如果系数数量少于变量数量但不为零，使用可用的系数（可能会导致变量名不匹配）
                                logger.log_message('warning', sprintf('系数数量不足，统计可用的%d个系数', actual_dim));
                                expected_dim = actual_dim;
                            else
                                % 极端情况：没有有效系数
                                logger.log_message('warning', sprintf('%s方法没有有效系数', method));
                                coef_stability.(method) = struct('status', 'no_valid_coefficients');
                                continue;
                            end
                        end
                        
                        % 计算系数统计量
                        coef_mean = mean(common_coefs, 1);
                        coef_std = std(common_coefs, 0, 1);
                        
                        % 变异系数计算时需要处理零和接近零的值
                        coef_cv = zeros(size(coef_mean));
                        for i = 1:length(coef_mean)
                            if abs(coef_mean(i)) > 1e-6  % 避免除以接近零的值
                                coef_cv(i) = abs(coef_std(i) / coef_mean(i));
                            else
                                if coef_std(i) > 1e-6  % 均值接近零但标准差不小
                                    coef_cv(i) = 999;  % 表示高变异性
                                else  % 均值和标准差都接近零
                                    coef_cv(i) = 0;    % 表示稳定（都是零）
                                end
                            end
                        end
                        
                        % 创建变量列表（包括截距）
                        if expected_dim <= 1
                            var_list = {'Intercept'};
                        else
                            if length(combo_indices) >= expected_dim - 1
                                var_list = ['Intercept'; var_names(combo_indices(1:expected_dim-1))];
                            else
                                % 如果变量组合索引不足，使用通用变量名
                                var_list = cell(expected_dim, 1);
                                var_list{1} = 'Intercept';
                                for i = 2:expected_dim
                                    if i-1 <= length(combo_indices)
                                        var_list{i} = var_names{combo_indices(i-1)};
                                    else
                                        var_list{i} = sprintf('Var%d', i-1);
                                    end
                                end
                            end
                        end
                        
                        % 确保所有变量长度一致
                        min_len = min([length(var_list), length(coef_mean), length(coef_std), length(coef_cv)]);
                        
                        % 截取所有数组到相同长度
                        var_list = var_list(1:min_len);
                        coef_mean = coef_mean(1:min_len);
                        coef_std = coef_std(1:min_len);
                        coef_cv = coef_cv(1:min_len);
                        
                        % 创建系数稳定性表
                        % 使用cell数组处理不同类型数据
                        table_data_cell = cell(length(var_list), 4);
                        for i = 1:length(var_list)
                            table_data_cell{i, 1} = var_list{i};
                            table_data_cell{i, 2} = coef_mean(i);
                            table_data_cell{i, 3} = coef_std(i);
                            table_data_cell{i, 4} = coef_cv(i);
                        end
                        
                        % 创建表格
                        table_data = cell2table(table_data_cell, 'VariableNames', {'Variable', 'Mean', 'StdDev', 'CV'});
                        
                        % 存储系数稳定性结果
                        coef_stability.(method).mean = coef_mean;
                        coef_stability.(method).std = coef_std;
                        coef_stability.(method).cv = coef_cv;
                        coef_stability.(method).variables = var_list;
                        coef_stability.(method).table = table_data;
                        coef_stability.(method).all_coefs = common_coefs;
                        
                        % 记录系数稳定性情况
                        logger.log_message('info', sprintf('%s方法的系数稳定性分析完成，分析了%d个模型', method, length(common_combo_indices)));
                        
                        % 识别不稳定的系数 (CV > 0.5)
                        unstable_idx = find(coef_cv > 0.5);
                        if ~isempty(unstable_idx)
                            unstable_vars = var_list(unstable_idx);
                            logger.log_message('warning', sprintf('%s方法中检测到不稳定系数：', method));
                            for i = 1:length(unstable_idx)
                                logger.log_message('warning', sprintf('  %s: CV=%.2f', unstable_vars{i}, coef_cv(unstable_idx(i))));
                            end
                        else
                            logger.log_message('info', sprintf('%s方法的所有系数都表现稳定 (CV <= 0.5)', method));
                        end
                    else
                        logger.log_message('warning', sprintf('%s方法没有足够的模型使用最常见变量组合(只有%d个，需要至少2个)，跳过系数稳定性分析', ...
                            method, length(common_combo_indices)));
                        coef_stability.(method) = struct('status', 'insufficient_data');
                    end
                else
                    % 对于非回归类模型
                    logger.log_message('info', sprintf('%s方法不适用于传统系数稳定性分析', method));
                    coef_stability.(method) = struct('status', 'not_applicable');
                end
            end
            
            t_end = toc;
            logger.log_message('info', sprintf('模型系数稳定性监控完成，耗时：%.2f秒', t_end - t_start));
        end
        
        function param_stats = calculate_parameter_statistics(results, methods, var_names)
            % 计算模型参数的置信区间和p值（同时基于BCa和t分布）
            % 输入:
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   param_stats - 参数统计结果
            
            t_start = toc;
            param_stats = struct();
            
            % 对每种方法计算参数统计量
            for m = 1:length(methods)
                method = methods{m};
                
                % 只针对回归类模型进行参数统计分析
                if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                    % 获取该方法的所有模型系数
                    all_coefs = results.(method).performance.all_coefs;
                    n_models = length(all_coefs);
                    
                    % 确定最常见的变量组合
                    var_combinations = results.(method).var_combinations;
                    combo_strings = cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false);
                    [unique_combos, ~, ic] = unique(combo_strings);
                    combo_counts = accumarray(ic, 1);
                    [~, max_idx] = max(combo_counts);
                    most_frequent_combo = unique_combos{max_idx};
                    combo_indices = str2num(['[' most_frequent_combo(1:end-1) ']']);
                    
                    % 找出使用了最常见组合的模型索引
                    common_combo_indices = find(ic == max_idx);
                    
                    % 降低模型数量要求
                    if length(common_combo_indices) >= 2
                        % 提取这些模型的系数
                        common_coefs = [];
                        valid_coefs_count = 0;
                        
                        for cidx = 1:length(common_combo_indices)
                            model_idx = common_combo_indices(cidx);
                            coef = all_coefs{model_idx};
                            
                            % 检查coef是有效的数值向量
                            if isnumeric(coef) && ~isempty(coef) && ~any(isnan(coef))
                                if isempty(common_coefs)
                                    common_coefs = coef';  % 转置为行向量
                                    valid_coefs_count = 1;
                                else
                                    % 确保长度匹配
                                    if length(coef) == size(common_coefs, 2)
                                        common_coefs = [common_coefs; coef'];
                                        valid_coefs_count = valid_coefs_count + 1;
                                    end
                                end
                            end
                        end
                        
                        % 如果没有有效系数，则跳过
                        if valid_coefs_count < 2
                            logger.log_message('warning', sprintf('%s方法没有足够的有效系数，跳过参数统计分析', method));
                            param_stats.(method) = struct('status', 'insufficient_valid_data');
                            continue;
                        end
                        
                        % 检查维度是否匹配
                        expected_dim = length(combo_indices) + 1;  % +1是截距项
                        actual_dim = size(common_coefs, 2);
                        
                        % 调整维度（如果需要）
                        if actual_dim ~= expected_dim
                            logger.log_message('warning', sprintf('%s方法的系数维度(%d)与变量组合维度(%d)不匹配，尝试调整', ...
                                method, actual_dim, expected_dim));
                            
                            if actual_dim > expected_dim
                                common_coefs = common_coefs(:, 1:expected_dim);
                            elseif actual_dim < expected_dim && actual_dim > 0
                                expected_dim = actual_dim;
                            else
                                param_stats.(method) = struct('status', 'no_valid_coefficients');
                                continue;
                            end
                        end
                        
                        % 创建变量列表
                        if expected_dim <= 1
                            var_list = {'Intercept'};
                        else
                            if length(combo_indices) >= expected_dim - 1
                                var_list = ['Intercept'; var_names(combo_indices(1:expected_dim-1))];
                            else
                                var_list = cell(expected_dim, 1);
                                var_list{1} = 'Intercept';
                                for i = 2:expected_dim
                                    if i-1 <= length(combo_indices)
                                        var_list{i} = var_names{combo_indices(i-1)};
                                    else
                                        var_list{i} = sprintf('Var%d', i-1);
                                    end
                                end
                            end
                        end
                        
                        % 确保所有变量长度一致
                        min_len = min([length(var_list), size(common_coefs, 2)]);
                        var_list = var_list(1:min_len);
                        
                        % 1. 基于t分布计算
                        % 计算系数统计量
                        coef_mean = mean(common_coefs, 1);
                        coef_std = std(common_coefs, 0, 1);
                        
                        % 样本数量
                        n_samples = size(common_coefs, 1);
                        
                        % 计算95%置信区间
                        t_critical = tinv(0.975, n_samples - 1); % 双侧95%置信度的t值
                        margin_error = t_critical * coef_std / sqrt(n_samples);
                        t_ci_lower = coef_mean - margin_error;
                        t_ci_upper = coef_mean + margin_error;
                        
                        % 计算p值 (双侧t检验，H0：系数=0)
                        t_stat = coef_mean ./ (coef_std / sqrt(n_samples));
                        t_p_values = 2 * (1 - tcdf(abs(t_stat), n_samples - 1));
                        
                        % 2. 基于BCa方法计算Bootstrap置信区间
                        bca_ci_lower = zeros(1, min_len);
                        bca_ci_upper = zeros(1, min_len);
                        
                        try
                            for i = 1:min_len
                                % 提取当前参数的所有Bootstrap样本值
                                theta_boot = common_coefs(:, i);
                                
                                % 计算BCa置信区间
                                [bca_ci_lower(i), bca_ci_upper(i)] = model_module.calculate_bca_ci(theta_boot, 0.05);
                            end
                        catch ME
                            logger.log_message('warning', sprintf('BCa置信区间计算失败: %s，使用基本Bootstrap置信区间', ME.message));
                            % 使用基本Bootstrap置信区间作为备选
                            alpha = 0.05;
                            lower_percentile = 100 * alpha / 2;
                            upper_percentile = 100 * (1 - alpha / 2);
                            for i = 1:min_len
                                theta_boot = common_coefs(:, i);
                                bca_ci_lower(i) = prctile(theta_boot, lower_percentile);
                                bca_ci_upper(i) = prctile(theta_boot, upper_percentile);
                            end
                        end
                        
                        % 显著性标记
                        significance = cell(size(t_p_values));
                        for i = 1:length(t_p_values)
                            if t_p_values(i) < 0.001
                                significance{i} = '***';
                            elseif t_p_values(i) < 0.01
                                significance{i} = '**';
                            elseif t_p_values(i) < 0.05
                                significance{i} = '*';
                            elseif t_p_values(i) < 0.1
                                significance{i} = '.';
                            else
                                significance{i} = '';
                            end
                        end
                        
                        % 创建参数统计表
                        table_data_cell = cell(length(var_list), 9);
                        for i = 1:length(var_list)
                            table_data_cell{i, 1} = var_list{i};
                            table_data_cell{i, 2} = coef_mean(i);
                            table_data_cell{i, 3} = coef_std(i);
                            table_data_cell{i, 4} = t_ci_lower(i);
                            table_data_cell{i, 5} = t_ci_upper(i);
                            table_data_cell{i, 6} = bca_ci_lower(i);
                            table_data_cell{i, 7} = bca_ci_upper(i);
                            table_data_cell{i, 8} = t_p_values(i);
                            table_data_cell{i, 9} = significance{i};
                        end
                        
                        % 创建表格
                        table_data = cell2table(table_data_cell, 'VariableNames', {'Variable', 'Estimate', 'StdError', ...
                            'CI_Lower_t', 'CI_Upper_t', 'CI_Lower_BCa', 'CI_Upper_BCa', 'p_value', 'Significance'});
                        
                        % 存储参数统计结果
                        param_stats.(method).mean = coef_mean;
                        param_stats.(method).std = coef_std;
                        param_stats.(method).t_ci_lower = t_ci_lower;
                        param_stats.(method).t_ci_upper = t_ci_upper;
                        param_stats.(method).bca_ci_lower = bca_ci_lower;
                        param_stats.(method).bca_ci_upper = bca_ci_upper;
                        param_stats.(method).p_values = t_p_values;
                        param_stats.(method).significance = significance;
                        param_stats.(method).variables = var_list;
                        param_stats.(method).table = table_data;
                        param_stats.(method).n_samples = n_samples;
                        
                        % 记录参数统计情况
                        logger.log_message('info', sprintf('%s方法的参数统计分析完成，分析了%d个模型', method, n_samples));
                        
                        % 输出显著的参数
                        sig_idx = find(t_p_values < 0.05);
                        if ~isempty(sig_idx)
                            logger.log_message('info', sprintf('%s方法中检测到显著参数 (p < 0.05)：', method));
                            for i = 1:length(sig_idx)
                                logger.log_message('info', sprintf('  %s: 估计值=%.4f, t-CI=[%.4f,%.4f], BCa-CI=[%.4f,%.4f], p=%.4f %s', ...
                                    var_list{sig_idx(i)}, coef_mean(sig_idx(i)), ...
                                    t_ci_lower(sig_idx(i)), t_ci_upper(sig_idx(i)), ...
                                    bca_ci_lower(sig_idx(i)), bca_ci_upper(sig_idx(i)), ...
                                    t_p_values(sig_idx(i)), significance{sig_idx(i)}));
                            end
                        else
                            logger.log_message('warning', sprintf('%s方法没有检测到显著参数 (p < 0.05)', method));
                        end
                    else
                        logger.log_message('warning', sprintf('%s方法没有足够的模型使用最常见变量组合(只有%d个，需要至少2个)，跳过参数统计分析', ...
                            method, length(common_combo_indices)));
                        param_stats.(method) = struct('status', 'insufficient_data');
                    end
                else
                    % 对于非回归类模型
                    logger.log_message('info', sprintf('%s方法不适用于传统参数统计分析', method));
                    param_stats.(method) = struct('status', 'not_applicable');
                end
            end
            
            t_end = toc;
            logger.log_message('info', sprintf('模型参数统计分析完成，耗时：%.2f秒', t_end - t_start));
        end
        
        function [lower, upper] = calculate_bca_ci(theta_boot, alpha)
            % 计算BCa (偏差校正加速) Bootstrap置信区间
            % 输入:
            %   theta_boot - Bootstrap样本
            %   alpha - 显著性水平 (默认0.05)
            % 输出:
            %   lower - 下界
            %   upper - 上界
            
            n = length(theta_boot);
            theta_mean = mean(theta_boot);
            
            % 计算偏差校正因子z0
            n_less = sum(theta_boot < theta_mean);
            z0 = norminv(n_less / n);
            
            % 计算加速因子a (使用jackknife方法)
            theta_jack = zeros(n, 1);
            for i = 1:n
                theta_jack(i) = mean(theta_boot([1:i-1, i+1:n]));
            end
            theta_jack_mean = mean(theta_jack);
            num = sum((theta_jack_mean - theta_jack).^3);
            den = 6 * (sum((theta_jack_mean - theta_jack).^2)).^(3/2);
            if den == 0
                a = 0;
            else
                a = num / den;
            end
            
            % 计算BCa置信区间
            z_alpha_lo = norminv(alpha/2);
            z_alpha_hi = norminv(1-alpha/2);
            
            % 计算BCa调整后的alpha
            alpha_1_adj = normcdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)));
            alpha_2_adj = normcdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)));
            
            % 根据调整后的alpha找出置信区间
            lower = prctile(theta_boot, 100 * alpha_1_adj);
            upper = prctile(theta_boot, 100 * alpha_2_adj);
        end
        
        function var_contribution = evaluate_variable_contribution(X, y, results, methods, var_names)
            % 评估每个变量对模型的贡献
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   results - 结果结构
            %   methods - 方法名称
            %   var_names - 变量名称
            % 输出:
            %   var_contribution - 变量贡献分析结果
            
            t_start = toc;
            var_contribution = struct();
            n_vars = length(var_names);
            
            % 1. 全局变量重要性分析 - 使用全数据集
            logger.log_message('info', '开始全局变量重要性分析...');
            
            % 1.1 基于相关性的分析
            try
                % 计算相关系数和p值
                [corr_coef, corr_pval] = corr(X, y, 'Type', 'Pearson');
                
                % 计算偏相关系数
                partial_corr = zeros(n_vars, 1);
                partial_pval = zeros(n_vars, 1);
                
                % 计算每个变量的偏相关系数
                for i = 1:n_vars
                    other_vars = setdiff(1:n_vars, i);
                    if ~isempty(other_vars)
                        % 残差化
                        mdl_x = fitlm(X(:, other_vars), X(:, i));
                        mdl_y = fitlm(X(:, other_vars), y);
                        
                        x_resid = mdl_x.Residuals.Raw;
                        y_resid = mdl_y.Residuals.Raw;
                        
                        % 计算残差间的相关性
                        [r, p] = corr(x_resid, y_resid);
                        partial_corr(i) = r;
                        partial_pval(i) = p;
                    else
                        % 如果只有一个变量，偏相关等于普通相关
                        partial_corr(i) = corr_coef(i);
                        partial_pval(i) = corr_pval(i);
                    end
                end
                
                % 保存结果
                var_contribution.correlation = table(var_names, corr_coef, corr_pval, partial_corr, partial_pval, ...
                    'VariableNames', {'Variable', 'Correlation', 'Corr_pvalue', 'PartialCorr', 'Partial_pvalue'});
                
                logger.log_message('info', '相关性分析完成');
            catch ME
                logger.log_message('warning', sprintf('相关性分析失败: %s', ME.message));
            end
            
            % 1.2 基于模型的重要性分析 - 执行变量重要性分析
            model_module.analyze_model_based_importance(X, y, var_names, var_contribution);
            
            % 2. 方法特定变量贡献分析
            logger.log_message('info', '开始方法特定变量贡献分析...');
            
            var_contribution.methods = struct();
            for m = 1:length(methods)
                method = methods{m};
                var_contribution.methods.(method) = struct();
                
                % 获取该方法选择的变量
                selected_vars = find(results.(method).selected_vars);
                selected_names = var_names(selected_vars);
                var_contribution.methods.(method).selected_vars = selected_vars;
                var_contribution.methods.(method).selected_names = selected_names;
                
                % 变量选择频率
                var_contribution.methods.(method).var_freq = results.(method).var_freq;
                
                % 对于回归类方法，计算系数贡献
                if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                    try
                        % 构建贡献表
                        method_mdl = fitglm(X(:, selected_vars), y, 'Distribution', 'binomial', 'Link', 'logit');
                        
                        % 模型参数
                        coefs = method_mdl.Coefficients.Estimate(2:end); % 排除截距
                        pvals = method_mdl.Coefficients.pValue(2:end);
                        
                        % 标准化系数
                        std_X_sel = std(X(:, selected_vars));
                        std_coefs = coefs .* std_X_sel';
                        
                        % 计算相对贡献 (基于系数绝对值)
                        abs_coefs = abs(std_coefs);
                        rel_contrib = abs_coefs / sum(abs_coefs) * 100;
                        
                        % 系数符号
                        coef_sign = sign(coefs);
                        effect_dir = cell(length(coef_sign), 1);
                        for i = 1:length(coef_sign)
                            if coef_sign(i) > 0
                                effect_dir{i} = '正向';
                            elseif coef_sign(i) < 0
                                effect_dir{i} = '负向';
                            else
                                effect_dir{i} = '无';
                            end
                        end
                        
                        % 显著性标记
                        sig_marks = cell(length(pvals), 1);
                        for i = 1:length(pvals)
                            if pvals(i) < 0.001
                                sig_marks{i} = '***';
                            elseif pvals(i) < 0.01
                                sig_marks{i} = '**';
                            elseif pvals(i) < 0.05
                                sig_marks{i} = '*';
                            elseif pvals(i) < 0.1
                                sig_marks{i} = '.';
                            else
                                sig_marks{i} = '';
                            end
                        end
                        
                        % 创建贡献表
                        contrib_table = table(selected_names, coefs, pvals, std_coefs, rel_contrib, effect_dir, sig_marks, ...
                            'VariableNames', {'Variable', 'Coefficient', 'p_value', 'Std_Coefficient', ...
                            'Relative_Contribution', 'Effect_Direction', 'Significance'});
                        
                        % 按相对贡献排序
                        contrib_table = sortrows(contrib_table, 'Relative_Contribution', 'descend');
                        
                        % 保存结果
                        var_contribution.methods.(method).contribution_table = contrib_table;
                        
                        logger.log_message('info', sprintf('%s方法的变量贡献分析完成', method));
                    catch ME
                        logger.log_message('warning', sprintf('%s方法的变量贡献分析失败: %s', method, ME.message));
                    end
                elseif strcmpi(method, 'randomforest')
                    % 对于随机森林，使用变量重要性
                    try
                        if exist('TreeBagger', 'file') && ~isempty(selected_vars)
                            % 只使用选定的变量训练森林
                            forest = TreeBagger(100, X(:, selected_vars), y, 'Method', 'classification', ...
                                'OOBPredictorImportance', 'on');
                            
                            rf_importance = forest.OOBPermutedPredictorDeltaError;
                            
                            % 标准化重要性分数
                            norm_importance = rf_importance / sum(rf_importance) * 100;
                            
                            % 创建贡献表
                            contrib_table = table(selected_names, rf_importance', norm_importance', ...
                                'VariableNames', {'Variable', 'Importance', 'Relative_Contribution'});
                            
                            % 按相对贡献排序
                            contrib_table = sortrows(contrib_table, 'Relative_Contribution', 'descend');
                            
                            % 保存结果
                            var_contribution.methods.(method).contribution_table = contrib_table;
                            
                            logger.log_message('info', sprintf('%s方法的变量贡献分析完成', method));
                        else
                            logger.log_message('warning', sprintf('%s方法的变量贡献分析失败：TreeBagger不可用或无选定变量', method));
                        end
                    catch ME
                        logger.log_message('warning', sprintf('%s方法的变量贡献分析失败: %s', method, ME.message));
                    end
                end
            end
            
            % 3. 综合变量重要性排名
            model_module.calculate_overall_variable_importance(var_names, results, methods, var_contribution);
            
            t_end = toc;
            logger.log_message('info', sprintf('变量贡献评估完成，耗时：%.2f秒', t_end - t_start));
        end
        
        function analyze_model_based_importance(X, y, var_names, var_contribution)
            % 基于模型的变量重要性分析
            % 输入:
            %   X - 自变量矩阵
            %   y - 因变量
            %   var_names - 变量名称
            %   var_contribution - 变量贡献分析结果结构，会被修改
            
            % 1.2.1 逻辑回归模型
            try
                mdl_logistic = fitglm(X, y, 'Distribution', 'binomial', 'Link', 'logit');
                coefs = mdl_logistic.Coefficients.Estimate(2:end); % 排除截距
                pvals = mdl_logistic.Coefficients.pValue(2:end);
                
                % 标准化系数 (使用标准差缩放)
                std_X = std(X);
                std_coefs = coefs .* std_X';
                
                % 基于Wald统计量的重要性
                wald_stats = (coefs ./ mdl_logistic.Coefficients.SE(2:end)).^2;
                
                % 保存结果
                var_contribution.logistic = table(var_names, coefs, pvals, std_coefs, wald_stats, ...
                    'VariableNames', {'Variable', 'Coefficient', 'p_value', 'Std_Coefficient', 'Wald_Statistic'});
                
                logger.log_message('info', '逻辑回归变量重要性分析完成');
            catch ME
                logger.log_message('warning', sprintf('逻辑回归变量重要性分析失败: %s', ME.message));
            end
            
            % 1.2.2 随机森林模型
            try
                if exist('TreeBagger', 'file')
                    % 移除可能导致问题的并行控制选项
                    parallelOptions = statset('UseParallel', true); % 移除UseSubstreams选项
                    
                    forest = TreeBagger(100, X, y, 'Method', 'classification', ...
                        'OOBPredictorImportance', 'on', ...
                        'MinLeafSize', max(1, floor(size(X,1)/50)), ...
                        'Options', parallelOptions);
                    
                    rf_importance = forest.OOBPermutedPredictorDeltaError;
                    
                    % 标准化重要性分数
                    norm_importance = rf_importance / sum(rf_importance);
                    
                    % 保存结果
                    var_contribution.randomforest = table(var_names, rf_importance', norm_importance', ...
                        'VariableNames', {'Variable', 'Importance', 'Normalized_Importance'});
                    
                    logger.log_message('info', '随机森林变量重要性分析完成');
                else
                    logger.log_message('warning', 'TreeBagger不可用，跳过随机森林变量重要性分析');
                end
            catch ME
                logger.log_message('warning', sprintf('随机森林变量重要性分析失败: %s', ME.message));
            end
        end
        
        function calculate_overall_variable_importance(var_names, results, methods, var_contribution)
            % 计算综合变量重要性排名
            % 输入:
            %   var_names - 变量名称
            %   results - 结果结构
            %   methods - 方法名称
            %   var_contribution - 变量贡献分析结果结构，会被修改
            
            logger.log_message('info', '计算综合变量重要性排名...');
            try
                % 收集所有方法的变量选择频率
                n_vars = length(var_names);
                all_freqs = zeros(n_vars, length(methods));
                for m = 1:length(methods)
                    method = methods{m};
                    all_freqs(:, m) = results.(method).var_freq;
                end
                
                % 计算平均选择频率
                avg_freq = mean(all_freqs, 2);
                
                % 计算基于选择频率的变量重要性排名
                [~, freq_rank] = sort(avg_freq, 'descend');
                freq_rank_score = n_vars + 1 - (1:n_vars)';
                freq_rank_score = freq_rank_score(freq_rank);
                
                % 如果有相关性分析结果，合并它
                if isfield(var_contribution, 'correlation')
                    corr_abs = abs(var_contribution.correlation.Correlation);
                    [~, corr_rank] = sort(corr_abs, 'descend');
                    corr_rank_score = n_vars + 1 - (1:n_vars)';
                    corr_rank_score = corr_rank_score(corr_rank);
                else
                    corr_rank_score = zeros(n_vars, 1);
                end
                
                % 如果有逻辑回归分析结果，合并它
                if isfield(var_contribution, 'logistic')
                    log_abs = abs(var_contribution.logistic.Std_Coefficient);
                    [~, log_rank] = sort(log_abs, 'descend');
                    log_rank_score = n_vars + 1 - (1:n_vars)';
                    log_rank_score = log_rank_score(log_rank);
                else
                    log_rank_score = zeros(n_vars, 1);
                end
                
                % 如果有随机森林分析结果，合并它
                if isfield(var_contribution, 'randomforest')
                    rf_imp = var_contribution.randomforest.Importance;
                    [~, rf_rank] = sort(rf_imp, 'descend');
                    rf_rank_score = n_vars + 1 - (1:n_vars)';
                    rf_rank_score = rf_rank_score(rf_rank);
                else
                    rf_rank_score = zeros(n_vars, 1);
                end
                
                % 计算综合分数 - 使用加权平均
                combined_score = 0.4 * freq_rank_score + 0.2 * corr_rank_score + 0.2 * log_rank_score + 0.2 * rf_rank_score;
                
                % 创建综合重要性表
                [sorted_score, score_idx] = sort(combined_score, 'descend');
                sorted_vars = var_names(score_idx);
                
                % 计算归一化重要性
                norm_score = sorted_score / sum(sorted_score) * 100;
                
                % 创建表格
                overall_importance = table(sorted_vars, sorted_score, norm_score, ...
                    'VariableNames', {'Variable', 'Overall_Score', 'Normalized_Importance'});
                
                % 保存结果
                var_contribution.overall_importance = overall_importance;
                
                logger.log_message('info', '综合变量重要性排名计算完成');
                
                % 输出前5个最重要的变量
                top_n = min(5, n_vars);
                logger.log_message('info', '前5个最重要的变量:');
                for i = 1:top_n
                    logger.log_message('info', sprintf('  %d. %s (重要性: %.2f%%)', i, overall_importance.Variable{i}, overall_importance.Normalized_Importance(i)));
                end
            catch ME
                logger.log_message('warning', sprintf('计算综合变量重要性排名失败: %s', ME.message));
            end
        end
    
        function params = get_model_parameters(models, var_names)
        % 从模型中提取参数
        % 输入:
        %   models - 训练好的模型数组
        %   var_names - 变量名称
        % 输出:
        %   params - 参数结构体数组
        
        % 初始化参数结构体数组
        n_models = length(models);
        params = cell(n_models, 1);
        
        for i = 1:n_models
            model = models{i};
            
            if isempty(model)
                % 如果模型为空，设置为空参数
                params{i} = [];
                continue;
            end
            
            % 根据模型类型提取参数
            if isa(model, 'GeneralizedLinearModel')
                % 对于广义线性模型
                try
                    coefs = model.Coefficients.Estimate;
                    pvals = model.Coefficients.pValue;
                    se = model.Coefficients.SE;
                    tstat = model.Coefficients.tStat;
                    
                    % 获取变量名称
                    row_names = model.Coefficients.Properties.RowNames;
                    
                    % 确保所有数据有相同的行数
                    n_rows = length(coefs);
                    if length(row_names) ~= n_rows
                        % 如果变量名行数不匹配，创建通用名称
                        row_names = cell(n_rows, 1);
                        for j = 1:n_rows
                            if j == 1
                                row_names{j} = 'Intercept';
                            else
                                row_names{j} = sprintf('Var%d', j-1);
                            end
                        end
                    end
                    
                    % 确保所有向量都是列向量
                    coefs = reshape(coefs, [], 1);
                    pvals = reshape(pvals, [], 1);
                    se = reshape(se, [], 1);
                    tstat = reshape(tstat, [], 1);
                    
                    % 创建参数表，使用cell2table确保尺寸匹配
                    data_cell = [row_names, num2cell(coefs), num2cell(se), num2cell(tstat), num2cell(pvals)];
                    coef_table = cell2table(data_cell, 'VariableNames', {'Variable', 'Estimate', 'StdError', 'tStat', 'pValue'});
                    
                    % 保存参数
                    params{i} = struct('type', 'glm', 'coef_table', coef_table, 'coefs', coefs);
                catch ME
                    % 如果表格创建失败，提供一个简化版本
                    logger.log_message('warning', sprintf('创建GLM参数表失败: %s', ME.message));
                    params{i} = struct('type', 'glm', 'coefs', model.Coefficients.Estimate, 'error', ME.message);
                end
                
            elseif isa(model, 'TreeBagger')
                % 对于随机森林
                try
                    if isempty(model.OOBPermutedPredictorDeltaError)
                        importance = zeros(size(model.X, 2), 1);
                    else
                        importance = model.OOBPermutedPredictorDeltaError;
                    end
                    
                    % 确保重要性是列向量
                    importance = reshape(importance, [], 1);
                    n_vars = length(importance);
                    
                    % 准备变量名称
                    var_labels = cell(n_vars, 1);
                    if nargin > 1 && ~isempty(var_names) && length(var_names) >= n_vars
                        for j = 1:n_vars
                            var_labels{j} = var_names{j};
                        end
                    else
                        for j = 1:n_vars
                            var_labels{j} = sprintf('Var%d', j);
                        end
                    end
                    
                    % 创建参数表，使用cell2table确保尺寸匹配
                    data_cell = [var_labels, num2cell(importance)];
                    imp_table = cell2table(data_cell, 'VariableNames', {'Variable', 'Importance'});
                    
                    % 保存参数
                    params{i} = struct('type', 'rf', 'importance_table', imp_table, 'importance', importance);
                catch ME
                    % 如果表格创建失败，提供一个简化版本
                    logger.log_message('warning', sprintf('创建RF参数表失败: %s', ME.message));
                    params{i} = struct('type', 'rf', 'importance', model.OOBPermutedPredictorDeltaError, 'error', ME.message);
                end
                
            elseif isstruct(model) && isfield(model, 'Coefficients')
                % 处理其他可能的模型类型，以结构体形式存储
                params{i} = struct('type', 'other', 'coefs', model.Coefficients);
                
            else
                % 未知模型类型
                logger.log_message('warning', sprintf('未知模型类型，无法提取参数 (模型类: %s)', class(model)));
                params{i} = struct('type', 'unknown');
            end
        end
        
        end
    end
end