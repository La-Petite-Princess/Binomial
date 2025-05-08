function Binomial(filename)
% 二元逻辑回归模型的完整实现 - 为MacBook Pro 2019 i9-9980HK优化
% 增强版：增加k折交叉验证、多指标监控、系数稳定性评估、参数置信区间和变量贡献分析
% 增强版2.0：增加BCa置信区间、箱线图、PR曲线、校准曲线、混淆矩阵、AIC/BIC评价、残差分析、累计方差
% 版本3.0：优化日志系统，增加日志级别控制，简化图形保存日志
% 版本4.0: 全面重构，更强模块化设计，提高代码可维护性
% 输入:
%   filename - 包含数据的.mat文件路径
% 示例:
%   Binomial('mydata.mat');

% 开始计时
tic;

% 初始化系统
config = initialize_system();

try
    % 加载和预处理数据
    [data_raw, data_processed, valid_rows] = data_module.load_and_preprocess(filename);
    
    % 准备变量
    [X, y, var_names, group_means] = data_module.prepare_variables(data_processed);
    
    % 检查多重共线性
    [X_final, vif_values, removed_vars] = feature_module.check_collinearity(X, var_names);
    
    % 分析变量相关性
    pca_results = feature_module.analyze_variable_correlations(X_final, var_names(~removed_vars));
    
    % 生成Bootstrap样本
    [train_indices, test_indices] = sampling_module.bootstrap_sampling(y, 0.8, 100);
    
    % 执行交叉验证
    k_value = 10;
    cv_results = model_module.k_fold_cross_validation(X_final, y, k_value, var_names(~removed_vars));
    visualization_module.create_kfold_performance_visualization(cv_results, config.figure_dir);
    
    % 使用不同方法进行变量选择并评估模型
    methods = {'stepwise', 'lasso', 'ridge', 'elasticnet', 'randomforest'};
    results = model_module.perform_variable_selection_and_modeling(X_final, y, train_indices, test_indices, methods, var_names(~removed_vars));
    
    % 进行模型系数稳定性监控
    coef_stability = model_module.monitor_coefficient_stability(results, methods, var_names(~removed_vars));
    
    % 计算模型参数统计
    param_stats = model_module.calculate_parameter_statistics(results, methods, var_names(~removed_vars));
    
    % 评估变量贡献
    var_contribution = model_module.evaluate_variable_contribution(X_final, y, results, methods, var_names(~removed_vars));
    
    % 执行残差分析
    model_module.create_residual_analysis(results, methods, config.figure_dir);
    
    % 保存结果
    reporting_module.save_enhanced_results(results, var_names, group_means, cv_results, coef_stability, param_stats, var_contribution);
    
    % 统计总耗时
    total_time = toc;
    logger.log_message('info', sprintf('分析完成！所有结果已保存到results文件夹，总耗时：%.2f秒', total_time));
    
    % 输出并行性能统计
    reporting_module.report_parallel_performance();
    
catch ME
    logger.log_message('error', sprintf('执行过程中发生错误：%s\n%s', ME.message, getReport(ME)));
end

% 清理资源
finalize_system();

end

function config = initialize_system()
% 初始化系统配置
% 输出:
%   config - 配置结构体

% 设置随机数种子以确保结果可重复
rng(42);

% 设置日志级别
logger.set_log_level('info'); % 可选值: 'debug', 'info', 'warning', 'error'

% 系统信息收集和输出
logger.log_system_info();

% 工具箱检查
utils.check_toolboxes();

% 配置并行池
config.parallel_pool = parallel_module.setup_parallel_pool();

% 创建输出目录
config.result_dir = 'results';
if ~exist(config.result_dir, 'dir')
    mkdir(config.result_dir);
end

% 创建图形文件夹
config.figure_dir = fullfile('results', 'figures');
if ~exist(config.figure_dir, 'dir')
    mkdir(config.figure_dir);
end

% 初始化日志
config.log_file = fullfile('results', 'log.txt');
if exist(config.log_file, 'file')
    delete(config.log_file);
end
logger.log_message('info', '开始执行二元逻辑回归分析...');

end

function finalize_system()
% 清理系统资源

% 关闭并行池
try
    delete(gcp('nocreate'));
    logger.log_message('info', '关闭并行池');
catch
    % 忽略关闭并行池时的错误
end
end