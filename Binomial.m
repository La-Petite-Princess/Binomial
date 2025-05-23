function Binomial(filename)
% 二元逻辑回归模型的完整实现 - 为MacBook Pro 2019 i9-9980HK优化
% 增强版：增加k折交叉验证、多指标监控、系数稳定性评估、参数置信区间和变量贡献分析
% 增强版2.0：增加BCa置信区间、箱线图、PR曲线、校准曲线、混淆矩阵、AIC/BIC评价、残差分析、累计方差
% 版本3.0：优化日志系统，增加日志级别控制，简化图形保存日志
% 输入:
%   filename - 包含数据的.mat文件路径
% 示例:
%   Binomial('mydata.mat');

% 开始计时
tic;

% 设置随机数种子以确保结果可重复
rng(42);

% 设置日志级别
set_log_level('debug'); % 可选值: 'debug', 'info', 'warning', 'error'

% 系统信息收集
log_message('info', '系统配置:');
log_message('info', sprintf('- CPU: Intel i9-9980HK (8核16线程)'));
log_message('info', sprintf('- 内存: 64GB RAM'));
log_message('info', sprintf('- GPU: AMD Radeon Pro 5500M 8GB'));

% 工具箱检查
if ~license('test', 'statistics_toolbox')
    error('需要安装 Statistics and Machine Learning Toolbox');
end

% 配置并行池
if isempty(gcp('nocreate'))
    % i9-9980HK有8核16线程，专业配置方案
    numLogicalProcessors = feature('numcores');
    log_message('info', sprintf('检测到%d个逻辑处理器', numLogicalProcessors));
    
    % 创建并配置本地集群对象
    c = parcluster('local');
    
    % 方案2: 平衡配置 - 适合混合计算任务(推荐)
    c.NumThreads = 2; % 每个worker使用2个线程
    poolSize = min(8, feature('numcores')); % 限制为物理核心数
    
    % 性能优化：设置工作目录到高速存储
    tmpDir = fullfile('/tmp', ['matlab_parallel_', datestr(now, 'yyyymmdd_HHMMSS')]);
    if ~exist(tmpDir, 'dir')
        mkdir(tmpDir);
    end
    c.JobStorageLocation = tmpDir;
    
    % 保存配置并创建并行池
    c.saveProfile;
    parpool(c, poolSize);
    log_message('info', sprintf('创建并行池，使用%d个worker，每个worker使用%d个线程', poolSize, c.NumThreads));
else
    log_message('info', '使用现有并行池');
end

% 创建输出文件夹
if ~exist('results', 'dir')
    mkdir('results');
end

% 创建图形文件夹
figure_dir = fullfile('results', 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

% 初始化日志
log_file = fullfile('results', 'log.txt');
if exist(log_file, 'file')
    delete(log_file);
end
log_message('info', '开始执行二元逻辑回归分析...');

try
    % 1. 加载数据
    t_start = toc;
    [data_raw, success, msg] = load_data(filename);
    if ~success
        log_message('error', msg);
        return;
    end
    t_end = toc;
    log_message('info', sprintf('成功加载数据，样本数：%d，变量数：%d，耗时：%.2f秒', ...
        size(data_raw, 1), size(data_raw, 2), t_end - t_start));
    
    % 2. 数据预处理
    t_start = toc;
    [data_processed, valid_rows] = preprocess_data(data_raw);
    t_end = toc;
    log_message('info', sprintf('数据预处理完成，有效样本数：%d，耗时：%.2f秒', ...
        length(valid_rows), t_end - t_start));
    
    % 3. 准备因变量和自变量
    t_start = toc;
    [X, y, var_names, group_means] = prepare_variables(data_processed);
    t_end = toc;
    log_message('info', sprintf('变量准备完成，自变量数：%d，耗时：%.2f秒', ...
        size(X, 2), t_end - t_start));
    
    % 4. 检查多重共线性
    t_start = toc;
    [X_final, vif_values, removed_vars] = check_collinearity(X, var_names);
    t_end = toc;
    log_message('info', sprintf('多重共线性检查完成，最终自变量数：%d，耗时：%.2f秒', ...
        size(X_final, 2), t_end - t_start));
    
    % 5. 变量分析（增强版，返回PCA结果）
    t_start = toc;
    pca_results = analyze_variable_correlations(X_final, var_names(~removed_vars));
    t_end = toc;
    log_message('info', sprintf('变量相关性分析完成，耗时：%.2f秒', t_end - t_start));
    
    % 6. 使用Bootstrap生成样本
    t_start = toc;
    [train_indices, test_indices] = bootstrap_sampling(y, 0.8, 1000);
    t_end = toc;
    log_message('info', sprintf('Bootstrap抽样完成，生成了%d个训练/测试集，耗时：%.2f秒', ...
        length(train_indices), t_end - t_start));
    
    % 7. 执行K折交叉验证（增强版可视化）
    t_start = toc;
    % 设置K的值为10
    k_value = 10;
    cv_results = k_fold_cross_validation(X_final, y, k_value, var_names(~removed_vars));
    create_kfold_performance_visualization(cv_results, figure_dir);
    t_end = toc;
    log_message('info', sprintf('K折交叉验证完成(K=%d)，耗时：%.2f秒', k_value, t_end - t_start));
    
    % 8. 使用不同方法筛选变量 - 并行处理
    t_start = toc;
    methods = {'stepwise', 'lasso', 'ridge', 'elasticnet', 'randomforest'};
    results = struct();
    
    % 创建函数句柄数组
    futures = cell(length(methods), 1);
    
    % 并行启动所有方法
    for i = 1:length(methods)
        method = methods{i};
        log_message('info', sprintf('开始并行使用%s方法筛选变量', method));
        
        % 使用parfeval异步执行
        futures{i} = parfeval(@process_method, 1, X_final, y, train_indices, test_indices, method, var_names(~removed_vars));
    end
    
    % 收集结果
    for i = 1:length(methods)
        method = methods{i};
        [methodResult] = fetchOutputs(futures{i});
        results.(method) = methodResult;
        log_message('info', sprintf('%s方法完成', method));
    end
    
    t_end = toc;
    log_message('info', sprintf('所有变量选择方法完成，耗时：%.2f秒', t_end - t_start));
    
    % 9. 监控模型系数稳定性（增强版，包含BCa区间）
    t_start = toc;
    coef_stability = monitor_coefficient_stability(results, methods, var_names(~removed_vars));
    t_end = toc;
    log_message('info', sprintf('模型系数稳定性监控完成，耗时：%.2f秒', t_end - t_start));
    
    % 10. 计算模型参数置信区间和p值（增强版，同时使用BCa和t分布）
    t_start = toc;
    param_stats = calculate_parameter_statistics(results, methods, var_names(~removed_vars));
    t_end = toc;
    log_message('info', sprintf('模型参数统计分析完成，耗时：%.2f秒', t_end - t_start));
    
    % 11. 评估每个变量对模型的贡献
    t_start = toc;
    var_contribution = evaluate_variable_contribution(X_final, y, results, methods, var_names(~removed_vars));
    t_end = toc;
    log_message('info', sprintf('变量贡献评估完成，耗时：%.2f秒', t_end - t_start));
    
    % 12. 执行残差分析
    t_start = toc;
    create_residual_analysis(results, methods, figure_dir);
    t_end = toc;
    log_message('info', sprintf('残差分析完成，耗时：%.2f秒', t_end - t_start));
    
    % 13. 删除这部分原有内容，替换为:
    t_start = toc;
    log_message('info', '开始准备可视化数据');
    t_end = toc;
    log_message('info', sprintf('可视化数据准备完成，耗时：%.2f秒', t_end - t_start));
    
    % 14. 保存结果
    t_start = toc;
    save_enhanced_results(results, var_names, group_means, cv_results, coef_stability, param_stats, var_contribution);
    t_end = toc;
    log_message('info', sprintf('结果保存完成，耗时：%.2f秒', t_end - t_start));
    
    % 统计总耗时
    total_time = toc;
    log_message('info', sprintf('分析完成！所有结果已保存到results文件夹，总耗时：%.2f秒', total_time));
    
    % 如果使用并行池，添加并行性能统计
    if ~isempty(gcp('nocreate'))
        try
            % 获取并行池对象
            p = gcp('nocreate');
            numWorkers = p.NumWorkers;
                    
            % 记录详细的并行池信息
            log_message('info', sprintf('并行池性能统计:'));
            log_message('info', sprintf('- 工作器数量: %d', numWorkers));
            log_message('info', sprintf('- 每个工作器的线程数: %d', p.NumThreads));
            log_message('info', sprintf('- 总并行线程: %d', p.NumWorkers * p.NumThreads));
                    
        catch ME
            log_message('warning', sprintf('无法获取详细的并行性能统计: %s', ME.message));
        end
    end
    
catch ME
    log_message('error', sprintf('执行过程中发生错误：%s\n%s', ME.message, getReport(ME)));
end

% 关闭并行池
try
    delete(gcp('nocreate'));
    log_message('info', '关闭并行池');
catch
    % 忽略关闭并行池时的错误
end

end

%% 辅助函数: 日志记录
function log_message(level, message)
    % 当前日志系统工作良好，可以增加：
    % 1. 颜色标记
    % 2. 更多日志信息
    % 3. 日志轮转
    
    % 获取当前日志级别
    persistent current_level log_file_size log_file_path log_file_count;
    if isempty(current_level)
        current_level = 'info'; % 默认级别
    end
    if isempty(log_file_size)
        log_file_size = 0; % 初始化日志文件大小
    end
    if isempty(log_file_path)
        log_file_path = fullfile('results', 'log.txt');
    end
    if isempty(log_file_count)
        log_file_count = 1;
    end
    
    % 定义级别优先级和颜色
    levels = {'debug', 'info', 'warning', 'error'};
    level_priority = containers.Map(levels, 1:4);
    level_colors = containers.Map(levels, {'\033[36m', '\033[32m', '\033[33m', '\033[31m'});
    
    % 确保level是有效的
    if ~ismember(lower(level), levels)
        level = 'info'; % 如果level无效，默认为'info'
    end
    
    % 获取当前时间
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % 根据级别设置前缀和颜色
    prefix = upper(level);
    
    % 构建完整日志消息（带颜色）
    if ispc
        % Windows不支持ANSI颜色代码
        log_str_console = sprintf('[%s] [%s] %s', timestamp, prefix, message);
    else
        % Unix系统支持ANSI颜色
        reset_color = '\033[0m';
        color_code = level_colors(lower(level));
        log_str_console = sprintf('[%s] %s[%s]%s %s', timestamp, color_code, prefix, reset_color, message);
    end
    
    % 文件日志不使用颜色代码
    log_str_file = sprintf('[%s] [%s] %s', timestamp, prefix, message);
    
    % 如果当前级别 >= 设置的级别，则输出到控制台
    if level_priority(lower(level)) >= level_priority(current_level)
        fprintf('%s\n', log_str_console);
    end
    
    % 写入日志文件（始终写入文件，不受级别限制）
    fid = fopen(log_file_path, 'a');
    if fid ~= -1
        fprintf(fid, '%s\n', log_str_file);
        fclose(fid);
        
        % 更新日志文件大小
        d = dir(log_file_path);
        if ~isempty(d)
            log_file_size = d.bytes;
            
            % 检查是否需要轮转日志（大于10MB）
            if log_file_size > 10 * 1024 * 1024
                % 关闭当前日志文件
                fclose('all');
                
                % 创建新的日志文件
                old_file = log_file_path;
                [path, name, ext] = fileparts(log_file_path);
                new_file = fullfile(path, sprintf('%s_%d%s', name, log_file_count, ext));
                
                % 重命名当前日志文件
                if exist(old_file, 'file')
                    movefile(old_file, new_file);
                end
                
                % 增加计数器
                log_file_count = log_file_count + 1;
                
                % 重置日志文件大小
                log_file_size = 0;
                
                % 记录日志轮转
                fid = fopen(log_file_path, 'a');
                if fid ~= -1
                    fprintf(fid, '[%s] [INFO] 日志文件已轮转，上一个文件: %s\n', timestamp, new_file);
                    fclose(fid);
                end
            end
        end
    end
end

% 在日志级别控制函数中添加图形保存专用级别
function set_log_level(level, options)
    % 设置全局日志级别
    % 输入:
    %   level - 日志级别 ('debug', 'info', 'warning', 'error')
    %   options - 可选配置参数

    persistent current_level figure_save_level;
    
    % 默认图形保存日志级别比一般日志级别高一级（减少输出）
    if nargin >= 2 && isfield(options, 'figure_save_level')
        figure_save_level = options.figure_save_level;
    elseif isempty(figure_save_level)
        % 图形保存默认使用更高级别
        switch lower(level)
            case 'debug'
                figure_save_level = 'info';
            case 'info'
                figure_save_level = 'warning';
            otherwise
                figure_save_level = level;
        end
    end
    
    % 默认级别逻辑
    valid_levels = {'debug', 'info', 'warning', 'error'};
    
    % 检查级别是否有效
    if ismember(lower(level), valid_levels)
        current_level = lower(level);
        fprintf('日志级别已设置为: %s (图形保存级别: %s)\n', upper(current_level), upper(figure_save_level));
    else
        fprintf('无效的日志级别: %s, 有效级别: debug, info, warning, error\n', level);
    end
end

% 增加获取图形保存级别的函数
function level = get_figure_save_level()
    % 获取图形保存日志级别
    % 输出:
    %   level - 当前图形保存日志级别

    persistent figure_save_level;
    if isempty(figure_save_level)
        figure_save_level = 'info'; % 默认级别
    end
    
    level = figure_save_level;
end

%获取当前日志级别
function level = get_log_level()
% 获取当前日志级别
% 输出:
%   level - 当前日志级别

    persistent current_level;
    if isempty(current_level)
        current_level = 'info'; % 默认级别
    end
    
    level = current_level;
end

%% 数据加载函数
function [data, success, message] = load_data(filename)
% 加载数据文件
% 输入:
%   filename - 数据文件名
% 输出:
%   data - 加载的数据
%   success - 是否成功加载
%   message - 成功或错误消息

success = false;
message = '';
data = [];

try
    % 使用try-catch捕获可能的错误
    try
        % 尝试直接加载变量名为'data'的数据
        s = load(filename);
        if isfield(s, 'data')
            data = s.data;
        else
            % 如果没有'data'字段，尝试获取第一个字段
            fn = fieldnames(s);
            if ~isempty(fn)
                data = s.(fn{1});
            else
                error('数据文件中没有找到有效变量');
            end
        end
    catch ME
        % 如果上面的方法失败，尝试无变量名加载
        data = load(filename);
    end
    
    % 检查数据类型并转换
    if istable(data)
        data = table2array(data);
    elseif ~isnumeric(data)
        message = '数据必须是数值矩阵或表格';
        return;
    end
    
    % 检查数据有效性
    if isempty(data)
        message = '加载的数据为空';
        return;
    end
    
    % 数据成功加载
    success = true;
    message = '数据加载成功';
catch ME
    message = sprintf('数据文件 %s 加载失败，请检查文件路径或内容！错误信息：%s', filename, ME.message);
end
end

%% GPU支持函数 - 优化版
function data_gpu = toGPU(data)
% 将数据转移到GPU(如果支持) - 为AMD 5500M GPU优化
% 输入:
%   data - 输入数据
% 输出:
%   data_gpu - GPU上的数据或原始数据
    
persistent gpuAvailable gpuMemLimit;
    
% 只检查一次GPU可用性
if isempty(gpuAvailable)
    gpuAvailable = (exist('gpuArray', 'file') == 2) && gpuDeviceCount > 0;
    
    if gpuAvailable
        gpu = gpuDevice();
        % 为AMD GPU设置更保守的内存限制（总显存的60%）
        gpuMemLimit = 0.6 * gpu.AvailableMemory;
        
        log_message('info', sprintf('GPU可用: %s, 总内存: %.2f GB, 可用内存: %.2f GB', ...
            gpu.Name, gpu.TotalMemory/1e9, gpu.AvailableMemory/1e9));
    else
        gpuMemLimit = 0;
        log_message('info', 'GPU不可用，使用CPU计算');
    end
end
    
% 智能决策：根据数据大小和传输开销决定是否使用GPU
if gpuAvailable
    try
        % 计算数据大小(字节)
        dataSize = numel(data) * 8; % 假设是双精度数据
        
        % 最小阈值：太小的数据传输开销大于收益(5MB)
        minThreshold = 5 * 1024 * 1024;
        
        % 如果数据太大或太小，不使用GPU
        if dataSize > gpuMemLimit || dataSize < minThreshold
            data_gpu = data;
            return;
        end
        
        % 使用GPU
        data_gpu = gpuArray(data);
    catch ME
        log_message('warning', sprintf('GPU转换失败: %s，使用CPU计算', ME.message));
        data_gpu = data;
    end
else
    data_gpu = data;
end
end

%% 数据预处理函数 - 优化版
function [data_processed, valid_rows] = preprocess_data(data)
% 数据清洗与预处理
% 输入:
%   data - 原始数据
% 输出:
%   data_processed - 处理后的数据
%   valid_rows - 有效行索引

% 定义有效行和需要反转的项目
rows = 1:375;
exclude_rows = [6, 10, 42, 74, 124, 127, 189, 252, 285, 298, 326, 331, 339];
valid_rows = setdiff(rows, exclude_rows);
reverse_items = [12, 19, 23];
max_score = 5;

% 预分配目标数组提高性能
data_processed = data;

% 反转指定列
data_processed(:, reverse_items) = max_score + 1 - data_processed(:, reverse_items);

% 选择有效行 - 直接索引比使用setdiff每次都计算更高效
data_processed = data_processed(valid_rows, :);

% 后处理：检查数据有效性
if any(isnan(data_processed(:)))
    % 填充NaN值
    data_processed = fillmissing(data_processed, 'linear');
    log_message('warning', '检测到NaN值并使用线性插值填充');
end

% 如果数据很大且符合GPU处理条件，则尝试GPU加速
if numel(data_processed) > 1e6
    data_processed = toGPU(data_processed);
end
end

%% 变量准备函数 - 优化版
function [X, y, var_names, group_means] = prepare_variables(data)
% 准备自变量和因变量
% 输入:
%   data - 预处理后的数据
% 输出:
%   X - 自变量矩阵
%   y - 因变量向量
%   var_names - 变量名称
%   group_means - 分组均值

% 提取因变量(第29列)
y = data(:, 29);

% 检查因变量范围
if any(y < 1 | y > 4)
    error('因变量中存在异常值，请检查！');
end

% 将因变量二元化(>2为1，<=2为0)
y = double(y > 2);

% 定义分组
groups = {
    [1, 2, 3, 4, 5, 6, 12, 19, 23],  % 组1
    [7, 8, 9],                       % 组2
    [10, 11],                        % 组3
    [13, 14],                        % 组4
    [15, 17, 18, 20, 21],            % 组5
    [22, 24],                        % 组6
    [25, 26],                        % 组7
    [27, 28]                         % 组8
};

% 标准化并计算各组均值
X = zeros(size(data, 1), length(groups));
group_means = cell(length(groups), 1);
var_names = cell(length(groups), 1);

% 分块计算参数 - 增大分块大小利用64GB内存
n_samples = size(data, 1);
BLOCK_SIZE = 10000;  % 更大的块大小
n_blocks = ceil(n_samples / BLOCK_SIZE);
use_blocks = n_samples > BLOCK_SIZE * 3; % 只有超大数据才分块

% 使用parfor并行处理各组
parfor i = 1:length(groups)
    % 获取当前组的列
    group_cols = groups{i};
    
    % 提取当前组的数据
    group_data = data(:, group_cols);
    
    % 标准化处理
    if use_blocks && n_samples > 100000
        % 先计算整体统计量
        mu = mean(group_data);
        sigma = std(group_data);
        
        % 初始化标准化后的数据
        group_data_std = zeros(size(group_data));
        
        % 分块标准化 - 仅对超大数据集使用
        for b = 1:n_blocks
            start_idx = (b-1)*BLOCK_SIZE + 1;
            end_idx = min(b*BLOCK_SIZE, n_samples);
            block_data = group_data(start_idx:end_idx, :);
            
            % 标准化
            group_data_std(start_idx:end_idx, :) = (block_data - mu) ./ sigma;
        end
    else
        % 对于一般大小数据集，直接标准化
        group_data_std = zscore(group_data);
    end
    
    % 计算标准化后的均值
    group_mean = mean(group_data_std, 2);
    
    % 存储原始均值（未标准化）
    group_orig_mean = mean(group_data, 2);
    
    % 返回结果
    X(:, i) = group_mean;
    group_means{i} = group_orig_mean;
    var_names{i} = sprintf('Group%d', i);
end

% 将X从GPU移回CPU(如果需要)
if isa(X, 'gpuArray')
    X = gather(X);
end
if isa(y, 'gpuArray')
    y = gather(y);
end
end

%% 多重共线性检查函数 - 优化版
function [X_cleaned, vif_values, removed_vars] = check_collinearity(X, var_names)
% 检查并处理多重共线性
% 输入:
%   X - 自变量矩阵
%   var_names - 变量名称
% 输出:
%   X_cleaned - 处理后的自变量矩阵
%   vif_values - VIF值
%   removed_vars - 被移除的变量标记

% 计算相关矩阵 - 使用更高效的计算
R = corr(X, 'Type', 'Pearson');

% 检查相关矩阵
if any(isnan(R(:))) || any(isinf(R(:)))
    error('相关矩阵 R 包含 NaN 或 Inf，请检查输入数据！');
end

% 初始化输出变量
removed_vars = false(size(X, 2), 1);
vif_values = zeros(size(X, 2), 1);

% 优化：先检查矩阵的条件数而不是秩
cond_num = cond(R);
if cond_num > 30
    log_message('warning', sprintf('相关矩阵条件数较高(%.2f)，可能存在多重共线性', cond_num));
    
    % 使用SVD代替直接求逆计算VIF，更数值稳定
    [U, S, V] = svd(R);
    s = diag(S);
    
    % 如果最小奇异值小于阈值，认为矩阵接近奇异
    if min(s) < 1e-10
        warning('MatrixError:NearSingular', '相关矩阵接近奇异，使用PCA处理多重共线性');
        
        % 使用主成分分析处理多重共线性
        [~, score, ~, ~, explained] = pca(X, 'Algorithm', 'svd');
        cum_var = cumsum(explained);
        k = find(cum_var >= 95, 1, 'first'); % 保留解释95%方差的成分，提高保留信息
        X_cleaned = score(:, 1:k);
        
        log_message('warning', sprintf('使用PCA降维，从%d个变量降至%d个主成分', size(X, 2), k));
        
        % 所有原始变量都被"移除"
        removed_vars = true(size(X, 2), 1);
        vif_values = ones(size(X, 2), 1) * Inf;
        return;
    else
        % 使用SVD计算VIF
        vif_values = zeros(size(X, 2), 1);
        for i = 1:size(X, 2)
            % 选择第i列作为因变量
            y_i = X(:, i);
            % 选择其他列作为自变量
            X_i = X(:, setdiff(1:size(X, 2), i));
            % 计算R²
            b = X_i \ y_i;
            y_hat = X_i * b;
            SS_total = sum((y_i - mean(y_i)).^2);
            SS_residual = sum((y_i - y_hat).^2);
            R_squared = 1 - SS_residual/SS_total;
            % 计算VIF
            vif_values(i) = 1 / (1 - R_squared);
        end
    end
else
    % 条件良好，直接计算VIF
    try
        % 使用更高效的计算方法
        vif_values = zeros(size(X, 2), 1);
        for i = 1:size(X, 2)
            % 使用线性回归而不是直接求逆计算VIF，数值更稳定
            idx = setdiff(1:size(X, 2), i);
            mdl = fitlm(X(:, idx), X(:, i));
            vif_values(i) = 1 / (1 - mdl.Rsquared.Ordinary);
        end
    catch ME
        log_message('warning', sprintf('VIF计算失败，使用SVD方法: %s', ME.message));
        % 使用SVD方法作为备选
        vif_values = ones(size(X, 2), 1) ./ diag(pinv(R));
    end
end

% 输出VIF值
log_message('info', '自变量的VIF值：');
for i = 1:length(vif_values)
    log_message('info', sprintf('%s: %.2f', var_names{i}, vif_values(i)));
end

% 找出高VIF值的变量 - 使用阈值为10
high_vif = find(vif_values > 10);
if ~isempty(high_vif)
    log_message('warning', '移除高VIF变量索引：');
    for i = 1:length(high_vif)
        log_message('warning', sprintf('%s (VIF = %.2f)', var_names{high_vif(i)}, vif_values(high_vif(i))));
    end
    
    % 移除高VIF值的变量
    removed_vars(high_vif) = true;
    X_cleaned = X(:, ~removed_vars);
    
    % 递归检查剩余变量的VIF
    if sum(~removed_vars) > 1
        log_message('info', '递归检查剩余变量的VIF值');
        [X_cleaned_rec, vif_values_rec, removed_vars_rec] = check_collinearity(X_cleaned, var_names(~removed_vars));
        
        % 更新removed_vars以反映递归结果
        still_removed = false(size(X, 2), 1);
        still_removed(~removed_vars) = removed_vars_rec;
        removed_vars = removed_vars | still_removed;
        
        X_cleaned = X_cleaned_rec;
    end
else
    X_cleaned = X;
end
end

%% 变量相关性分析函数 - 优化版
function [pca_results] = analyze_variable_correlations(X, var_names)
% 分析变量之间的相关性
% 输入:
%   X - 自变量矩阵
%   var_names - 变量名称
% 输出:
%   pca_results - PCA分析结果

% 计算变量之间的相关性
R = corr(X);

% 创建更高分辨率的热图
fig = figure('Name', '变量相关性矩阵', 'Position', [100, 100, 1000, 900]);

% 使用更美观的热图
h = heatmap(R, 'XDisplayLabels', var_names, 'YDisplayLabels', var_names);
h.Title = '变量相关性矩阵';
h.FontSize = 10;
h.Colormap = parula;

% 调整colorbar
caxis([-1, 1]);
colorbar;

% 保存矢量图
save_figure(fig, 'results', 'variable_correlation', 'Formats', {'svg'});
close(fig);

% 识别高度相关的变量对
[rows, cols] = find(triu(abs(R) > 0.8, 1));
if ~isempty(rows)
    log_message('warning', '发现高度相关的变量对 (|r| > 0.8):');
    for i = 1:length(rows)
        log_message('warning', sprintf('%s 与 %s: r = %.2f', var_names{rows(i)}, var_names{cols(i)}, R(rows(i), cols(i))));
    end
else
    log_message('info', '未发现高度相关的变量对 (|r| > 0.8)');
end

% 增加主成分分析可视化 - 仅在变量较多时使用
pca_results = struct(); % 初始化PCA结果结构

if length(var_names) > 3
    try
        % 执行PCA
        [coeff, score, ~, ~, explained, mu] = pca(X);
        
        % 存储PCA结果
        pca_results.coeff = coeff;
        pca_results.score = score;
        pca_results.explained = explained;
        pca_results.mu = mu;
        pca_results.cum_explained = cumsum(explained);
        
        % 创建PCA双线图
        fig2 = figure('Name', '主成分分析', 'Position', [100, 100, 1200, 900]);
        
        % 绘制变量在前两个主成分上的投影
        subplot(2, 2, 1);
        biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', var_names);
        title('变量在主成分1-2上的投影');
        grid on;
        
        % 绘制解释方差比例
        subplot(2, 2, 2);
        bar(explained);
        xlabel('主成分');
        ylabel('解释方差百分比');
        title('各主成分解释方差比例');
        grid on;
        
        % 绘制累积解释方差
        subplot(2, 2, 3);
        plot(cumsum(explained), 'o-', 'LineWidth', 2);
        xlabel('主成分数量');
        ylabel('累积解释方差百分比');
        title('累积解释方差');
        grid on;
        
        % 增加百分比标注
        cum_explained = cumsum(explained);
        hold on;
        
        % 选取几个关键点标注
        key_components = [1, min(3, length(cum_explained)), min(5, length(cum_explained))];
        for i = 1:length(key_components)
            idx = key_components(i);
            plot(idx, cum_explained(idx), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
            text(idx, cum_explained(idx) + 2, sprintf('%.1f%%', cum_explained(idx)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
        end
        
        % 绘制95%方差的线
        idx_95 = find(cum_explained >= 95, 1, 'first');
        if ~isempty(idx_95)
            plot([0, idx_95, idx_95], [95, 95, 0], 'k--');
            text(idx_95-0.5, 96, sprintf('95%%方差需要%d个主成分', idx_95), ...
                'HorizontalAlignment', 'right', 'FontSize', 9);
        end
        
        % 绘制变量对主成分的贡献
        subplot(2, 2, 4);
        imagesc(abs(coeff(:, 1:min(5, size(coeff, 2)))));
        colorbar;
        xlabel('主成分');
        set(gca, 'YTick', 1:length(var_names), 'YTickLabel', var_names);
        title('变量对前5个主成分的贡献(绝对值)');
        
        % 保存PCA图
        save_figure(fig2, 'results', 'pca_analysis', 'Formats', {'svg'});
        close(fig2);
        
        % 创建主成分累计方差表
        cum_var_table = table((1:length(explained))', explained, cum_explained, ...
            'VariableNames', {'Component', 'ExplainedVariance', 'CumulativeVariance'});
        
        % 输出累计方差
        log_message('info', '主成分累计方差:');
        key_idx = [1, 2, 3, min(5, length(cum_explained)), min(10, length(cum_explained))];
        key_idx = unique(key_idx);
        for i = 1:length(key_idx)
            idx = key_idx(i);
            log_message('info', sprintf('  前%d个主成分解释了%.2f%%的总方差', idx, cum_explained(idx)));
        end
        
        % 创建新的累计方差图
        fig3 = figure('Name', '主成分累计方差', 'Position', [100, 100, 900, 600]);
        
        % 绘制阶梯图
        stairs(cum_explained, 'LineWidth', 2);
        hold on;
        
        % 标记95%方差
        if ~isempty(idx_95)
            plot([0, idx_95], [95, 95], 'r--');
            plot([idx_95, idx_95], [0, 95], 'r--');
            text(idx_95 + 0.1, 60, sprintf('95%%方差需要%d个主成分', idx_95), ...
                'FontSize', 10, 'Color', 'r');
        end
        
        % 标记80%方差
        idx_80 = find(cum_explained >= 80, 1, 'first');
        if ~isempty(idx_80)
            plot([0, idx_80], [80, 80], 'g--');
            plot([idx_80, idx_80], [0, 80], 'g--');
            text(idx_80 + 0.1, 40, sprintf('80%%方差需要%d个主成分', idx_80), ...
                'FontSize', 10, 'Color', 'g');
        end
        
        % 设置图形属性
        xlabel('主成分数量', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('累计解释方差百分比', 'FontSize', 12, 'FontWeight', 'bold');
        title('主成分分析累计解释方差', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        xlim([0, min(15, length(explained))]);
        ylim([0, 100]);
        
        % 保存图形
        save_figure(fig3, 'results', 'cumulative_variance', 'Formats', {'svg'});
        close(fig3);
        
        % 创建主成分载荷可视化
        fig4 = figure('Name', '主成分载荷', 'Position', [100, 100, 1200, 600]);
        
        % 显示前3个主成分的载荷
        num_pc = min(3, size(coeff, 2));
        for i = 1:num_pc
            subplot(1, num_pc, i);
            bar(coeff(:, i));
            xlabel('变量');
            ylabel('载荷系数');
            title(sprintf('主成分%d载荷', i));
            set(gca, 'XTick', 1:length(var_names), 'XTickLabel', var_names, 'XTickLabelRotation', 45);
            grid on;
        end
        
        % 保存图形
        save_figure(fig4, 'results', 'principal_component_loadings', 'Formats', {'svg'});
        close(fig4);
    catch ME
        log_message('warning', sprintf('PCA可视化失败: %s', ME.message));
    end
end
end

%% Bootstrap抽样函数 - 优化版
function [train_indices, test_indices] = bootstrap_sampling(y, train_ratio, n_samples)
% 使用Bootstrap进行分层抽样
% 输入:
%   y - 因变量
%   train_ratio - 训练集比例
%   n_samples - 样本数量
% 输出:
%   train_indices - 训练集索引
%   test_indices - 测试集索引

% 找出各类别的索引
class_0_idx = find(y == 0);
class_1_idx = find(y == 1);

% 预计算常量避免在parfor中重复计算
n0 = length(class_0_idx);
n1 = length(class_1_idx);
n0_train = round(train_ratio * n0);
n1_train = round(train_ratio * n1);
total_samples = length(y);

% 预分配结果数组
train_indices = cell(n_samples, 1);
test_indices = cell(n_samples, 1);

% 预分配随机种子数组确保并行迭代随机性
rng_seeds = randi(1000000, n_samples, 1);

% 创建逻辑索引数组提高效率
total_mask = false(total_samples, n_samples);

% 使用parfor并行处理
parfor i = 1:n_samples
    % 设置当前迭代的随机种子
    rng(rng_seeds(i));
    
    % 对每个类别进行分层抽样
    train_idx_0 = class_0_idx(randsample(n0, n0_train));
    train_idx_1 = class_1_idx(randsample(n1, n1_train));
    
    % 合并训练集索引
    train_idx = [train_idx_0; train_idx_1];
    
    % 使用逻辑索引代替setdiff提高性能
    mask = false(total_samples, 1);
    mask(train_idx) = true;
    
    % 存储训练集和测试集
    train_indices{i} = train_idx;
    test_indices{i} = find(~mask);
    
    % 存储total_mask
    total_mask(:, i) = mask;
end

% 输出Bootstrap样本的统计信息
train_sizes = cellfun(@length, train_indices);
test_sizes = cellfun(@length, test_indices);
log_message('info', sprintf('Bootstrap样本统计: 平均训练集大小=%.1f, 平均测试集大小=%.1f', ...
    mean(train_sizes), mean(test_sizes)));

% 计算样本覆盖率
coverage = mean(sum(total_mask, 2) > 0);
log_message('info', sprintf('数据覆盖率: %.2f%%', coverage * 100));
end

%% K折表现的可视化函数
function create_kfold_performance_visualization(cv_results, figure_dir)
% 创建K折交叉验证各折性能可视化
% 输入:
%   cv_results - 交叉验证结果
%   figure_dir - 图形保存目录

% 获取折数
k = length(cv_results.accuracy);

% 创建图形
fig = figure('Name', 'K-Fold Performance by Fold', 'Position', [100, 100, 1200, 800]);

% 准备数据
metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
n_metrics = length(metrics);

% 创建子图布局
rows = 2;
cols = 3;

% 绘制每个指标的折线图
for i = 1:n_metrics
    metric = metrics{i};
    metric_label = metric_labels{i};
    
    % 创建子图
    subplot(rows, cols, i);
    
    % 获取数据
    values = cv_results.(metric);
    mean_val = cv_results.(['avg_' metric]);
    std_val = cv_results.(['std_' metric]);
    
    % 绘制折线图
    plot(1:k, values, 'o-', 'LineWidth', 1.5, 'Color', [0.3, 0.6, 0.8], 'MarkerFaceColor', [0.3, 0.6, 0.8]);
    hold on;
    
    % 绘制均值线
    plot([0.5, k+0.5], [mean_val, mean_val], 'r--', 'LineWidth', 1.5);
    
    % 绘制标准差区间
    fill([1:k, fliplr(1:k)], [mean_val + std_val * ones(1, k), fliplr(mean_val - std_val * ones(1, k))], ...
        [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    
    % 设置图形属性
    xlabel('折数', 'FontSize', 10);
    ylabel(metric_label, 'FontSize', 10);
    title(sprintf('%s (均值=%.3f, 标准差=%.3f)', metric_label, mean_val, std_val), 'FontSize', 12);
    grid on;
    xlim([0.5, k+0.5]);
    
    % 调整Y轴范围
    if strcmp(metric, 'auc') || strcmp(metric, 'f1_score')
        ylim([0.5, 1]);
    else
        ylim([0, 1]);
    end
    
    % 添加数据点标签
    for j = 1:k
        text(j, values(j) + 0.02, sprintf('%.3f', values(j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7);
    end
    
    % 添加图例
    legend({'各折值', '均值', '标准差区间'}, 'Location', 'best', 'FontSize', 8);
end

% 添加总标题
sgtitle(sprintf('K折交叉验证各折性能指标 (K=%d)', k), 'FontSize', 16, 'FontWeight', 'bold');
set(gcf, 'Color', 'white');

% 保存图形
save_figure(fig, figure_dir, 'kfold_performance_by_fold', 'Formats', {'svg'});
close(fig);
end

%% K折交叉验证函数 - 新增
function results = k_fold_cross_validation(X, y, k, var_names)
% 执行K折交叉验证来评估模型稳定性
% 输入:
%   X - 自变量矩阵
%   y - 因变量
%   k - 折数
%   var_names - 变量名称（新增）
% 输出:
%   results - 交叉验证结果

% 确认K值有效
if k < 2
    error('K值必须大于等于2');
end

% 获取总样本数
n = length(y);

% 如果K大于样本数，调整K
if k > n
    log_message('warning', sprintf('K值(%d)大于样本数(%d)，调整为%d', k, n, n));
    k = n;
end

% 创建交叉验证分组
cv = cvpartition(y, 'KFold', k);

% 初始化性能指标结果
results = struct();
results.accuracy = zeros(k, 1);
results.precision = zeros(k, 1);
results.recall = zeros(k, 1);
results.specificity = zeros(k, 1);
results.f1_score = zeros(k, 1);
results.auc = zeros(k, 1);
results.aic = zeros(k, 1);     % 新增
results.bic = zeros(k, 1);     % 新增
results.coefs = cell(k, 1);
results.fold_indices = cell(k, 1);
results.y_pred = cell(k, 1);       % 新增
results.y_test = cell(k, 1);       % 新增
results.y_pred_prob = cell(k, 1);  % 新增

% 记录每个模型的变量系数分布
n_vars = size(X, 2);
all_coefs = zeros(k, n_vars+1); % +1是因为有截距项

% 对每个折执行训练和评估
for i = 1:k
    % 获取当前折的训练集和测试集
    train_idx = cv.training(i);
    test_idx = cv.test(i);
    
    % 存储当前折的索引
    results.fold_indices{i} = struct('train', find(train_idx), 'test', find(test_idx));
    
    % 使用训练集训练逻辑回归模型
    try
        % 使用更强大的fitglm函数
        mdl = fitglm(X(train_idx, :), y(train_idx), 'Distribution', 'binomial', 'Link', 'logit');
        
        % 存储模型系数
        coefs = mdl.Coefficients.Estimate;
        results.coefs{i} = coefs;
        all_coefs(i, :) = coefs';
        
        % 使用测试集预测
        y_pred_prob = predict(mdl, X(test_idx, :));
        y_pred = y_pred_prob > 0.5;
        
        % 保存预测结果
        results.y_pred{i} = y_pred;
        results.y_test{i} = y(test_idx);
        results.y_pred_prob{i} = y_pred_prob;
        
        % 计算评估指标
        y_test = y(test_idx);
        
        % 准确率
        results.accuracy(i) = sum(y_pred == y_test) / length(y_test);
        
        % 计算混淆矩阵
        TP = sum(y_pred == 1 & y_test == 1);
        TN = sum(y_pred == 0 & y_test == 0);
        FP = sum(y_pred == 1 & y_test == 0);
        FN = sum(y_pred == 0 & y_test == 1);
        
        % 精确率
        if (TP + FP) > 0
            results.precision(i) = TP / (TP + FP);
        else
            results.precision(i) = 0;
        end
        
        % 召回率/敏感性
        if (TP + FN) > 0
            results.recall(i) = TP / (TP + FN);
        else
            results.recall(i) = 0;
        end
        
        % 特异性
        if (TN + FP) > 0
            results.specificity(i) = TN / (TN + FP);
        else
            results.specificity(i) = 0;
        end
        
        % F1分数
        if (results.precision(i) + results.recall(i)) > 0
            results.f1_score(i) = 2 * (results.precision(i) * results.recall(i)) / (results.precision(i) + results.recall(i));
        else
            results.f1_score(i) = 0;
        end
        
        % AUC
        if length(unique(y_test)) > 1 % 确保正负样本都有
            [~, ~, ~, auc] = perfcurve(y_test, y_pred_prob, 1);
            results.auc(i) = auc;
        else
            results.auc(i) = NaN;
        end
        
        % 计算AIC和BIC - 新增
        deviance = mdl.Deviance;
        n_params = length(coefs);
        n_samples = sum(train_idx);
        
        results.aic(i) = deviance + 2 * n_params;
        results.bic(i) = deviance + log(n_samples) * n_params;
        
    catch ME
        log_message('warning', sprintf('第%d折交叉验证失败: %s', i, ME.message));
        results.accuracy(i) = NaN;
        results.precision(i) = NaN;
        results.recall(i) = NaN;
        results.specificity(i) = NaN;
        results.f1_score(i) = NaN;
        results.auc(i) = NaN;
        results.aic(i) = NaN;
        results.bic(i) = NaN;
        results.coefs{i} = NaN(n_vars+1, 1);
        all_coefs(i, :) = NaN(1, n_vars+1);
    end
end

% 计算每个指标的平均值和标准差
% 计算每个指标的平均值和标准差
fields = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', 'aic', 'bic'};
for j = 1:length(fields)
    field = fields{j};
    results.(['avg_' field]) = mean(results.(field), 'omitnan');
    results.(['std_' field]) = std(results.(field), 'omitnan');
    results.(['cv_' field]) = results.(['std_' field]) / results.(['avg_' field]); % 新增：变异系数
end

% 系数稳定性分析
results.coef_mean = mean(all_coefs, 'omitnan');
results.coef_std = std(all_coefs, 'omitnan');
results.coef_cv = abs(results.coef_std ./ results.coef_mean); % 变异系数
results.all_coefs = all_coefs;

% 添加变量名字段
results.variables = cell(n_vars + 1, 1); % +1是因为有截距项
results.variables{1} = 'Intercept';
for i = 1:n_vars
    if nargin > 3 && i <= length(var_names) % 检查是否传入了变量名
        results.variables{i+1} = var_names{i};
    else
        results.variables{i+1} = sprintf('Var%d', i);
    end
end

% 记录K折验证的整体情况
log_message('info', sprintf('K折交叉验证指标: 准确率=%.3f(±%.3f), 精确率=%.3f(±%.3f), 召回率=%.3f(±%.3f), F1=%.3f(±%.3f), AUC=%.3f(±%.3f), AIC=%.1f(±%.1f), BIC=%.1f(±%.1f)', ...
    results.avg_accuracy, results.std_accuracy, ...
    results.avg_precision, results.std_precision, ...
    results.avg_recall, results.std_recall, ...
    results.avg_f1_score, results.std_f1_score, ...
    results.avg_auc, results.std_auc, ...
    results.avg_aic, results.std_aic, ...
    results.avg_bic, results.std_bic));

% 创建K折交叉验证可视化
create_kfold_visualization(results, k);

end

% 辅助函数：创建K折交叉验证可视化
function create_kfold_visualization(results, k)
% 创建K折交叉验证可视化
% 输入:
%   results - 交叉验证结果
%   k - 折数

% 创建图形
fig = figure('Name', 'K-Fold Cross-Validation Results', 'Position', [100, 100, 1200, 900]);

% 准备数据
metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
colors = lines(length(metrics));

% 创建子图1：各折对比
subplot(2, 2, 1);
hold on;

for i = 1:length(metrics)
    metric = metrics{i};
    values = results.(metric);
    
    % 绘制折线
    plot(1:k, values, 'o-', 'LineWidth', 1.5, 'Color', colors(i,:), 'DisplayName', metric_labels{i});
    
    % 绘制均值线
    mean_val = results.(['avg_' metric]);
    plot([0.5, k+0.5], [mean_val, mean_val], '--', 'Color', colors(i,:), 'LineWidth', 1, 'HandleVisibility', 'off');
end

% 设置图形属性
xlabel('折数', 'FontSize', 12);
ylabel('性能值', 'FontSize', 12);
title('K折交叉验证各折性能', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Location', 'best', 'FontSize', 9);
xlim([0.5, k+0.5]);
ylim([0, 1.05]);
set(gca, 'XTick', 1:k);

% 创建子图2：性能指标比较
subplot(2, 2, 2);

% 准备数据
mean_values = zeros(length(metrics), 1);
std_values = zeros(length(metrics), 1);

for i = 1:length(metrics)
    metric = metrics{i};
    mean_values(i) = results.(['avg_' metric]);
    std_values(i) = results.(['std_' metric]);
end

% 创建条形图
bar_h = bar(mean_values);
set(bar_h, 'FaceColor', 'flat');
for i = 1:length(metrics)
    bar_h.CData(i,:) = colors(i,:);
end

% 添加误差线
hold on;
errorbar(1:length(metrics), mean_values, std_values, '.k');

% 设置图形属性
set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metric_labels, 'XTickLabelRotation', 45);
ylabel('平均值', 'FontSize', 12);
title('各性能指标均值和标准差', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0, 1.05]);

% 添加数值标签
for i = 1:length(metrics)
    text(i, mean_values(i) + std_values(i) + 0.03, sprintf('%.3f±%.3f', mean_values(i), std_values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% 创建子图3：参数统计
subplot(2, 2, 3);

% 准备AIC和BIC数据
aic_values = results.aic;
bic_values = results.bic;

% 创建箱线图
boxplot([aic_values, bic_values], 'Labels', {'AIC', 'BIC'}, 'Notch', 'on');
ylabel('值', 'FontSize', 12);
title('AIC和BIC分布', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加均值和标准差
text(1, max(aic_values) + 0.1 * range(aic_values), sprintf('均值: %.1f\n标准差: %.1f', results.avg_aic, results.std_aic), ...
    'HorizontalAlignment', 'center', 'FontSize', 9);
text(2, max(bic_values) + 0.1 * range(bic_values), sprintf('均值: %.1f\n标准差: %.1f', results.avg_bic, results.std_bic), ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

% 创建子图4：系数稳定性
subplot(2, 2, 4);

% 排除截距，只分析变量系数
coef_cv = results.coef_cv(2:end);
var_names = results.variables(2:end);

% 按变异系数排序
[sorted_cv, idx] = sort(coef_cv, 'descend');
sorted_vars = var_names(idx);

% 限制显示的变量数量
max_vars = min(10, length(sorted_vars));
sorted_cv = sorted_cv(1:max_vars);
sorted_vars = sorted_vars(1:max_vars);

% 创建条形图
bar_h = barh(sorted_cv);
set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);

% 设置图形属性
set(gca, 'YTick', 1:max_vars, 'YTickLabel', sorted_vars);
xlabel('变异系数 (CV)', 'FontSize', 12);
title('系数稳定性分析 (CV)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加阈值线
hold on;
plot([0.5, 0.5], [0, max_vars+1], 'r--', 'LineWidth', 1.5);
text(0.52, 1, '不稳定阈值 (CV>0.5)', 'Color', 'r', 'FontSize', 9, 'VerticalAlignment', 'bottom');

% 保存图形
sgtitle(sprintf('K折交叉验证分析 (K=%d)', k), 'FontSize', 16, 'FontWeight', 'bold');
set(gcf, 'Color', 'white');

% 保存为图像
    save_figure(fig, 'results', 'kfold_visualization', 'Formats', {'svg'});
close(fig);
end

%% 变量选择函数 - 优化版
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
    log_message('warning', sprintf('变量频率长度(%d)与变量数量(%d)不匹配，进行调整', length(var_freq), n_vars));
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
log_message('info', sprintf('使用出现频率最高的变量组合作为总体变量 (出现%d次，占比%.2f%%)', ...
    combo_counts(max_idx), 100*combo_counts(max_idx)/n_samples));
log_message('info', sprintf('选择的变量索引: %s', mat2str(combo_indices)));

% 删除重复的变量组合并统计每种组合的出现次数
[unique_combinations, ~, ic] = unique(cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false));
combination_counts = accumarray(ic, 1);

% 按出现次数排序
[sorted_counts, idx] = sort(combination_counts, 'descend');
sorted_combinations = unique_combinations(idx);

% 打印前5个最常见的变量组合
log_message('info', sprintf('前%d个最常见的变量组合:', min(5, length(sorted_combinations))));
for i = 1:min(5, length(sorted_combinations))
    log_message('info', sprintf('组合 #%d (出现%d次): %s', i, sorted_counts(i), sorted_combinations{i}));
end
end

%% 模型训练和评估函数 - 修改版，增加更多评估指标
function [models, overall_performance, group_performance] = train_and_evaluate_models_with_groups(X, y, train_indices, test_indices, var_combinations, method, var_names)
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
    log_message('info', sprintf('前%d个最常见的变量组合的性能:', top_n));
    for i = 1:top_n
        combo = group_performance(i);
        var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
        log_message('info', sprintf('组合 #%d (出现%d次, AUC=%.3f, F1=%.3f, AIC=%.1f, BIC=%.1f): %s', ...
            i, combo.count, combo.auc, combo.f1_score, combo.aic, combo.bic, var_str));
    end
end

%% 残差分析函数
function create_residual_analysis(results, methods, figure_dir)
% 创建残差分析和可视化
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   figure_dir - 图形保存目录

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
                
                % 计算残差（皮尔逊残差）
                residuals = (all_labels - all_probs) ./ sqrt(all_probs .* (1 - all_probs));
                
                % 计算Deviance残差
                deviance_residuals = sign(all_labels - all_probs) .* ...
                    sqrt(-2 * (all_labels .* log(all_probs) + (1 - all_labels) .* log(1 - all_probs)));
                
                % 创建残差分析图
                fig = figure('Name', sprintf('%s Residual Analysis', method), 'Position', [100, 100, 1200, 900]);
                
                % 第一行：皮尔逊残差相关图表
                
                % 创建子图1：皮尔逊残差 vs 预测概率
                subplot(2, 3, 1);
                scatter(all_probs, residuals, 30, 'filled', 'MarkerFaceAlpha', 0.6);
                hold on;
                plot([0, 1], [0, 0], 'k--');
                xlabel('预测概率', 'FontSize', 10);
                ylabel('皮尔逊残差', 'FontSize', 10);
                title('皮尔逊残差 vs 预测概率', 'FontSize', 12);
                grid on;
                
                % 添加平滑曲线
                try
                    [xData, yData] = prepareCurveData(all_probs, residuals);
                    smoothed = smooth(xData, yData, 0.2, 'loess');
                    plot(xData, smoothed, 'r-', 'LineWidth', 2);
                catch
                    % 如果平滑失败，忽略
                end
                
                % 创建子图2：皮尔逊残差箱线图
                subplot(2, 3, 2);
                boxplot(residuals, all_labels, 'Labels', {'0', '1'});
                ylabel('皮尔逊残差', 'FontSize', 10);
                title('按实际类别分组的皮尔逊残差', 'FontSize', 12);
                grid on;
                
                % 创建子图3：皮尔逊残差QQ图
                subplot(2, 3, 3);
                qqplot(residuals);
                title('皮尔逊残差QQ图', 'FontSize', 12);
                grid on;
                
                % 第二行：Deviance残差相关图表
                
                % 创建子图4：Deviance残差 vs 预测概率
                subplot(2, 3, 4);
                scatter(all_probs, deviance_residuals, 30, 'filled', 'MarkerFaceAlpha', 0.6);
                hold on;
                plot([0, 1], [0, 0], 'k--');
                xlabel('预测概率', 'FontSize', 10);
                ylabel('Deviance残差', 'FontSize', 10);
                title('Deviance残差 vs 预测概率', 'FontSize', 12);
                grid on;
                
                % 添加平滑曲线
                try
                    [xData, yData] = prepareCurveData(all_probs, deviance_residuals);
                    smoothed = smooth(xData, yData, 0.2, 'loess');
                    plot(xData, smoothed, 'r-', 'LineWidth', 2);
                catch
                    % 如果平滑失败，忽略
                end
                
                % 创建子图5：Deviance残差箱线图
                subplot(2, 3, 5);
                boxplot(deviance_residuals, all_labels, 'Labels', {'0', '1'});
                ylabel('Deviance残差', 'FontSize', 10);
                title('按实际类别分组的Deviance残差', 'FontSize', 12);
                grid on;
                
                % 创建子图6：Deviance残差QQ图
                subplot(2, 3, 6);
                qqplot(deviance_residuals);
                title('Deviance残差QQ图', 'FontSize', 12);
                grid on;
                
                % 添加总标题
                sgtitle(sprintf('%s方法的残差分析', method), 'FontSize', 14, 'FontWeight', 'bold');
                
                % 保存图形
                save_figure(fig, figure_dir, sprintf('%s_residual_analysis', method), 'Formats', {'svg'});
                close(fig);
                
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
                stats.deviance_min = min(deviance_residuals);
                stats.deviance_max = max(deviance_residuals);
                stats.deviance_skewness = skewness(deviance_residuals);
                stats.deviance_kurtosis = kurtosis(deviance_residuals);
                
                % 输出残差统计信息
                log_message('info', sprintf('%s方法残差统计:', method));
                log_message('info', sprintf('皮尔逊残差: 均值=%.3f, 标准差=%.3f, 偏度=%.3f, 峰度=%.3f', ...
                    stats.pearson_mean, stats.pearson_std, stats.pearson_skewness, stats.pearson_kurtosis));
                log_message('info', sprintf('Deviance残差: 均值=%.3f, 标准差=%.3f, 偏度=%.3f, 峰度=%.3f', ...
                    stats.deviance_mean, stats.deviance_std, stats.deviance_skewness, stats.deviance_kurtosis));
                
                % 检测潜在的异常点
                pearson_outliers = abs(residuals) > 2.5;
                deviance_outliers = abs(deviance_residuals) > 2.5;
                
                if any(pearson_outliers)
                    log_message('info', sprintf('%s方法检测到%d个皮尔逊残差异常点(|残差|>2.5)', ...
                        method, sum(pearson_outliers)));
                end
                
                if any(deviance_outliers)
                    log_message('info', sprintf('%s方法检测到%d个Deviance残差异常点(|残差|>2.5)', ...
                        method, sum(deviance_outliers)));
                end
            else
                log_message('warning', sprintf('%s方法没有足够的预测数据进行残差分析', method));
            end
        else
            log_message('warning', sprintf('%s方法缺少预测概率或真实标签数据，无法进行残差分析', method));
        end
    else
        log_message('info', sprintf('%s方法不适用于传统残差分析', method));
    end
end

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
        % 创建残差比较图 - 皮尔森残差
        fig1 = figure('Name', 'Pearson Residuals Comparison', 'Position', [100, 100, 1000, 600]);
        
        % 创建箱线图比较
        boxplot(cell2mat(all_pearson_residuals), repelem(1:length(methods_with_residuals), cellfun(@length, all_pearson_residuals)), ...
            'Labels', methods_with_residuals, 'Notch', 'on');
        
        % 设置图形属性
        ylabel('皮尔森残差', 'FontSize', 12, 'FontWeight', 'bold');
        title('各方法皮尔森残差分布比较', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加零线
        hold on;
        plot(xlim, [0 0], 'k--');
        
        % 保存图形
        save_figure(fig1, figure_dir, 'pearson_residuals_comparison', 'Formats', {'svg'});
        close(fig1);
        
        % 创建残差比较图 - Deviance残差 (新增)
        fig2 = figure('Name', 'Deviance Residuals Comparison', 'Position', [100, 100, 1000, 600]);
        
        % 创建箱线图比较
        boxplot(cell2mat(all_deviance_residuals), repelem(1:length(methods_with_residuals), cellfun(@length, all_deviance_residuals)), ...
            'Labels', methods_with_residuals, 'Notch', 'on');
        
        % 设置图形属性
        ylabel('Deviance残差', 'FontSize', 12, 'FontWeight', 'bold');
        title('各方法Deviance残差分布比较', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加零线
        hold on;
        plot(xlim, [0 0], 'k--');
        
        % 保存图形
        save_figure(fig2, figure_dir, 'deviance_residuals_comparison', 'Formats', {'svg'});
        close(fig2);
    end
catch ME
    log_message('warning', sprintf('创建残差比较图失败: %s', ME.message));
end
end

%% 修改4：改进系数维度与变量组合维度不匹配的处理
function coef_stability = monitor_coefficient_stability(results, methods, var_names)
% 监控模型系数的稳定性
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   var_names - 变量名称
% 输出:
%   coef_stability - 系数稳定性分析结果

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
                log_message('warning', sprintf('%s方法没有足够的有效系数，跳过系数稳定性分析', method));
                coef_stability.(method) = struct('status', 'insufficient_valid_data');
                continue;
            end
            
            % 检查系数维度是否与变量组合匹配
            expected_dim = length(combo_indices) + 1; % +1是截距项
            actual_dim = size(common_coefs, 2);
            
            log_message('info', sprintf('%s方法的系数维度=%d，变量组合维度=%d', ...
                method, actual_dim, expected_dim));
                
            % 如果维度不匹配，尝试调整
            if actual_dim ~= expected_dim
                log_message('warning', sprintf('%s方法的系数维度(%d)与变量组合维度(%d)不匹配，尝试调整', ...
                    method, actual_dim, expected_dim));
                
                % 根据实际情况调整
                if actual_dim > expected_dim
                    % 如果系数数量多于变量数量，取前面的部分（截距项和选择的变量）
                    log_message('info', sprintf('截取系数维度从%d到%d', actual_dim, expected_dim));
                    common_coefs = common_coefs(:, 1:expected_dim);
                elseif actual_dim < expected_dim && actual_dim > 0
                    % 如果系数数量少于变量数量但不为零，使用可用的系数（可能会导致变量名不匹配）
                    log_message('warning', sprintf('系数数量不足，统计可用的%d个系数', actual_dim));
                    expected_dim = actual_dim;
                else
                    % 极端情况：没有有效系数
                    log_message('warning', sprintf('%s方法没有有效系数', method));
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
            log_message('info', sprintf('%s方法的系数稳定性分析完成，分析了%d个模型', method, length(common_combo_indices)));
            
            % 识别不稳定的系数 (CV > 0.5)
            unstable_idx = find(coef_cv > 0.5);
            if ~isempty(unstable_idx)
                unstable_vars = var_list(unstable_idx);
                log_message('warning', sprintf('%s方法中检测到不稳定系数：', method));
                for i = 1:length(unstable_idx)
                    log_message('warning', sprintf('  %s: CV=%.2f', unstable_vars{i}, coef_cv(unstable_idx(i))));
                end
            else
                log_message('info', sprintf('%s方法的所有系数都表现稳定 (CV <= 0.5)', method));
            end
        else
            log_message('warning', sprintf('%s方法没有足够的模型使用最常见变量组合(只有%d个，需要至少2个)，跳过系数稳定性分析', ...
                method, length(common_combo_indices)));
            coef_stability.(method) = struct('status', 'insufficient_data');
        end
    else
        % 对于非回归类模型
        log_message('info', sprintf('%s方法不适用于传统系数稳定性分析', method));
        coef_stability.(method) = struct('status', 'not_applicable');
    end
end
end

%% 修改5：修复calculate_parameter_statistics函数中的维度处理问题
function param_stats = calculate_parameter_statistics(results, methods, var_names)
% 计算模型参数的置信区间和p值（同时基于BCa和t分布）
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   var_names - 变量名称
% 输出:
%   param_stats - 参数统计结果

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
                log_message('warning', sprintf('%s方法没有足够的有效系数，跳过参数统计分析', method));
                param_stats.(method) = struct('status', 'insufficient_valid_data');
                continue;
            end
            
            % 检查维度是否匹配
            expected_dim = length(combo_indices) + 1;  % +1是截距项
            actual_dim = size(common_coefs, 2);
            
            % 调整维度（如果需要）
            if actual_dim ~= expected_dim
                log_message('warning', sprintf('%s方法的系数维度(%d)与变量组合维度(%d)不匹配，尝试调整', ...
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
                    [bca_ci_lower(i), bca_ci_upper(i)] = calculate_bca_ci(theta_boot, 0.05);
                end
            catch ME
                log_message('warning', sprintf('BCa置信区间计算失败: %s，使用基本Bootstrap置信区间', ME.message));
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
            log_message('info', sprintf('%s方法的参数统计分析完成，分析了%d个模型', method, n_samples));
            
            % 输出显著的参数
            sig_idx = find(t_p_values < 0.05);
            if ~isempty(sig_idx)
                log_message('info', sprintf('%s方法中检测到显著参数 (p < 0.05)：', method));
                for i = 1:length(sig_idx)
                    log_message('info', sprintf('  %s: 估计值=%.4f, t-CI=[%.4f,%.4f], BCa-CI=[%.4f,%.4f], p=%.4f %s', ...
                        var_list{sig_idx(i)}, coef_mean(sig_idx(i)), ...
                        t_ci_lower(sig_idx(i)), t_ci_upper(sig_idx(i)), ...
                        bca_ci_lower(sig_idx(i)), bca_ci_upper(sig_idx(i)), ...
                        t_p_values(sig_idx(i)), significance{sig_idx(i)}));
                end
            else
                log_message('warning', sprintf('%s方法没有检测到显著参数 (p < 0.05)', method));
            end
        else
            log_message('warning', sprintf('%s方法没有足够的模型使用最常见变量组合(只有%d个，需要至少2个)，跳过参数统计分析', ...
                method, length(common_combo_indices)));
            param_stats.(method) = struct('status', 'insufficient_data');
        end
    else
        % 对于非回归类模型
        log_message('info', sprintf('%s方法不适用于传统参数统计分析', method));
        param_stats.(method) = struct('status', 'not_applicable');
    end
end
end

% 辅助函数：计算BCa置信区间
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

% 参数估计值及其置信区间的森林图（Forest Plot）
function create_parameter_comparison_across_methods(param_stats, methods, figure_dir)
    % 创建森林图（参数估计及其置信区间）
    % 输入:
    %   param_stats - 参数统计结果
    %   methods - 方法名称
    %   figure_dir - 图形保存目录
    
    % 选择一个最佳方法来创建森林图（通常是stepwise或elasticnet）
    viable_methods = {};
    method_data = {};
    
    % 收集可用的方法数据
    for i = 1:length(methods)
        method = methods{i};
        if isfield(param_stats, method) && isfield(param_stats.(method), 'table') && ...
           height(param_stats.(method).table) >= 3  % 至少需要截距+2个变量
            viable_methods{end+1} = method;
            method_data{end+1} = param_stats.(method);
        end
    end
    
    % 如果没有可用方法，尝试放宽条件
    if isempty(viable_methods) && length(methods) > 0
        for i = 1:length(methods)
            method = methods{i};
            if isfield(param_stats, method) && isfield(param_stats.(method), 'variables')
                if length(param_stats.(method).variables) >= 2  % 至少需要截距+1个变量
                    viable_methods{end+1} = method;
                    method_data{end+1} = param_stats.(method);
                end
            end
        end
    end
    
    % 如果仍然没有可用方法，记录警告并退出
    if isempty(viable_methods)
        log_message('warning', '没有足够的参数数据来创建森林图');
        return;
    end
    
    % 使用第一个可行方法创建森林图
    method = viable_methods{1};
    params = method_data{1};
    
    % 准备森林图数据
    var_names = params.variables;
    estimates = params.mean;
    
    % 获取置信区间
    % 优先使用t分布CI，其次使用BCa CI
    if isfield(params, 't_ci_lower') && isfield(params, 't_ci_upper')
        ci_lower = params.t_ci_lower;
        ci_upper = params.t_ci_upper;
        ci_type = 't分布';
    elseif isfield(params, 'bca_ci_lower') && isfield(params, 'bca_ci_upper')
        ci_lower = params.bca_ci_lower;
        ci_upper = params.bca_ci_upper;
        ci_type = 'BCa Bootstrap';
    else
        % 如果没有CI，使用标准差创建一个基本的CI
        if isfield(params, 'std')
            std_vals = params.std;
            ci_lower = estimates - 1.96 * std_vals;
            ci_upper = estimates + 1.96 * std_vals;
            ci_type = '基于标准差的近似';
        else
            % 如果连标准差都没有，无法创建森林图
            log_message('warning', '没有足够的参数置信区间数据来创建森林图');
            return;
        end
    end
    
    % 获取p值和显著性
    if isfield(params, 'p_values')
        p_values = params.p_values;
        
        % 创建显著性标记
        significance = cell(size(p_values));
        for i = 1:length(p_values)
            if p_values(i) < 0.001
                significance{i} = '***';
            elseif p_values(i) < 0.01
                significance{i} = '**';
            elseif p_values(i) < 0.05
                significance{i} = '*';
            elseif p_values(i) < 0.1
                significance{i} = '.';
            else
                significance{i} = '';
            end
        end
    else
        p_values = nan(size(estimates));
        significance = repmat({''}, size(estimates));
    end
    
    % 创建森林图
    fig = figure('Name', 'Forest Plot', 'Position', [100, 100, 1200, max(600, 100 + 30*length(var_names))]);
    
    % 计算Y轴位置
    y_pos = length(var_names):-1:1;
    
    % 绘制森林图
    hold on;
    
    % 绘制置信区间线
    for i = 1:length(var_names)
        plot([ci_lower(i), ci_upper(i)], [y_pos(i), y_pos(i)], 'b-', 'LineWidth', 1.5);
    end
    
    % 绘制估计点
    scatter(estimates, y_pos, 100, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b');
    
    % 添加参数标签
    for i = 1:length(var_names)
        text(min(ci_lower) - abs(min(ci_lower))*0.15, y_pos(i), var_names{i}, ...
            'HorizontalAlignment', 'right', 'FontSize', 10, 'FontWeight', 'bold');
        
        % 添加估计值和CI
        ci_text = sprintf('%.3f [%.3f, %.3f] %s', estimates(i), ci_lower(i), ci_upper(i), significance{i});
        text(max(ci_upper) + abs(max(ci_upper))*0.05, y_pos(i), ci_text, ...
            'HorizontalAlignment', 'left', 'FontSize', 9);
    end
    
    % 添加零线
    plot([0, 0], [0, length(var_names)+1], 'k--', 'LineWidth', 1);
    
    % 设置图形属性
    ylim([0, length(var_names)+1]);
    xlim([min(ci_lower) - abs(min(ci_lower))*0.2, max(ci_upper) + abs(max(ci_upper))*0.2]);
    title(sprintf('%s方法的参数估计值及其%s置信区间', method, ci_type), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('参数估计值', 'FontSize', 12);
    set(gca, 'YTick', []);
    grid on;
    box on;
    
    % 保存图形
    save_figure(fig, figure_dir, 'parameter_comparison_across_methods_plot', 'Formats', {'svg'});
    log_message('info', '参数估计值及其置信区间森林图（Forest Plot）已保存');
    close(fig);
end

% 参数显著性火山图（Volcano Plot）
function create_parameter_significance_volcano_plot(param_stats, methods, figure_dir)
% 创建参数显著性火山图
% 输入:
%   param_stats - 参数统计结果
%   methods - 方法名称
%   figure_dir - 图形保存目录

% 收集参数显著性信息
method_names = {};
var_names = {};
estimates = [];
p_values = [];
colors = lines(length(methods));

for i = 1:length(methods)
    method = methods{i};
    
    % 检查该方法是否有参数统计
    if isfield(param_stats, method) && isfield(param_stats.(method), 'table')
        % 提取参数表
        table_data = param_stats.(method).table;
        
        % 添加到结果数组
        for j = 1:height(table_data)
            method_names{end+1} = method;
            var_names{end+1} = table_data.Variable{j};
            estimates(end+1) = table_data.Estimate(j);
            p_values(end+1) = table_data.p_value(j);
        end
    end
end

% 如果没有收集到数据，则退出
if isempty(estimates)
    log_message('warning', '没有足够的参数数据来创建火山图');
    return;
end

% 转换p值为-log10(p)
log_p = -log10(p_values);

% 创建火山图
fig = figure('Name', 'Parameter Significance Volcano Plot', 'Position', [100, 100, 1000, 800]);

% 绘制散点图
hold on;
for i = 1:length(method_names)
    % 确定方法索引以获取颜色
    method_idx = find(strcmp(methods, method_names{i}));
    
    % 如果找不到匹配的方法，使用默认颜色
    if isempty(method_idx)
        color = [0.3, 0.3, 0.3];
    else
        color = colors(method_idx, :);
    end
    
    % 确定点大小（基于参数估计值）
    size_factor = 20 + 20 * abs(estimates(i)) / max(abs(estimates));
    
    % 绘制点
    scatter(estimates(i), log_p(i), size_factor, color, 'filled', 'MarkerFaceAlpha', 0.7);
    
    % 添加显著点标签
    if log_p(i) > -log10(0.05)
        text(estimates(i), log_p(i), var_names{i}, 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end

% 添加显著性阈值线
threshold_line = -log10(0.05);
line(xlim, [threshold_line, threshold_line], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1);
text(max(xlim)*0.9, threshold_line+0.2, 'p = 0.05', 'Color', 'r', 'FontSize', 10);

% 添加零线
line([0, 0], ylim, 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--', 'LineWidth', 1);

% 设置坐标轴
xlabel('参数估计值', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('-log10(p值)', 'FontSize', 12, 'FontWeight', 'bold');
title('参数显著性火山图', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加方法图例
legend_handles = [];
legend_labels = {};
for i = 1:length(methods)
    h = scatter(NaN, NaN, 100, colors(i,:), 'filled');
    legend_handles = [legend_handles, h];
    legend_labels{end+1} = methods{i};
end
legend(legend_handles, legend_labels, 'Location', 'best', 'FontSize', 10);

% 添加颜色条时使用正确的属性设置方法
% 不再使用直接设置Label属性的方法，而是使用Title属性

% 保存图形
save_figure(fig, figure_dir, 'parameter_significance_volcano', 'Formats', {'svg'});
close(fig);
end

% 各方法参数比较图
function create_parameter_comparison_across_methods_plot(results, methods, figure_dir)
% 创建不同方法之间的参数比较森林图
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   figure_dir - 图形保存目录

% 提取参数统计信息
method_names = {};
var_names = {};
estimates = [];
ci_lower = [];
ci_upper = [];
p_values = [];
colors = lines(length(methods));

for i = 1:length(methods)
    method = methods{i};
    
    % 检查该方法是否有参数统计
    if isfield(results, method) && isfield(results.(method), 'params')
        % 获取该方法的参数统计
        models = results.(method).models;
        
        % 获取最常见组合模型的索引
        var_combinations = results.(method).var_combinations;
        combo_strings = cellfun(@(x) sprintf('%d,', sort(x)), var_combinations, 'UniformOutput', false);
        [unique_combos, ~, ic] = unique(combo_strings);
        combo_counts = accumarray(ic, 1);
        [~, max_idx] = max(combo_counts);
        combo_indices = find(ic == max_idx);
        
        % 提取模型参数
        for j = 1:min(3, length(combo_indices)) % 限制为每个方法最多3个模型
            if combo_indices(j) <= length(models)
                mdl = models{combo_indices(j)};
                
                % 提取参数
                try
                    if isa(mdl, 'TreeBagger')
                        continue; % 跳过随机森林模型
                    elseif isa(mdl, 'GeneralizedLinearModel')
                        coefs = mdl.Coefficients;
                        est = coefs.Estimate;
                        stderr = coefs.SE;
                        pval = coefs.pValue;
                        
                        % 计算置信区间
                        tval = tinv(0.975, mdl.DFE);
                        ci_lo = est - tval * stderr;
                        ci_hi = est + tval * stderr;
                        
                        % 获取变量名
                        var_list = coefs.Properties.RowNames;
                    else
                        continue; % 跳过其他类型的模型
                    end
                    
                    % 添加到结果数组
                    for k = 1:length(est)
                        method_names{end+1} = method;
                        var_names{end+1} = var_list{k};
                        estimates(end+1) = est(k);
                        ci_lower(end+1) = ci_lo(k);
                        ci_upper(end+1) = ci_hi(k);
                        p_values(end+1) = pval(k);
                    end
                catch
                    % 如果提取失败，跳过
                    continue;
                end
            end
        end
    end
end

% 如果没有收集到数据，则退出
if isempty(estimates)
    log_message('warning', '没有足够的参数数据来创建森林图');
    return;
end

% 创建参数排序
[var_list, ~, var_groups] = unique(var_names);
var_counts = accumarray(var_groups, 1);

% 排序参数估计值 (按变量名称排序)
[sorted_vars, idx] = sort(var_list);
sorted_counts = var_counts(idx);

% 创建森林图
fig = figure('Name', 'Parameter Comparison Across Methods', 'Position', [100, 100, 1000, 800]);

% 设置子图位置
subplot('Position', [0.25, 0.1, 0.7, 0.8]);

% 计算Y轴位置
y_positions = [];
var_positions = [];
current_pos = 1;

for i = 1:length(sorted_vars)
    var_indices = find(strcmp(var_names, sorted_vars{i}));
    
    % 为当前变量生成Y位置
    for j = 1:length(var_indices)
        y_positions(var_indices(j)) = current_pos;
        current_pos = current_pos + 1;
    end
    
    % 记录变量中心位置
    var_positions(i) = mean(y_positions(var_indices));
    
    % 添加分隔空间
    current_pos = current_pos + 1;
end

% 绘制水平线
hold on;
for i = 1:length(y_positions)
    % 确定方法索引以获取颜色
    method_idx = find(strcmp(methods, method_names{i}));
    
    % 如果找不到匹配的方法，使用默认颜色
    if isempty(method_idx)
        color = [0.3, 0.3, 0.3];
    else
        color = colors(method_idx, :);
    end
    
    % 绘制估计值点
    plot(estimates(i), y_positions(i), 'o', 'MarkerSize', 8, 'MarkerFaceColor', color, 'MarkerEdgeColor', 'none');
    
    % 绘制置信区间线
    line([ci_lower(i), ci_upper(i)], [y_positions(i), y_positions(i)], 'Color', color, 'LineWidth', 2);
    
    % 绘制端点
    line([ci_lower(i), ci_lower(i)], [y_positions(i)-0.2, y_positions(i)+0.2], 'Color', color, 'LineWidth', 2);
    line([ci_upper(i), ci_upper(i)], [y_positions(i)-0.2, y_positions(i)+0.2], 'Color', color, 'LineWidth', 2);
    
    % 添加方法标签
    text(ci_upper(i) + 0.2, y_positions(i), method_names{i}, 'FontSize', 8, 'Color', color);
end

% 绘制零线
line([0, 0], [0, current_pos], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--', 'LineWidth', 1);

% 设置坐标轴
% 确保y_positions是递增的
[sorted_ypos, sort_idx] = sort(y_positions);
xlim([min(ci_lower) - 0.5, max(ci_upper) + 1]);
ylim([0, current_pos]);
set(gca, 'YTick', var_positions, 'YTickLabel', sorted_vars, 'FontSize', 10);
xlabel('参数估计值及95%置信区间', 'FontSize', 12, 'FontWeight', 'bold');
title('不同方法之间的参数比较', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 添加方法图例
legend_handles = [];
legend_labels = {};
for i = 1:length(methods)
    h = plot(NaN, NaN, 'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'none');
    legend_handles = [legend_handles, h];
    legend_labels{end+1} = methods{i};
end
legend(legend_handles, legend_labels, 'Location', 'best', 'FontSize', 10);

% 保存图形
save_figure(fig, figure_dir, 'parameter_comparison_across_methods', 'Formats', {'svg'});
close(fig);
end

% 参数置信区间比较图
function create_confidence_interval_comparison(param_stats, methods, figure_dir)
    % 创建参数置信区间比较图（t分布vs BCa方法）
    % 输入:
    %   param_stats - 参数统计结果
    %   methods - 方法名称
    %   figure_dir - 图形保存目录
    
    for m = 1:length(methods)
        method = methods{m};
        
        % 只处理有参数统计数据的回归类方法
        if isfield(param_stats, method) && ...
           isfield(param_stats.(method), 't_ci_lower') && ...
           isfield(param_stats.(method), 'bca_ci_lower')
            
            % 提取数据
            variables = param_stats.(method).variables;
            t_ci_lower = param_stats.(method).t_ci_lower;
            t_ci_upper = param_stats.(method).t_ci_upper;
            bca_ci_lower = param_stats.(method).bca_ci_lower;
            bca_ci_upper = param_stats.(method).bca_ci_upper;
            
            % 创建图形
            fig = figure('Name', sprintf('%s CI Comparison', method), 'Position', [100, 100, 1200, 600]);
            
            % 左侧：t分布置信区间
            subplot(1, 2, 1);
            for i = 1:length(variables)
                y_pos = length(variables) - i + 1;
                line([t_ci_lower(i), t_ci_upper(i)], [y_pos, y_pos], 'LineWidth', 2, 'Color', 'b');
                hold on;
                scatter(param_stats.(method).mean(i), y_pos, 80, 'filled', 'MarkerFaceColor', 'b');
            end
            
            % 设置图形属性
            set(gca, 'YTick', 1:length(variables), 'YTickLabel', flip(variables), 'FontSize', 10);
            xlabel('参数估计值', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('%s: t分布95%%置信区间', method), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            line([0, 0], [0, length(variables)+1], 'LineStyle', '--', 'Color', 'k');
            
            % 右侧：BCa置信区间
            subplot(1, 2, 2);
            for i = 1:length(variables)
                y_pos = length(variables) - i + 1;
                line([bca_ci_lower(i), bca_ci_upper(i)], [y_pos, y_pos], 'LineWidth', 2, 'Color', 'r');
                hold on;
                scatter(param_stats.(method).mean(i), y_pos, 80, 'filled', 'MarkerFaceColor', 'r');
            end
            
            % 设置图形属性
            set(gca, 'YTick', 1:length(variables), 'YTickLabel', flip(variables), 'FontSize', 10);
            xlabel('参数估计值', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('%s: BCa 95%%置信区间', method), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            line([0, 0], [0, length(variables)+1], 'LineStyle', '--', 'Color', 'k');
            
            % 保存图形
            save_figure(fig, figure_dir, sprintf('%s_ci_comparison', method), 'Formats', {'svg'});
            close(fig);
        end
    end
end

% 参数稳定性热图
function create_coefficient_variation_heatmap(results, methods, var_names, figure_dir)
    % 创建系数变异系数热力图
    % 输入:
    %   results - 结果结构
    %   methods - 方法名称
    %   var_names - 变量名称
    %   figure_dir - 图形保存目录
    
    % 初始化变异系数矩阵和方法列表
    valid_methods = {};
    cv_matrices = {};
    
    % 对每种方法计算变异系数
    for i = 1:length(methods)
        method = methods{i};
        
        % 提取性能数据中的系数
        if isfield(results, method) && isfield(results.(method), 'performance') && ...
           isfield(results.(method).performance, 'all_coefs')
            
            all_coefs = results.(method).performance.all_coefs;
            
            % 确保至少有3个有效系数数组
            valid_coefs = 0;
            max_length = 0;
            
            % 计算有效系数数量和最大长度
            for j = 1:length(all_coefs)
                if ~isempty(all_coefs{j}) && isnumeric(all_coefs{j}) && ~any(isnan(all_coefs{j}))
                    valid_coefs = valid_coefs + 1;
                    max_length = max(max_length, length(all_coefs{j}));
                end
            end
            
            if valid_coefs >= 3 && max_length > 1
                % 构建系数矩阵
                coef_matrix = nan(valid_coefs, max_length);
                idx = 1;
                
                for j = 1:length(all_coefs)
                    if ~isempty(all_coefs{j}) && isnumeric(all_coefs{j}) && ~any(isnan(all_coefs{j}))
                        % 修正长度不一致的问题
                        current_coefs = all_coefs{j};
                        if length(current_coefs) <= max_length
                            coef_matrix(idx, 1:length(current_coefs)) = current_coefs;
                        else
                            coef_matrix(idx, :) = current_coefs(1:max_length);
                        end
                        idx = idx + 1;
                    end
                end
                
                % 计算每个变量的均值和标准差
                coef_mean = nanmean(coef_matrix, 1);
                coef_std = nanstd(coef_matrix, 0, 1);
                
                % 计算变异系数，避免除以零问题
                coef_cv = zeros(size(coef_mean));
                for j = 1:length(coef_mean)
                    if abs(coef_mean(j)) > 1e-6  % 避免除以接近零的值
                        coef_cv(j) = abs(coef_std(j) / coef_mean(j));
                    else
                        if coef_std(j) > 1e-6  % 均值接近零但标准差不小
                            coef_cv(j) = 999;  % 表示高变异性
                        else  % 均值和标准差都接近零
                            coef_cv(j) = 0;    % 表示稳定（都是零）
                        end
                    end
                end
                
                % 添加到变异系数列表
                valid_methods{end+1} = method;
                cv_matrices{end+1} = coef_cv;
                
                log_message('info', sprintf('成功为方法%s计算变异系数', method));
            else
                log_message('warning', sprintf('无法为方法%s找到或计算变异系数', method));
            end
        else
            log_message('warning', sprintf('无法为方法%s找到或计算变异系数', method));
        end
    end
    
    % 如果没有有效方法，退出
    if isempty(valid_methods)
        log_message('warning', '没有足够的系数稳定性数据来创建热力图');
        return;
    end
    
    % 创建简化的变量名称列表（截断过长的名称）
    short_var_names = cell(size(var_names));
    for i = 1:length(var_names)
        if length(var_names{i}) > 15
            short_var_names{i} = [var_names{i}(1:12) '...'];
        else
            short_var_names{i} = var_names{i};
        end
    end
    
    % 创建热力图
    fig = figure('Name', 'Coefficient Variation Heatmap', 'Position', [100, 100, 1000, 800]);
    
    % 确定CV矩阵的最大尺寸
    max_cols = 0;
    for i = 1:length(cv_matrices)
        max_cols = max(max_cols, length(cv_matrices{i}));
    end
    
    % 创建热力图数据
    heatmap_data = nan(length(valid_methods), max_cols);
    
    for i = 1:length(valid_methods)
        cv = cv_matrices{i};
        heatmap_data(i, 1:length(cv)) = cv;
    end
    
    % 限制最大CV值，避免极端值影响颜色比例
    heatmap_data(heatmap_data > 2) = 2;
    
    % 创建变量标签
    var_labels = cell(1, max_cols);
    var_labels{1} = 'Intercept';
    for i = 2:max_cols
        if i-1 <= length(short_var_names)
            var_labels{i} = short_var_names{i-1};
        else
            var_labels{i} = sprintf('Var%d', i-1);
        end
    end
    
    % 创建热力图
    imagesc(heatmap_data);
    colormap(jet);
    colorbar;
    
    % 设置轴标签
    set(gca, 'XTick', 1:max_cols, 'XTickLabel', var_labels, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:length(valid_methods), 'YTickLabel', valid_methods);
    
    % 设置标题和标签
    title('各方法参数变异系数 (CV) 热力图', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('变量', 'FontSize', 12);
    ylabel('方法', 'FontSize', 12);
    
    % 给每个单元格添加CV值
    for i = 1:length(valid_methods)
        for j = 1:max_cols
            if ~isnan(heatmap_data(i, j))
                text(j, i, sprintf('%.2f', heatmap_data(i, j)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
    end
    
    % 调整图形大小
    set(gcf, 'Position', [100, 100, max(1000, 200 + 40*max_cols), max(800, 150 + 40*length(valid_methods))]);
    
    % 保存图形
    save_figure(fig, figure_dir, 'coefficient_variation_heatmap', 'Formats', {'svg'});
    log_message('info', '参数稳定性热力图已保存');
    close(fig);
end

% 参数估计箱线图
function create_parameter_boxplot(param_stats, methods, figure_dir)
% 创建参数估计箱线图
% 输入:
%   param_stats - 参数统计结果
%   methods - 方法名称
%   figure_dir - 图形保存目录

for m = 1:length(methods)
    method = methods{m};
    
    % 检查该方法是否有参数统计数据
    if isfield(param_stats, method) && isfield(param_stats.(method), 'table') && ...
       ~isempty(param_stats.(method).table)
        
        % 提取表格数据
        table_data = param_stats.(method).table;
        
        % 获取变量名和估计值
        var_names = table_data.Variable;
        estimates = table_data.Estimate;
        
        % 获取置信区间
        ci_lower = table_data.CI_Lower_t;
        ci_upper = table_data.CI_Upper_t;
        
        % 创建图形
        fig = figure('Name', sprintf('%s Parameter Boxplot', method), 'Position', [100, 100, 800, 600]);
        
        % 绘制箱线图 (使用误差条代替箱线图，因为我们只有点估计和置信区间)
        y_pos = length(var_names):-1:1;
        
        % 绘制估计点
        plot(estimates, y_pos, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none');
        hold on;
        
        % 绘制置信区间
        for i = 1:length(var_names)
            line([ci_lower(i), ci_upper(i)], [y_pos(i), y_pos(i)], 'Color', 'b', 'LineWidth', 1.5);
            
            % 绘制端点
            line([ci_lower(i), ci_lower(i)], [y_pos(i)-0.2, y_pos(i)+0.2], 'Color', 'b', 'LineWidth', 1.5);
            line([ci_upper(i), ci_upper(i)], [y_pos(i)-0.2, y_pos(i)+0.2], 'Color', 'b', 'LineWidth', 1.5);
        end
        
        % 绘制零线
        line([0, 0], [0, length(var_names)+1], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--', 'LineWidth', 1);
        
        % 设置坐标轴
        set(gca, 'YTick', y_pos, 'YTickLabel', var_names);
        xlabel('参数估计值', 'FontSize', 12);
        title(sprintf('%s方法的参数估计及95%%置信区间', method), 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加p值标记
        if isfield(table_data, 'p_value') && isfield(table_data, 'Significance')
            for i = 1:length(var_names)
                % 修正 - 不使用range函数，直接使用固定偏移
                offset = max(ci_upper) * 0.1; % 使用最大置信上限的10%作为偏移量
                text(ci_upper(i) + offset, y_pos(i), ...
                    sprintf('p=%.3f %s', table_data.p_value(i), table_data.Significance{i}), ...
                    'VerticalAlignment', 'middle', 'FontSize', 9);
            end
        end
        
        % 保存图形
        save_figure(fig, figure_dir, sprintf('%s_parameter_boxplot', method), 'Formats', {'svg'});
        log_message('info', sprintf('%s方法的参数估计箱线图已保存', method));
        close(fig);
    else
        log_message('warning', sprintf('%s方法没有有效的参数统计数据，跳过创建参数估计箱线图', method));
    end
end

% 创建汇总比较图
viable_methods = {};
method_data = {};

% 收集可用的方法数据
for i = 1:length(methods)
    method = methods{i};
    if isfield(param_stats, method) && isfield(param_stats.(method), 'table') && ...
       ~isempty(param_stats.(method).table)
        viable_methods{end+1} = method;
        method_data{end+1} = param_stats.(method).table;
    end
end

if ~isempty(viable_methods)
    % 寻找共有的变量
    all_vars = {};
    for i = 1:length(viable_methods)
        if iscell(method_data{i}.Variable)
            vars = method_data{i}.Variable;
        else
            % 确保变量名是cell数组格式
            vars = cellstr(method_data{i}.Variable);
        end
        all_vars = union(all_vars, vars);
    end
    
    % 为每个变量创建比较图
    for v = 1:length(all_vars)
        var_name = all_vars{v};
        
        % 收集该变量在各方法中的估计值和置信区间
        estimates = [];
        ci_lower = [];
        ci_upper = [];
        method_names = {};
        
        for i = 1:length(viable_methods)
            table_data = method_data{i};
            % 确保变量名比较使用正确的格式
            if iscell(table_data.Variable)
                var_idx = find(strcmp(table_data.Variable, var_name));
            else
                var_idx = find(strcmp(cellstr(table_data.Variable), var_name));
            end
            
            if ~isempty(var_idx)
                estimates = [estimates; table_data.Estimate(var_idx)];
                ci_lower = [ci_lower; table_data.CI_Lower_t(var_idx)];
                ci_upper = [ci_upper; table_data.CI_Upper_t(var_idx)];
                method_names{end+1} = viable_methods{i};
            end
        end
        
        if length(method_names) >= 2  % 至少需要两种方法才能比较
            % 创建比较图
            fig = figure('Name', sprintf('Variable Comparison: %s', var_name), 'Position', [100, 100, 800, 500]);
            
            % 绘制比较图
            y_pos = length(method_names):-1:1;
            
            % 绘制估计点
            plot(estimates, y_pos, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none');
            hold on;
            
            % 绘制置信区间
            for i = 1:length(method_names)
                line([ci_lower(i), ci_upper(i)], [y_pos(i), y_pos(i)], 'Color', 'b', 'LineWidth', 2);
                
                % 绘制端点
                line([ci_lower(i), ci_lower(i)], [y_pos(i)-0.2, y_pos(i)+0.2], 'Color', 'b', 'LineWidth', 2);
                line([ci_upper(i), ci_upper(i)], [y_pos(i)-0.2, y_pos(i)+0.2], 'Color', 'b', 'LineWidth', 2);
            end
            
            % 绘制零线
            line([0, 0], [0, length(method_names)+1], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--', 'LineWidth', 1);
            
            % 设置坐标轴
            set(gca, 'YTick', y_pos, 'YTickLabel', method_names);
            xlabel('参数估计值', 'FontSize', 12);
            title(sprintf('变量 %s 在各方法中的估计及95%%置信区间', var_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            % 保存图形
            var_filename = strrep(var_name, ' ', '_');
            var_filename = strrep(var_filename, '/', '_');  % 移除可能造成文件名问题的字符
            var_filename = strrep(var_filename, '\', '_');
            var_filename = strrep(var_filename, ':', '_');
            save_figure(fig, figure_dir, sprintf('var_comparison_%s', var_filename), 'Formats', {'svg'});
            log_message('info', sprintf('变量 %s 的方法比较图已保存', var_name));
            close(fig);
        end
    end
else
    log_message('warning', '没有足够的方法有参数统计数据，跳过创建参数比较图');
end
end

% 方法间比较：为每个参数创建箱线图比较不同方法
function create_methods_comparison_boxplot(results, methods, var_names, figure_dir)
    % 创建各方法共有变量的比较箱线图
    % 输入:
    %   results - 结果结构
    %   methods - 方法名称
    %   var_names - 变量名称
    %   figure_dir - 图形保存目录
    
    % 收集每种方法选择的变量
    selected_vars_by_method = cell(length(methods), 1);
    
    for i = 1:length(methods)
        method = methods{i};
        if isfield(results, method) && isfield(results.(method), 'selected_vars')
            selected_vars_by_method{i} = find(results.(method).selected_vars);
        else
            selected_vars_by_method{i} = [];
        end
    end
    
    % 找出所有方法共有的变量
    common_vars = [];
    if ~isempty(selected_vars_by_method) && ~isempty(selected_vars_by_method{1})
        common_vars = selected_vars_by_method{1};
        
        for i = 2:length(methods)
            if ~isempty(selected_vars_by_method{i})
                common_vars = intersect(common_vars, selected_vars_by_method{i});
            end
        end
    end
    
    % 如果没有共有变量，尝试放宽条件，找出至少有两种方法共有的变量
    if isempty(common_vars)
        all_vars = [];
        for i = 1:length(methods)
            all_vars = union(all_vars, selected_vars_by_method{i});
        end
        
        var_count = zeros(length(all_vars), 1);
        for i = 1:length(methods)
            for j = 1:length(all_vars)
                if any(selected_vars_by_method{i} == all_vars(j))
                    var_count(j) = var_count(j) + 1;
                end
            end
        end
        
        % 选择出现在至少两种方法中的变量
        common_vars = all_vars(var_count >= 2);
    end
    
    % 如果仍然没有共有变量，使用变量频率最高的前5个变量
    if isempty(common_vars)
        % 计算每个变量的选择频率
        var_freq = zeros(length(var_names), 1);
        for i = 1:length(methods)
            method = methods{i};
            if isfield(results, method) && isfield(results.(method), 'var_freq')
                var_freq = var_freq + results.(method).var_freq;
            end
        end
        
        % 选择频率最高的前5个变量
        [~, idx] = sort(var_freq, 'descend');
        common_vars = idx(1:min(5, length(idx)));
    end
    
    % 如果仍然没有共有变量，无法创建boxplot
    if isempty(common_vars)
        log_message('warning', '没有找到所有方法共有的变量，无法创建方法比较箱线图');
        return;
    end
    
    % 收集各方法中这些变量的系数
    coefs_by_method = cell(length(methods), length(common_vars));
    
    for i = 1:length(methods)
        method = methods{i};
        
        % 提取性能数据中的系数
        if isfield(results, method) && isfield(results.(method), 'performance') && ...
           isfield(results.(method).performance, 'all_coefs')
            
            all_coefs = results.(method).performance.all_coefs;
            
            % 遍历所有系数样本
            for j = 1:length(all_coefs)
                if ~isempty(all_coefs{j}) && isnumeric(all_coefs{j})
                    coef = all_coefs{j};
                    
                    % 对于每个共有变量
                    for k = 1:length(common_vars)
                        var_idx = common_vars(k);
                        
                        % 检查变量索引是否在系数范围内
                        % 注意系数可能包含截距，所以变量索引需要+1
                        if var_idx + 1 <= length(coef)
                            if isempty(coefs_by_method{i, k})
                                coefs_by_method{i, k} = coef(var_idx + 1);
                            else
                                coefs_by_method{i, k} = [coefs_by_method{i, k}; coef(var_idx + 1)];
                            end
                        end
                    end
                end
            end
        end
    end
    
    % 检查是否有足够的数据
    has_data = false;
    for i = 1:length(methods)
        for j = 1:length(common_vars)
            if ~isempty(coefs_by_method{i, j}) && length(coefs_by_method{i, j}) >= 3
                has_data = true;
                break;
            end
        end
        if has_data
            break;
        end
    end
    
    if ~has_data
        log_message('warning', '没有足够的系数数据来创建方法比较箱线图');
        return;
    end
    
    % 为每个共有变量创建一个boxplot
    for k = 1:length(common_vars)
        var_idx = common_vars(k);
        var_name = '';
        
        % 获取变量名称
        if var_idx <= length(var_names)
            var_name = var_names{var_idx};
        else
            var_name = sprintf('Var%d', var_idx);
        end
        
        % 收集该变量的所有系数
        var_coefs = [];
        var_methods = {};
        
        for i = 1:length(methods)
            if ~isempty(coefs_by_method{i, k}) && length(coefs_by_method{i, k}) >= 3
                var_coefs = [var_coefs; coefs_by_method{i, k}];
                var_methods = [var_methods; repmat({methods{i}}, length(coefs_by_method{i, k}), 1)];
            end
        end
        
        if ~isempty(var_coefs)
            % 创建箱线图
            fig = figure('Name', sprintf('Variable Comparison: %s', var_name), 'Position', [100, 100, 800, 600]);
            
            boxplot(var_coefs, var_methods, 'Notch', 'on');
            
            % 设置图形属性
            title(sprintf('变量 %s 在各方法中的系数分布', var_name), 'FontSize', 14, 'FontWeight', 'bold');
            ylabel('系数值', 'FontSize', 12);
            xlabel('方法', 'FontSize', 12);
            grid on;
            
            % 添加均值点
            hold on;
            methods_unique = unique(var_methods);
            for i = 1:length(methods_unique)
                method = methods_unique{i};
                method_coefs = var_coefs(strcmp(var_methods, method));
                
                % 计算均值和95%置信区间
                mean_val = mean(method_coefs);
                [~, ci_lower, ci_upper] = ttest(method_coefs);
                
                % 绘制均值点
                scatter(i, mean_val, 100, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
                
                % 添加均值标签
                text(i, max(method_coefs) + 0.1*range(var_coefs), ...
                    sprintf('均值: %.3f\n95%%CI: [%.3f, %.3f]', mean_val, ci_lower, ci_upper), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8);
            end
            
            % 添加零线
            plot(xlim, [0, 0], 'k--', 'LineWidth', 1);
            
            % 保存图形
            save_figure(fig, figure_dir, sprintf('var_comparison_%s', strrep(var_name, ' ', '_')), 'Formats', {'svg'});
            close(fig);
        end
    end
    
    % 创建汇总比较图
    fig = figure('Name', 'Methods Comparison Summary', 'Position', [100, 100, 1200, 800]);
    
    % 收集每个方法的平均系数
    n_methods = length(methods);
    n_vars = length(common_vars);
    
    mean_coefs = nan(n_methods, n_vars);
    
    for i = 1:n_methods
        for j = 1:n_vars
            if ~isempty(coefs_by_method{i, j})
                mean_coefs(i, j) = mean(coefs_by_method{i, j}, 'omitnan');
            end
        end
    end
    
    % 创建柱状图
    bar_h = bar(mean_coefs);
    
    % 设置图形属性
    var_names_short = cell(size(common_vars));
    for i = 1:length(common_vars)
        if common_vars(i) <= length(var_names)
            var_name = var_names{common_vars(i)};
            if length(var_name) > 15
                var_names_short{i} = [var_name(1:12) '...'];
            else
                var_names_short{i} = var_name;
            end
        else
            var_names_short{i} = sprintf('Var%d', common_vars(i));
        end
    end
    
    legend(var_names_short, 'Location', 'best');
    
    % 设置x轴标签
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', methods);
    
    % 设置标题和标签
    title('各方法中共有变量的平均系数比较', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('平均系数值', 'FontSize', 12);
    xlabel('方法', 'FontSize', 12);
    grid on;
    
    % 保存图形
    save_figure(fig, figure_dir, 'method_comparison_summary', 'Formats', {'svg'});
    log_message('info', '参数统计箱线图已保存');
    close(fig);
end

% 参数稳定性比较：为每个方法创建所有参数的稳定性箱线图
function create_parameter_stability_boxplot(param_stats, methods, figure_dir)
    % 为每个方法创建参数稳定性箱线图
    for m = 1:length(methods)
        method = methods{m};
        
        if isfield(param_stats, method) && isfield(param_stats.(method), 'all_coefs') && ...
           isfield(param_stats.(method), 'variables')
            
            variables = param_stats.(method).variables;
            all_coefs = param_stats.(method).all_coefs;
            
            % 提取每个变量在所有模型中的系数值
            param_matrix = [];
            for i = 1:length(all_coefs)
                if length(all_coefs{i}) >= length(variables)
                    param_matrix = [param_matrix; all_coefs{i}(1:length(variables))'];
                end
            end
            
            if ~isempty(param_matrix) && size(param_matrix, 1) >= 5  % 至少需要5个样本
                fig = figure('Name', sprintf('%s Parameter Stability', method), 'Position', [100, 100, 1000, 600]);
                
                % 创建箱线图
                boxplot(param_matrix, 'Labels', variables, 'Notch', 'on');
                
                % 设置图形属性
                title(sprintf('%s方法的参数稳定性分析', method), 'FontSize', 14, 'FontWeight', 'bold');
                xlabel('参数', 'FontSize', 12);
                ylabel('估计值', 'FontSize', 12);
                xtickangle(45);  % 倾斜变量名称以免重叠
                grid on;
                
                % 添加零线
                hold on;
                line([0, length(variables)+1], [0, 0], 'LineStyle', '--', 'Color', 'k');
                
                % 标记显著参数
                for i = 1:length(variables)
                    if isfield(param_stats.(method), 'p_values') && ...
                       param_stats.(method).p_values(i) < 0.05
                        text(i, max(param_matrix(:,i))*1.1, '*', 'FontSize', 16, 'Color', 'r', ...
                            'HorizontalAlignment', 'center');
                    end
                end
                
                % 计算变异系数并添加到图上
                cv_values = std(param_matrix) ./ abs(mean(param_matrix));
                for i = 1:length(variables)
                    if ~isinf(cv_values(i)) && ~isnan(cv_values(i))
                        text(i, min(param_matrix(:,i))*1.1, sprintf('CV=%.2f', cv_values(i)), ...
                            'FontSize', 8, 'HorizontalAlignment', 'center');
                    end
                end
                
                % 保存图形
                save_figure(fig, figure_dir, sprintf('%s_parameter_stability', method), 'Formats', {'svg'});
                close(fig);
                
                % 创建变异系数条形图
                fig2 = figure('Name', sprintf('%s CV Barplot', method), 'Position', [100, 100, 900, 500]);
                
                % 排除无效或无限的变异系数
                valid_idx = ~isinf(cv_values) & ~isnan(cv_values);
                valid_cv = cv_values(valid_idx);
                valid_vars = variables(valid_idx);
                
                % 按变异系数排序
                [sorted_cv, sort_idx] = sort(valid_cv, 'descend');
                sorted_vars = valid_vars(sort_idx);
                
                % 创建条形图
                barh(sorted_cv);
                
                % 设置图形属性
                set(gca, 'YTick', 1:length(sorted_vars), 'YTickLabel', sorted_vars);
                xlabel('变异系数 (CV)', 'FontSize', 12);
                title(sprintf('%s方法的参数稳定性 (变异系数)', method), 'FontSize', 14, 'FontWeight', 'bold');
                grid on;
                
                % 添加参考线
                hold on;
                line([0.5, 0.5], [0, length(sorted_vars)+1], 'LineStyle', '--', 'Color', 'r');
                text(0.51, 1, '不稳定阈值 (CV>0.5)', 'Color', 'r', 'FontSize', 10);
                
                % 保存图形
                save_figure(fig2, figure_dir, sprintf('%s_parameter_cv', method), 'Formats', {'svg'});
                close(fig2);
            else
                log_message('warning', sprintf('%s方法没有足够的样本来创建参数稳定性箱线图', method));
            end
        end
    end
end

%% 评估变量贡献函数 - 新增
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

var_contribution = struct();
n_vars = length(var_names);

% 1. 全局变量重要性分析 - 使用全数据集
log_message('info', '开始全局变量重要性分析...');

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
    
    log_message('info', '相关性分析完成');
catch ME
    log_message('warning', sprintf('相关性分析失败: %s', ME.message));
end

% 1.2 基于模型的重要性分析
% 尝试使用逻辑回归和随机森林两种方法

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
    
    log_message('info', '逻辑回归变量重要性分析完成');
catch ME
    log_message('warning', sprintf('逻辑回归变量重要性分析失败: %s', ME.message));
end

% 1.2.2 随机森林模型
try
    if exist('TreeBagger', 'file')
        % 移除可能导致问题的并行控制选项
        % 修改前: parallelOptions = statset('UseParallel', true, 'UseSubstreams', true);
        % 修改后:
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
        
        log_message('info', '随机森林变量重要性分析完成');
    else
        log_message('warning', 'TreeBagger不可用，跳过随机森林变量重要性分析');
    end
catch ME
    log_message('warning', sprintf('随机森林变量重要性分析失败: %s', ME.message));
end

% 2. 方法特定变量贡献分析
log_message('info', '开始方法特定变量贡献分析...');

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
            
            log_message('info', sprintf('%s方法的变量贡献分析完成', method));
        catch ME
            log_message('warning', sprintf('%s方法的变量贡献分析失败: %s', method, ME.message));
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
                
                log_message('info', sprintf('%s方法的变量贡献分析完成', method));
            else
                log_message('warning', sprintf('%s方法的变量贡献分析失败：TreeBagger不可用或无选定变量', method));
            end
        catch ME
            log_message('warning', sprintf('%s方法的变量贡献分析失败: %s', method, ME.message));
        end
    end
end

% 3. 综合变量重要性排名
log_message('info', '计算综合变量重要性排名...');
try
    % 收集所有方法的变量选择频率
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
    
    log_message('info', '综合变量重要性排名计算完成');
    
    % 输出前5个最重要的变量
    top_n = min(5, n_vars);
    log_message('info', '前5个最重要的变量:');
    for i = 1:top_n
        log_message('info', sprintf('  %d. %s (重要性: %.2f%%)', i, overall_importance.Variable{i}, overall_importance.Normalized_Importance(i)));
    end
catch ME
    log_message('warning', sprintf('计算综合变量重要性排名失败: %s', ME.message));
end

end

%% 并行方法处理函数
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
log_message('info', sprintf('%s: 开始变量筛选', method));
[selected_vars, var_freq, var_combinations] = select_variables(X_final, y, train_indices, method);
log_message('info', sprintf('%s: 变量筛选完成', method));

% 训练和评估模型
log_message('info', sprintf('%s: 开始模型训练与评估', method));
[models, performance, group_performance] = train_and_evaluate_models_with_groups(X_final, y, train_indices, test_indices, var_combinations, method, var_names);
log_message('info', sprintf('%s: 模型训练与评估完成', method));

% 获取模型参数
log_message('info', sprintf('%s: 提取模型参数', method));
params = get_model_parameters(models, var_names);
log_message('info', sprintf('%s: 参数提取完成', method));

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

%% 模型参数函数 - 优化版
function params = get_model_parameters(models, var_names)
% 获取模型参数
% 输入:
%   models - 模型
%   var_names - 变量名称
% 输出:
%   params - 模型参数

n_models = length(models);

% 预分配并行处理数组
coef_cell = cell(n_models, 1);
pval_cell = cell(n_models, 1);
var_cell = cell(n_models, 1);

% 并行处理模型参数
parfor i = 1:n_models
    mdl = models{i};
    
    if isa(mdl, 'TreeBagger')
        % 对于 Random Forest，使用变量重要性作为"系数"
        local_imp = mdl.OOBPermutedPredictorDeltaError;
        local_coef = local_imp;
        local_pval = nan(size(local_imp));
        local_vars = cellstr(mdl.PredictorNames);
    else
        % 对于广义线性模型
        try
            % 尝试获取系数表
            coef_table = mdl.Coefficients;
            local_coef = coef_table.Estimate';
            local_pval = coef_table.pValue';
            local_vars = coef_table.Row';
        catch
            % 如果上面的方法失败，使用备选方法
            local_coef = mdl.Coefficients{:, 'Estimate'}';
            local_pval = mdl.Coefficients{:, 'pValue'}';
            local_vars = mdl.Coefficients.Properties.RowNames';
        end
    end
    
    % 存储结果
    coef_cell{i} = local_coef;
    pval_cell{i} = local_pval;
    var_cell{i} = local_vars;
end

% 构建参数结构
params = struct();
params.coef_cell = coef_cell;
params.pval_cell = pval_cell;
params.var_cell = var_cell;
end

%% 保存增强结果函数 - 新增
function save_enhanced_results(results, var_names, group_means, cv_results, coef_stability, param_stats, var_contribution)
    % 保存分析结果，包括变量组合信息、交叉验证结果、系数稳定性、参数统计和变量贡献
    % 输入:
    %   results - 结果结构
    %   var_names - 变量名称
    %   group_means - 分组均值
    %   cv_results - 交叉验证结果
    %   coef_stability - 系数稳定性
    %   param_stats - 参数统计
    %   var_contribution - 变量贡献

    % 从results结构中获取方法名称
    methods = fieldnames(results);

    % 创建高级结果目录
    result_dir = fullfile('results');
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    % 保存时间戳
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    result_mat = fullfile(result_dir, sprintf('enhanced_analysis_%s.mat', timestamp));

    % 保存原始结果 - 使用-v7.3支持大文件
    try
        save(result_mat, 'results', 'var_names', 'group_means', 'cv_results', ...
             'coef_stability', 'param_stats', 'var_contribution', '-v7.3');
        log_message('info', sprintf('增强分析结果已保存至 %s', result_mat));
    catch ME
        log_message('error', sprintf('保存MAT文件时出错: %s', ME.message));
    end

    % 创建CSV结果目录
    csv_dir = fullfile(result_dir, 'csv');
    if ~exist(csv_dir, 'dir')
        mkdir(csv_dir);
    end

    % 创建图形结果目录
    figure_dir = fullfile(result_dir, 'figures');
    if ~exist(figure_dir, 'dir')
        mkdir(figure_dir);
    end

    % 创建报告目录
    report_dir = fullfile(result_dir, 'reports');
    if ~exist(report_dir, 'dir')
        mkdir(report_dir);
    end

    %% 1. 创建CSV统计表
    
    % 创建AIC和BIC比较表
    file_path = fullfile(csv_dir, 'aic_bic_comparison.csv');
    if ~exist(file_path, 'file')
        try
            aic_bic_table = create_aic_bic_table(results, methods);
            writetable(aic_bic_table, file_path);
            log_message('info', 'AIC和BIC比较表已保存');
        catch ME
            log_message('error', sprintf('创建AIC和BIC比较表时出错: %s', ME.message));
        end
    end
    
    % 创建变量选择频率表
    file_path = fullfile(csv_dir, 'variable_selection_frequency.csv');
    if ~exist(file_path, 'file')
        try
            var_freq_table = create_var_freq_table(results, methods, var_names);
            writetable(var_freq_table, file_path);
            log_message('info', '变量选择频率表已保存');
        catch ME
            log_message('error', sprintf('创建变量选择频率表时出错: %s', ME.message));
        end
    end
    
    % 创建模型性能表（增强版，包括均值和标准差）
    file_path = fullfile(csv_dir, 'model_performance_detailed.csv');
    if ~exist(file_path, 'file')
        % 创建详细性能表（现有代码）
        try
            perf_detail_table = create_performance_detail_table(results, methods);
            writetable(perf_detail_table, fullfile(csv_dir, 'model_performance_detailed.csv'));
            log_message('info', '详细模型性能表已保存');
        catch ME
            log_message('error', sprintf('创建详细模型性能表时出错: %s', ME.message));
        end
        
        % 创建增强性能表（新增代码）
        try
            enhanced_perf_table = create_enhanced_performance_table(results, methods);
            writetable(enhanced_perf_table, fullfile(csv_dir, 'model_performance_enhanced.csv'));
            log_message('info', '增强模型性能表已保存');
        catch ME
            log_message('error', sprintf('创建增强模型性能表时出错: %s', ME.message));
        end
    end
    
    % 创建变量组合性能表
    file_path = fullfile(csv_dir, 'variable_group_performance.csv');
    if ~exist(file_path, 'file')
        try
            var_group_table = create_variable_group_table(results, methods);
            writetable(var_group_table, file_path);
            log_message('info', '变量组合性能表已保存');
        catch ME
            log_message('error', sprintf('创建变量组合性能表时出错: %s', ME.message));
        end
    end
    
    % 创建K折交叉验证结果表
    file_path = fullfile(csv_dir, 'cv_results.csv');
    if ~exist(file_path, 'file')
        try
            cv_table = create_cv_results_table(cv_results);
            writetable(cv_table, file_path);
            log_message('info', 'K折交叉验证结果表已保存');
        catch ME
            log_message('error', sprintf('创建K折交叉验证结果表时出错: %s', ME.message));
        end
    end
    
    % 创建模型参数表
    file_path = fullfile(csv_dir, 'model_parameters.csv');
    if ~exist(file_path, 'file')
        try
            param_table = create_parameter_table(results, methods);
            writetable(param_table, file_path);
            log_message('info', '模型参数表已保存');
        catch ME
            log_message('error', sprintf('创建模型参数表时出错: %s', ME.message));
        end
    end
    
    % 创建各方法系数稳定性表
    try
        for m = 1:length(methods)
            method = methods{m};
            file_path = fullfile(csv_dir, sprintf('%s_coefficient_stability.csv', method));
            if ~exist(file_path, 'file')
                if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
                    writetable(coef_stability.(method).table, file_path);
                    log_message('info', sprintf('%s方法的系数稳定性表已保存', method));
                end
            end
        end
    catch ME
        log_message('error', sprintf('创建系数稳定性表时出错: %s', ME.message));
    end
    
    % 创建各方法参数统计表
    try
        for m = 1:length(methods)
            method = methods{m};
            file_path = fullfile(csv_dir, sprintf('%s_parameter_statistics.csv', method));
            if ~exist(file_path, 'file')
                if isfield(param_stats, method) && isfield(param_stats.(method), 'table')
                    writetable(param_stats.(method).table, file_path);
                    log_message('info', sprintf('%s方法的参数统计表已保存', method));
                end
            end
        end
    catch ME
        log_message('error', sprintf('创建参数统计表时出错: %s', ME.message));
    end
    
    % 创建变量贡献相关表
    try
        % 保存全局重要性表
        if isfield(var_contribution, 'correlation')
            file_path = fullfile(csv_dir, 'correlation_importance.csv');
            if ~exist(file_path, 'file')
                writetable(var_contribution.correlation, file_path);
            end
        end
        
        if isfield(var_contribution, 'logistic')
            file_path = fullfile(csv_dir, 'logistic_importance.csv');
            if ~exist(file_path, 'file')
                writetable(var_contribution.logistic, file_path);
            end
        end
        
        if isfield(var_contribution, 'randomforest')
            file_path = fullfile(csv_dir, 'randomforest_importance.csv');
            if ~exist(file_path, 'file')
                writetable(var_contribution.randomforest, file_path);
            end
        end
        
        if isfield(var_contribution, 'overall_importance')
            file_path = fullfile(csv_dir, 'overall_importance.csv');
            if ~exist(file_path, 'file')
                writetable(var_contribution.overall_importance, file_path);
            end
        end
        
        % 保存方法特定贡献表
        for m = 1:length(methods)
            method = methods{m};
            file_path = fullfile(csv_dir, sprintf('%s_variable_contribution.csv', method));
            if ~exist(file_path, 'file')
                if isfield(var_contribution, 'methods') && isfield(var_contribution.methods, method) && ...
                   isfield(var_contribution.methods.(method), 'contribution_table')
                    writetable(var_contribution.methods.(method).contribution_table, file_path);
                    log_message('info', sprintf('%s方法的变量贡献表已保存', method));
                end
            end
        end
        
        log_message('info', '变量贡献表已保存');
    catch ME
        log_message('error', sprintf('创建变量贡献表时出错: %s', ME.message));
    end

    %% 2. 创建可视化图表
    
    % ROC曲线图
    figure_path = fullfile(figure_dir, 'roc_curves.svg');
    if ~exist(figure_path, 'file')
        try
            create_roc_curves(results, methods, figure_dir);
            log_message('info', 'ROC曲线图已保存');
        catch ME
            log_message('error', sprintf('创建ROC曲线图时出错: %s', ME.message));
        end
    end
    
    % 变量重要性图
    figure_path = fullfile(figure_dir, 'variable_importance.svg');
    if ~exist(figure_path, 'file')
        try
            create_variable_importance_plot(results, methods, var_names, figure_dir);
            log_message('info', '变量重要性图已保存');
        catch ME
            log_message('error', sprintf('创建变量重要性图时出错: %s', ME.message));
        end
    end
    
    % 变量组合可视化图
    figure_path = fullfile(figure_dir, 'top_combinations_comparison.svg');
    if ~exist(figure_path, 'file')
        try
            create_variable_group_plot(results, methods, var_names, figure_dir);
            log_message('info', '变量组合可视化图已保存');
        catch ME
            log_message('error', sprintf('创建变量组合可视化图时出错: %s', ME.message));
        end
    end
    
    % K折交叉验证性能图
    figure_path = fullfile(figure_dir, 'cv_performance.svg');
    if ~exist(figure_path, 'file')
        try
            create_cv_performance_plot(cv_results, figure_dir);
            log_message('info', 'K折交叉验证性能图已保存');
        catch ME
            log_message('error', sprintf('创建K折交叉验证性能图时出错: %s', ME.message));
        end
    end
    
    % 系数稳定性图
    figure_path = fullfile(figure_dir, 'coefficient_stability_comparison.svg');
    if ~exist(figure_path, 'file')
        try
            create_coefficient_stability_plot(coef_stability, methods, figure_dir);
            log_message('info', '系数稳定性图已保存');
        catch ME
            log_message('error', sprintf('创建系数稳定性图时出错: %s', ME.message));
        end
    end
    
    % 变量贡献图
    figure_path = fullfile(figure_dir, 'overall_variable_importance.svg');
    if ~exist(figure_path, 'file')
        try
            create_variable_contribution_plot(var_contribution, figure_dir);
            log_message('info', '变量贡献图已保存');
        catch ME
            log_message('error', sprintf('创建变量贡献图时出错: %s', ME.message));
        end
    end
    
    % 创建箱线图可视化
    metrics = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc'};
    boxplot_created = false;
    for i = 1:length(metrics)
        metric = metrics{i};
        figure_path = fullfile(figure_dir, sprintf('boxplot_%s.svg', metric));
        if ~exist(figure_path, 'file') && ~boxplot_created
            try
                create_boxplot_visualization(results, methods, figure_dir);
                boxplot_created = true; % 函数会创建所有箱线图，所以只需调用一次
                log_message('info', '箱线图可视化已保存');
            catch ME
                log_message('error', sprintf('创建箱线图可视化时出错: %s', ME.message));
                boxplot_created = true; % 即使出错也不再尝试
            end
        end
    end
    
    % 创建PR曲线
    figure_path = fullfile(figure_dir, 'precision_recall_curves.svg');
    if ~exist(figure_path, 'file')
        try
            create_pr_curves(results, methods, figure_dir);
            log_message('info', 'PR曲线图已保存');
        catch ME
            log_message('error', sprintf('创建PR曲线图时出错: %s', ME.message));
        end
    end
    
    % 创建校准曲线
    figure_path = fullfile(figure_dir, 'calibration_curves.svg');
    if ~exist(figure_path, 'file')
        try
            create_calibration_curves(results, methods, figure_dir);
            log_message('info', '校准曲线图已保存');
        catch ME
            log_message('error', sprintf('创建校准曲线图时出错: %s', ME.message));
        end
    end
    
    % 创建混淆矩阵
    figure_path = fullfile(figure_dir, 'confusion_matrix_comparison.svg');
    if ~exist(figure_path, 'file')
        try
            create_confusion_matrices(results, methods, figure_dir);
            log_message('info', '混淆矩阵图已保存');
        catch ME
            log_message('error', sprintf('创建混淆矩阵图时出错: %s', ME.message));
        end
    end
    
    % 创建参数估计值及其置信区间森林图（Forest Plot）
    figure_path = fullfile(figure_dir, 'parameter_forest.svg');
    if ~exist(figure_path, 'file')
        try
            create_parameter_comparison_across_methods(param_stats, methods, figure_dir);
            log_message('info', '参数估计值及其置信区间森林图（Forest Plot）已保存');
        catch ME
            log_message('error', sprintf('创建参数估计值及其置信区间森林图（Forest Plot）时出错: %s', ME.message));
        end
    end

    % 创建参数显著性火山图（Volcano Plot）
    figure_path = fullfile(figure_dir, 'parameter_significance_volcano.svg');
    if ~exist(figure_path, 'file')
        try
            create_parameter_significance_volcano_plot(param_stats, methods, figure_dir);
            log_message('info', '参数显著性火山图（Volcano Plot）已保存');
        catch ME
            log_message('error', sprintf('创建参数显著性火山图（Volcano Plot）时出错: %s', ME.message));
        end
    end

    % 创建各方法参数比较图
    figure_path = fullfile(figure_dir, 'parameter_comparison_across_methods.svg');
    if ~exist(figure_path, 'file')
        try
            create_parameter_comparison_across_methods_plot(param_stats, methods, figure_dir);
            log_message('info', '各方法参数比较图已保存');
        catch ME
            log_message('error', sprintf('创建各方法参数比较图时出错: %s', ME.message));
        end
    end

    % 创建参数置信区间比较图
    figure_path = fullfile(figure_dir, 'confidence_interval_comparison.svg');
    if ~exist(figure_path, 'file')
        try
            create_confidence_interval_comparison(param_stats, methods, figure_dir);
            log_message('info', '参数置信区间比较图已保存');
        catch ME
            log_message('error', sprintf('创建参数置信区间比较图时出错: %s', ME.message));
        end
    end

    % 创建参数稳定性热图
    figure_path = fullfile(figure_dir, 'parameter_stability_heatmap.svg');
    if ~exist(figure_path, 'file')
        try
            create_coefficient_variation_heatmap(results, methods, var_names, figure_dir)
            log_message('info', '参数稳定性热力图已保存');
        catch ME
            log_message('error', sprintf('创建参数稳定性热力图时出错: %s', ME.message));
        end
    end

    % 创建参数估计箱线图
    figure_path = fullfile(figure_dir, 'parameter_boxplot.svg');
    if ~exist(figure_path, 'file')
        try
            create_parameter_boxplot(param_stats, methods, figure_dir);
            log_message('info', '参数统计箱线图已保存');
        catch ME
            log_message('error', sprintf('创建参数统计箱线图时出错: %s', ME.message));
        end
    end

    % 创建性能指标雷达图
    figure_path = fullfile(figure_dir, 'performance_radar.svg');
    if ~exist(figure_path, 'file')
        try
            create_performance_radar_chart(results, methods, figure_dir);
            log_message('info', '增性能指标雷达图已保存');
        catch ME
            log_message('error', sprintf('创建性能指标雷达图时出错: %s', ME.message));
        end
    end

    % 创建性能指标热图
    figure_path = fullfile(figure_dir, 'performance_heatmap.svg');
    if ~exist(figure_path, 'file')
        try
            create_performance_heatmap(results, methods, figure_dir);
            log_message('info', '性能指标热图已保存');
        catch ME
            log_message('error', sprintf('创建性能指标热图时出错: %s', ME.message));
        end
    end

    % 创建性能指标对比条形图
    figure_path = fullfile(figure_dir, 'performance_barplot_comparison.svg');
    if ~exist(figure_path, 'file')
        try
            create_performance_barplot_comparison(results, methods, figure_dir);
            log_message('info', '性能指标对比条形图已保存');
        catch ME
            log_message('error', sprintf('创建性能指标对比条形图时出错: %s', ME.message));
        end
    end

    % 创建性能指标箱线图
    figure_path = fullfile(figure_dir, 'performance_distribution.svg');
    if ~exist(figure_path, 'file')
        try
            create_performance_distribution_plot(results, methods, figure_dir);
            log_message('info', '性能指标箱线图已保存');
        catch ME
            log_message('error', sprintf('创建性能指标箱线图时出错: %s', ME.message));
        end
    end

    % 创建综合性能评分图
    figure_path = fullfile(figure_dir, 'comprehensive_performance.svg');
    if ~exist(figure_path, 'file')
        try
            create_comprehensive_performance_plot(results, methods, figure_dir);
            log_message('info', '综合性能评分图已保存');
        catch ME
            log_message('error', sprintf('创建综合性能评分图时出错: %s', ME.message));
        end
    end

    % 方法间参数比较箱线图
    figure_path = fullfile(figure_dir, 'methods_comparison_boxplot.svg');
    if ~exist(figure_path, 'file')
        try
            create_methods_comparison_boxplot(results, methods, var_names, figure_dir);
            log_message('info', '方法间参数比较箱线图已保存');
        catch ME
            log_message('error', sprintf('创建方法间参数比较箱线图时出错: %s', ME.message));
        end
    end

    % 参数稳定性比较箱线图
    figure_path = fullfile(figure_dir, 'parameter_stability_boxplot.svg');
    if ~exist(figure_path, 'file')
        try
            create_parameter_stability_boxplot(param_stats, methods, figure_dir);
            log_message('info', '参数稳定性比较箱线图已保存');
        catch ME
            log_message('error', sprintf('创建参数稳定性比较箱线图时出错: %s', ME.message));
        end
    end

    %% 3. 创建综合比较报告
    report_path = fullfile(report_dir, 'enhanced_summary_report.txt');
    if ~exist(report_path, 'file')
        try
            create_enhanced_summary_report(results, methods, var_names, cv_results, ...
                coef_stability, param_stats, var_contribution, report_dir);
            log_message('info', '增强综合比较报告已保存');
        catch ME
            log_message('error', sprintf('创建增强综合比较报告时出错: %s', ME.message));
        end
    end
end

% 辅助函数：创建AIC和BIC比较表
function aic_bic_table = create_aic_bic_table(results, methods)
% 创建AIC和BIC比较表
% 输入:
%   results - 结果结构
%   methods - 方法名称
% 输出:
%   aic_bic_table - AIC和BIC比较表

% 初始化变量
methods_cell = cell(length(methods), 1);
aic_values = zeros(length(methods), 1);
aic_std_values = zeros(length(methods), 1);
bic_values = zeros(length(methods), 1);
bic_std_values = zeros(length(methods), 1);
n_params = zeros(length(methods), 1);

% 提取每种方法的AIC和BIC
for i = 1:length(methods)
    method = methods{i};
    methods_cell{i} = method;
    
    % 检查该方法是否有AIC和BIC值
    if isfield(results.(method).performance, 'avg_aic') && ...
       isfield(results.(method).performance, 'avg_bic')
        aic_values(i) = results.(method).performance.avg_aic;
        aic_std_values(i) = results.(method).performance.std_aic;
        bic_values(i) = results.(method).performance.avg_bic;
        bic_std_values(i) = results.(method).performance.std_bic;
        
        % 获取参数数量（基于最常见的变量组合）
        selected_vars = find(results.(method).selected_vars);
        n_params(i) = length(selected_vars) + 1; % +1是因为有截距项
    else
        aic_values(i) = NaN;
        aic_std_values(i) = NaN;
        bic_values(i) = NaN;
        bic_std_values(i) = NaN;
        n_params(i) = 0;
    end
end

% 创建表格
aic_bic_table = table(methods_cell, n_params, aic_values, aic_std_values, bic_values, bic_std_values, ...
    'VariableNames', {'Method', 'NumParams', 'AIC', 'AIC_StdDev', 'BIC', 'BIC_StdDev'});

% 按AIC排序
aic_bic_table = sortrows(aic_bic_table, 'AIC', 'ascend');
end

% 辅助函数：创建详细性能表
function perf_detail_table = create_performance_detail_table(results, methods)
% 创建详细模型性能表，包括均值和标准差
% 输入:
%   results - 结果结构
%   methods - 方法名称
% 输出:
%   perf_detail_table - 详细性能表

% 初始化变量
methods_cell = cell(length(methods), 1);
metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
metric_names = {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'AUC'};

% 初始化数组
mean_values = zeros(length(methods), length(metrics));
std_values = zeros(length(methods), length(metrics));
cv_values = zeros(length(methods), length(metrics));

% 提取每种方法的性能指标
for i = 1:length(methods)
    method = methods{i};
    methods_cell{i} = method;
    
    for j = 1:length(metrics)
        metric = metrics{j};
        
        % 均值
        if isfield(results.(method).performance, ['avg_' metric])
            mean_values(i, j) = results.(method).performance.(['avg_' metric]);
        else
            mean_values(i, j) = NaN;
        end
        
        % 标准差
        if isfield(results.(method).performance, ['std_' metric])
            std_values(i, j) = results.(method).performance.(['std_' metric]);
        else
            std_values(i, j) = NaN;
        end
        
        % 变异系数
        if mean_values(i, j) > 0
            cv_values(i, j) = std_values(i, j) / mean_values(i, j);
        else
            cv_values(i, j) = NaN;
        end
    end
end

% 创建表格
table_vars = {'Method'};
data_vars = {methods_cell};

% 添加各指标的均值、标准差和变异系数
for j = 1:length(metrics)
    table_vars{end+1} = [metric_names{j} '_Mean'];
    data_vars{end+1} = mean_values(:, j);
    
    table_vars{end+1} = [metric_names{j} '_StdDev'];
    data_vars{end+1} = std_values(:, j);
    
    table_vars{end+1} = [metric_names{j} '_CV'];
    data_vars{end+1} = cv_values(:, j);
end

% 创建表格
perf_detail_table = table(data_vars{:}, 'VariableNames', table_vars);

% 按F1分数均值排序
f1_col = find(strcmp(table_vars, 'F1_Score_Mean'));
perf_detail_table = sortrows(perf_detail_table, f1_col, 'descend');
end

% 辅助函数：创建变量频率表 - 修复版
function var_freq_table = create_var_freq_table(results, methods, var_names)
% 创建变量选择频率表
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   var_names - 变量名称
% 输出:
%   var_freq_table - 变量频率表

% 初始化变量名称列
var_names_cell = cell(length(var_names), 1);
for i = 1:length(var_names)
    var_names_cell{i} = var_names{i};
end

% 初始化表格
var_freq_table = table(var_names_cell, 'VariableNames', {'VariableName'});

% 添加各方法的频率
for i = 1:length(methods)
    method = methods{i};
    var_freq = results.(method).var_freq;
    
    % 确保长度一致
    if length(var_freq) ~= length(var_names)
        log_message('warning', sprintf('%s方法的变量频率长度(%d)与变量名长度(%d)不匹配，进行调整', method, length(var_freq), length(var_names)));
        
        % 如果var_freq较短，扩展它
        if length(var_freq) < length(var_names)
            var_freq_extended = zeros(length(var_names), 1);
            var_freq_extended(1:length(var_freq)) = var_freq;
            var_freq = var_freq_extended;
        % 如果var_freq较长，截断它
        else
            var_freq = var_freq(1:length(var_names));
        end
    end
    
    % 确保var_freq是列向量
    if size(var_freq, 2) > 1
        var_freq = var_freq';
    end
    
    % 添加到表格中
    var_freq_table.(method) = var_freq;
end

% 添加平均频率
avg_vals = zeros(height(var_freq_table), 1);
for i = 1:length(methods)
    method = methods{i};
    avg_vals = avg_vals + var_freq_table.(method);
end
avg_vals = avg_vals / length(methods);
var_freq_table.Average = avg_vals;

% 对表格进行排序
var_freq_table = sortrows(var_freq_table, 'Average', 'descend');
end

% 辅助函数：创建增强性能表 - 新增
function perf_table = create_enhanced_performance_table(results, methods)
% 创建增强模型性能表，包含更多评估指标
% 输入:
%   results - 结果结构
%   methods - 方法名称
% 输出:
%   perf_table - 性能表

% 确保methods是列向量cell数组
if ~iscell(methods)
    methods = cellstr(methods);
end
methods_cell = cell(length(methods), 1);
for i = 1:length(methods)
    methods_cell{i} = methods{i};
end

% 初始化表格
perf_table = table(methods_cell, 'VariableNames', {'Method'});

% 添加各性能指标（增强版）
metrics = {'avg_accuracy', 'avg_sensitivity', 'avg_specificity', 'avg_precision', 'avg_f1_score', 'avg_auc'};
metric_names = {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC'};

for i = 1:length(metrics)
    metric = metrics{i};
    metric_name = metric_names{i};
    
    values = zeros(length(methods), 1);
    for j = 1:length(methods)
        values(j) = results.(methods{j}).performance.(metric);
    end
    
    perf_table.(metric_name) = values;
end

% 对表格进行排序
perf_table = sortrows(perf_table, 'F1_Score', 'descend');
end

% 子函数1：雷达图
function create_performance_radar_chart(results, methods, figure_dir)
% 创建性能指标雷达图
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   figure_dir - 图形保存目录

% 定义性能指标
metrics = {'avg_accuracy', 'avg_precision', 'avg_sensitivity', 'avg_specificity', 'avg_f1_score', 'avg_auc'};
metric_labels = {'准确率', '精确率', '敏感性', '特异性', 'F1分数', 'AUC'};
n_metrics = length(metrics);

% 收集性能数据
perf_data = zeros(length(methods), n_metrics);
for i = 1:length(methods)
    method = methods{i};
    
    if isfield(results, method) && isfield(results.(method), 'performance')
        for j = 1:n_metrics
            metric = metrics{j};
            if isfield(results.(method).performance, metric)
                perf_data(i, j) = results.(method).performance.(metric);
            else
                perf_data(i, j) = 0;
            end
        end
    end
end

% 创建图形
fig = figure('Name', 'Performance Radar Chart', 'Position', [100, 100, 800, 800]);

% 使用传统方法绘制雷达图，而不是使用极坐标轴
% 计算角度
theta = linspace(0, 2*pi, n_metrics+1);
theta = theta(1:end-1); % 移除最后一个点，使其不重复

% 绘制雷达图
colors = lines(length(methods));
line_styles = {'-', '--', ':', '-.', '-', '--', ':', '-.'};
hold on;

% 绘制背景网格(环)
levels = [0.2, 0.4, 0.6, 0.8, 1.0];
for l = 1:length(levels)
    level = levels(l);
    x_grid = level * cos(linspace(0, 2*pi, 100));
    y_grid = level * sin(linspace(0, 2*pi, 100));
    plot(x_grid, y_grid, 'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
    
    % 添加标签
    text(0, level+0.02, sprintf('%.1f', level), 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.5, 0.5, 0.5]);
end

% 绘制轴线
for i = 1:n_metrics
    line([0, cos(theta(i))], [0, sin(theta(i))], 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1);
    
    % 添加轴标签
    label_dist = 1.2; % 标签距离
    text(label_dist*cos(theta(i)), label_dist*sin(theta(i)), metric_labels{i}, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold');
end

% 绘制每个方法的性能
legend_handles = [];
for i = 1:length(methods)
    % 准备数据点
    r = perf_data(i, :);
    
    % 计算坐标
    x = r .* cos(theta);
    y = r .* sin(theta);
    
    % 闭合多边形
    x = [x, x(1)];
    y = [y, y(1)];
    
    % 绘制线
    line_style = line_styles{mod(i-1, length(line_styles))+1};
    h_line = plot(x, y, line_style, 'Color', colors(i,:), 'LineWidth', 2);
    
    % 绘制点
    for j = 1:n_metrics
        plot(r(j)*cos(theta(j)), r(j)*sin(theta(j)), 'o', 'MarkerSize', 6, ...
            'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'none');
    end
    
    % 添加到图例
    legend_handles = [legend_handles, h_line];
    
    % 计算多边形顶点
    vertices_x = r .* cos(theta);
    vertices_y = r .* sin(theta);
    
    % 绘制填充区域
    pgon = polyshape(vertices_x, vertices_y);
    h_fill = fill(pgon.Vertices(:,1), pgon.Vertices(:,2), colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% 设置坐标轴
axis equal;
axis([-1.3, 1.3, -1.3, 1.3]);
axis off;

% 添加标题
title('性能指标雷达图', 'FontSize', 14, 'FontWeight', 'bold');

% 添加图例
legend(legend_handles, methods, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 10);

% 保存图形
save_figure(fig, figure_dir, 'performance_radar_chart', 'Formats', {'svg'});
close(fig);
end

% 子函数2：性能热图
function create_performance_heatmap(results, methods, figure_dir)
    % 创建性能指标热图
    
    % 准备数据
    metrics = {'avg_accuracy', 'avg_sensitivity', 'avg_specificity', 'avg_precision', 'avg_f1_score', 'avg_auc'};
    metric_labels = {'准确率', '敏感性', '特异性', '精确率', 'F1分数', 'AUC'};
    
    % 构建数据矩阵
    data_matrix = zeros(length(methods), length(metrics));
    for i = 1:length(methods)
        for j = 1:length(metrics)
            data_matrix(i, j) = results.(methods{i}).performance.(metrics{j});
        end
    end
    
    % 创建热图
    fig = figure('Name', 'Performance Heatmap', 'Position', [100, 100, 1000, 600]);
    
    % 绘制热图
    h = heatmap(metric_labels, methods, data_matrix);
    h.Title = '各方法性能指标热图';
    h.XLabel = '性能指标';
    h.YLabel = '方法';
    h.ColorbarVisible = 'on';
    h.FontSize = 12;
    
    % 设置颜色映射
    colormap(jet);
    h.ColorLimits = [0, 1];
    
    % 格式化数值显示
    h.CellLabelFormat = '%.3f';
    
    % 保存图形
    save_figure(fig, figure_dir, 'performance_heatmap', 'Formats', {'svg'});
    close(fig);
end

% 子函数3：条形图对比
function create_performance_barplot_comparison(results, methods, figure_dir)
    % 创建性能指标条形图对比
    
    % 准备数据
    metrics = {'avg_accuracy', 'avg_sensitivity', 'avg_specificity', 'avg_precision', 'avg_f1_score', 'avg_auc'};
    metric_labels = {'准确率', '敏感性', '特异性', '精确率', 'F1分数', 'AUC'};
    
    % 构建数据矩阵
    data_matrix = zeros(length(metrics), length(methods));
    error_matrix = zeros(length(metrics), length(methods));
    
    for i = 1:length(metrics)
        for j = 1:length(methods)
            data_matrix(i, j) = results.(methods{j}).performance.(metrics{i});
            % 获取标准差
            std_metric = strrep(metrics{i}, 'avg_', 'std_');
            if isfield(results.(methods{j}).performance, std_metric)
                error_matrix(i, j) = results.(methods{j}).performance.(std_metric);
            end
        end
    end
    
    % 创建图形
    fig = figure('Name', 'Performance Barplot Comparison', 'Position', [100, 100, 1200, 700]);
    
    % 创建分组条形图
    bar_handle = bar(data_matrix);
    hold on;
    
    % 添加误差线
    num_groups = size(data_matrix, 1);
    num_bars = size(data_matrix, 2);
    group_width = min(0.8, num_bars / (num_bars + 1.5));
    
    for i = 1:num_bars
        x = (1:num_groups) + (i - (num_bars + 1) / 2) * group_width / num_bars;
        errorbar(x, data_matrix(:, i), error_matrix(:, i), 'k', 'LineStyle', 'none');
    end
    
    % 设置图形属性
    set(gca, 'XTick', 1:length(metric_labels), 'XTickLabel', metric_labels, 'XTickLabelRotation', 45);
    ylabel('性能值', 'FontSize', 12);
    title('不同方法的性能指标对比', 'FontSize', 16, 'FontWeight', 'bold');
    ylim([0, 1]);
    grid on;
    
    % 添加图例
    legend(methods, 'Location', 'northeast');
    
    % 保存图形
    save_figure(fig, figure_dir, 'performance_barplot_comparison', 'Formats', {'svg'});
    close(fig);
end

% 子函数4：性能分布图
function create_performance_distribution_plot(results, methods, figure_dir)
    % 创建性能指标分布图
    
    metrics = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc'};
    metric_labels = {'准确率', '敏感性', '特异性', '精确率', 'F1分数', 'AUC'};
    
    for m = 1:length(metrics)
        metric = metrics{m};
        
        % 收集所有方法在该指标上的分布数据
        all_data = [];
        group_labels = [];
        
        for i = 1:length(methods)
            method = methods{i};
            if isfield(results.(method).performance, metric)
                data = results.(method).performance.(metric);
                all_data = [all_data; data];
                group_labels = [group_labels; repmat({method}, length(data), 1)];
            end
        end
        
        if ~isempty(all_data)
            % 创建单独的图形
            fig = figure('Name', sprintf('Distribution - %s', metric_labels{m}), 'Position', [100, 100, 800, 600]);
            
            % 创建箱线图
            boxplot(all_data, group_labels, 'Notch', 'on');
            
            % 设置图形属性
            title(sprintf('%s分布对比', metric_labels{m}), 'FontSize', 16, 'FontWeight', 'bold');
            ylabel(metric_labels{m}, 'FontSize', 12);
            xlabel('方法', 'FontSize', 12);
            grid on;
            
            % 添加均值点
            hold on;
            method_unique = unique(group_labels);
            x_pos = 1:length(method_unique);
            mean_values = zeros(length(method_unique), 1);
            
            for i = 1:length(method_unique)
                idx = strcmp(group_labels, method_unique{i});
                mean_values(i) = mean(all_data(idx));
            end
            
            scatter(x_pos, mean_values, 100, 'r', 'filled', 'Marker', 'd');
            
            % 添加总体均值线
            overall_mean = mean(all_data);
            line([0.5, length(method_unique)+0.5], [overall_mean, overall_mean], ...
                'LineStyle', '--', 'Color', 'k', 'LineWidth', 1.5);
            text(length(method_unique)+0.5, overall_mean, sprintf(' 均值: %.3f', overall_mean), ...
                'VerticalAlignment', 'middle', 'FontSize', 10);
            
            % 保存图形
            save_figure(fig, figure_dir, sprintf('distribution_%s', metric), 'Formats', {'svg'});
            close(fig);
        end
    end
end

% 子函数5：综合性能评分图
function create_comprehensive_performance_plot(results, methods, figure_dir)
    % 创建综合性能评分图
    
    % 定义权重（可根据需要调整）
    weights = struct();
    weights.accuracy = 0.15;
    weights.sensitivity = 0.2;  % 召回率通常更重要
    weights.specificity = 0.15;
    weights.precision = 0.2;
    weights.f1_score = 0.2;
    weights.auc = 0.1;
    
    % 计算综合评分
    comprehensive_scores = zeros(length(methods), 1);
    metric_contributions = zeros(length(methods), 6);
    
    metrics = {'avg_accuracy', 'avg_sensitivity', 'avg_specificity', 'avg_precision', 'avg_f1_score', 'avg_auc'};
    metric_labels = {'准确率', '敏感性', '特异性', '精确率', 'F1分数', 'AUC'};
    weight_values = [weights.accuracy, weights.sensitivity, weights.specificity, ...
                     weights.precision, weights.f1_score, weights.auc];
    
    for i = 1:length(methods)
        score = 0;
        for j = 1:length(metrics)
            value = results.(methods{i}).performance.(metrics{j});
            contribution = value * weight_values(j);
            metric_contributions(i, j) = contribution;
            score = score + contribution;
        end
        comprehensive_scores(i) = score;
    end
    
    % 1. 创建综合评分条形图
    fig1 = figure('Name', 'Comprehensive Performance Score', 'Position', [100, 100, 900, 600]);
    
    [sorted_scores, sort_idx] = sort(comprehensive_scores, 'descend');
    sorted_methods = methods(sort_idx);
    
    % 创建水平条形图
    barh(sorted_scores, 'FaceColor', [0.3, 0.6, 0.9]);
    
    % 设置图形属性
    set(gca, 'YTick', 1:length(sorted_methods), 'YTickLabel', sorted_methods);
    xlabel('综合性能评分', 'FontSize', 12);
    title('方法综合性能评分（加权）', 'FontSize', 16, 'FontWeight', 'bold');
    grid on;
    
    % 添加分数标签
    for i = 1:length(sorted_scores)
        text(sorted_scores(i) + 0.002, i, sprintf('%.3f', sorted_scores(i)), ...
            'VerticalAlignment', 'middle', 'FontSize', 10);
    end
    
    % 保存图形
    save_figure(fig1, figure_dir, 'comprehensive_performance_score', 'Formats', {'svg'});
    close(fig1);
    
    % 2. 创建堆叠贡献图
    fig2 = figure('Name', 'Performance Contribution', 'Position', [100, 100, 1200, 700]);
    
    % 重新排序贡献矩阵
    sorted_contributions = metric_contributions(sort_idx, :);
    
    % 创建堆叠条形图
    bar_h = bar(sorted_contributions, 'stacked');
    
    % 设置图形属性
    set(gca, 'XTick', 1:length(sorted_methods), 'XTickLabel', sorted_methods, 'XTickLabelRotation', 45);
    ylabel('贡献值', 'FontSize', 12);
    title('各指标对综合评分的贡献', 'FontSize', 16, 'FontWeight', 'bold');
    grid on;
    
    % 添加图例
    legend(metric_labels, 'Location', 'eastoutside');
    
    % 设置不同的颜色
    colormap(lines(6));
    
    % 保存图形
    save_figure(fig2, figure_dir, 'performance_contribution', 'Formats', {'svg'});
    close(fig2);
    
    % 3. 创建权重分布饼图
    fig3 = figure('Name', 'Weight Distribution', 'Position', [100, 100, 800, 800]);
    
    % 创建饼图
    pie(weight_values, metric_labels);
    title('性能指标权重分布', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存图形
    save_figure(fig3, figure_dir, 'weight_distribution', 'Formats', {'svg'});
    close(fig3);
end

% 辅助函数：创建交叉验证结果表 - 新增
function cv_table = create_cv_results_table(cv_results)
% 创建K折交叉验证结果表
% 输入:
%   cv_results - 交叉验证结果
% 输出:
%   cv_table - 交叉验证结果表

% 提取每个折的性能
n_folds = length(cv_results.accuracy);
fold_numbers = (1:n_folds)';

% 初始化表格
cv_table = table(fold_numbers, 'VariableNames', {'Fold'});

% 添加各性能指标
cv_table.Accuracy = cv_results.accuracy;
cv_table.Precision = cv_results.precision;
cv_table.Recall = cv_results.recall;
cv_table.Specificity = cv_results.specificity;
cv_table.F1_Score = cv_results.f1_score;
cv_table.AUC = cv_results.auc;

% 添加均值和标准差行
mean_row = table('Size', [1 size(cv_table,2)], 'VariableTypes', repmat({'double'}, 1, size(cv_table,2)), 'VariableNames', cv_table.Properties.VariableNames);
std_row = mean_row;

mean_row.Fold = 0; % 用0表示均值行
std_row.Fold = -1; % 用-1表示标准差行

% 填充数据
metrics = {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'AUC'};
for i = 1:length(metrics)
    metric = metrics{i};
    mean_row.(metric) = mean(cv_table.(metric), 'omitnan');
    std_row.(metric) = std(cv_table.(metric), 'omitnan');
end

% 合并表格
cv_table = [cv_table; mean_row; std_row];

end

%% 校准曲线图函数
function create_calibration_curves(results, methods, figure_dir)
% 创建校准曲线图
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   figure_dir - 图形保存目录

fig = figure('Name', 'Calibration Curves', 'Position', [100, 100, 1000, 800]);

colors = lines(length(methods));
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
hold on;

legend_entries = cell(length(methods), 1);

% 绘制理想校准曲线
plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);

n_bins = 10;  % 分箱数量
bin_edges = linspace(0, 1, n_bins+1);
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;

for i = 1:length(methods)
    method = methods{i};
    
    % 检查是否存在预测概率
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
            % 计算校准曲线
            [fraction_of_positives, mean_predicted_value] = calibration_curve(all_labels, all_probs, n_bins);
            
            % 绘制校准曲线
            color_idx = mod(i-1, size(colors, 1)) + 1;
            marker_idx = mod(i-1, length(markers)) + 1;
            
            plot(mean_predicted_value, fraction_of_positives, ['-' markers{marker_idx}], ...
                'Color', colors(color_idx,:), 'LineWidth', 2, 'MarkerSize', 8, ...
                'MarkerFaceColor', colors(color_idx,:));
            
            % 计算Brier分数（校准误差）
            brier_score = mean((all_probs - all_labels).^2);
            
            legend_entries{i} = sprintf('%s (Brier=%.3f)', method, brier_score);
        else
            legend_entries{i} = method;
        end
    else
        % 使用单点性能指标
        y_pred = results.(method).performance.avg_sensitivity;
        y_true = results.(method).performance.avg_precision;
        
        color_idx = mod(i-1, size(colors, 1)) + 1;
        marker_idx = mod(i-1, length(markers)) + 1;
        
        plot(y_pred, y_true, markers{marker_idx}, 'Color', colors(color_idx,:), ...
            'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
        
        legend_entries{i} = sprintf('%s (单点)', method);
    end
end

% 设置图形属性
xlim([0, 1]);
ylim([0, 1]);
xlabel('预测概率', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('实际阳性比例', 'FontSize', 12, 'FontWeight', 'bold');
title('不同变量选择方法的校准曲线比较', 'FontSize', 14, 'FontWeight', 'bold');
legend([legend_entries; {'完美校准'}], 'Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);
box on;

% 保存图形
save_figure(fig, figure_dir, 'calibration_curves', 'Formats', {'svg'});
close(fig);
end

% 辅助函数：计算校准曲线
function [fraction_of_positives, mean_predicted_value] = calibration_curve(y_true, y_prob, n_bins)
% 计算校准曲线
% 输入:
%   y_true - 真实标签
%   y_prob - 预测概率
%   n_bins - 分箱数量
% 输出:
%   fraction_of_positives - 每个分箱中的实际阳性比例
%   mean_predicted_value - 每个分箱的平均预测概率

% 计算分箱边界
bin_edges = linspace(0, 1, n_bins+1);

% 初始化结果数组
fraction_of_positives = zeros(n_bins, 1);
mean_predicted_value = zeros(n_bins, 1);

% 对每个分箱计算
for i = 1:n_bins
    % 找出落入当前分箱的样本
    bin_mask = (y_prob >= bin_edges(i)) & (y_prob < bin_edges(i+1));
    
    % 如果分箱为空，使用默认值
    if sum(bin_mask) == 0
        fraction_of_positives(i) = 0;
        mean_predicted_value(i) = (bin_edges(i) + bin_edges(i+1)) / 2;
    else
        % 计算实际阳性比例
        fraction_of_positives(i) = mean(y_true(bin_mask));
        
        % 计算平均预测概率
        mean_predicted_value(i) = mean(y_prob(bin_mask));
    end
end
end

%% 创建变量组合性能表 - 修复版
function var_group_table = create_variable_group_table(results, methods)
% 创建变量组合性能表
% 输入:
%   results - 结果结构
%   methods - 方法名称
% 输出:
%   var_group_table - 变量组合性能表

% 初始化表格行
rows = [];

for i = 1:length(methods)
    method = methods{i};
    group_perf = results.(method).group_performance;
    
    % 对于每个变量组合创建一行
    for j = 1:length(group_perf)
        combo = group_perf(j);
        var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
        row = {method, var_str, combo.count, combo.accuracy, combo.sensitivity, combo.specificity, combo.precision, combo.f1_score, combo.auc};
        rows = [rows; row];
    end
end

% 创建表格 - 将"Variables"改为"VarCombination"并增加新指标
var_group_table = cell2table(rows, 'VariableNames', {'Method', 'VarCombination', 'Count', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC'});

% 对表格进行排序（先按方法，再按F1分数）
var_group_table = sortrows(var_group_table, {'Method', 'F1_Score'}, {'ascend', 'descend'});
end

%% 创建参数表
function param_table = create_parameter_table(results, methods)
% 创建模型参数表
% 输入:
%   results - 结果结构
%   methods - 方法名称
% 输出:
%   param_table - 参数表

% 初始化表格行
rows = [];

for i = 1:length(methods)
    method = methods{i};
    params = results.(method).params;
    
    % 对于每个模型
    for j = 1:length(params.coef_cell)
        coef = params.coef_cell{j};
        pval = params.pval_cell{j};
        vars = params.var_cell{j};
        
        % 对于每个变量
        for k = 1:length(coef)
            if k <= length(vars)
                var_name = vars{k};
                row = {method, j, var_name, coef(k), pval(k)};
                rows = [rows; row];
            end
        end
    end
end

% 创建表格
param_table = cell2table(rows, 'VariableNames', {'Method', 'Model_Index', 'Variable', 'Coefficient', 'P_Value'});
end

%% 创建K折交叉验证性能图 - 新增
function create_cv_performance_plot(cv_results, figure_dir)
% 创建K折交叉验证性能图
% 输入:
%   cv_results - 交叉验证结果
%   figure_dir - 图形保存目录


% 创建图形
fig = figure('Name', 'Cross-Validation Performance', 'Position', [100, 100, 1200, 800]);

% 获取折数
k = length(cv_results.accuracy);

% 准备数据
metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
metric_labels = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
metric_colors = lines(length(metrics));

% 创建子图1：各折性能
subplot(2, 2, 1);
hold on;

% 为每个指标绘制折线
for i = 1:length(metrics)
    metric = metrics{i};
    values = cv_results.(metric);
    
    % 绘制折线
    plot(1:k, values, 'o-', 'LineWidth', 1.5, 'Color', metric_colors(i,:), 'DisplayName', metric_labels{i});
end

% 设置图形属性
xlabel('折数', 'FontSize', 12);
ylabel('性能值', 'FontSize', 12);
title('K折交叉验证中各折的性能表现', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Location', 'best');
xlim([0.5, k+0.5]);
ylim([0, 1.05]);
set(gca, 'XTick', 1:k);

% 创建子图2：各指标均值和标准差
subplot(2, 2, 2);

% 计算均值和标准差
metric_means = zeros(length(metrics), 1);
metric_stds = zeros(length(metrics), 1);

for i = 1:length(metrics)
    metric = metrics{i};
    metric_means(i) = mean(cv_results.(metric), 'omitnan');
    metric_stds(i) = std(cv_results.(metric), 'omitnan');
end

% 创建条形图
bar_h = bar(metric_means);
set(bar_h, 'FaceColor', 'flat');
for i = 1:length(metrics)
    bar_h.CData(i,:) = metric_colors(i,:);
end

% 添加误差线
hold on;
errorbar(1:length(metrics), metric_means, metric_stds, '.k');

% 设置图形属性
set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metric_labels);
ylabel('平均性能', 'FontSize', 12);
title('各评估指标的均值和标准差', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0, 1.05]);

% 添加数值标签
for i = 1:length(metrics)
    text(i, metric_means(i) + 0.03, sprintf('%.3f±%.3f', metric_means(i), metric_stds(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% 创建子图3：系数稳定性
subplot(2, 2, 3);

% 提取系数变异系数
coef_cv = cv_results.coef_cv;
var_list = ['Intercept'; cv_results.variables(2:end)]; % 排除截距

% 创建条形图
bar_h = barh(coef_cv);
set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);

% 设置图形属性
set(gca, 'YTick', 1:length(coef_cv), 'YTickLabel', var_list);
xlabel('变异系数 (CV)', 'FontSize', 12);
title('模型系数稳定性 (变异系数)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 创建子图4：学习曲线或ROC曲线
subplot(2, 2, 4);

try
    % 计算平均ROC曲线（如果可能）
    x_points = linspace(0, 1, 100);
    y_points = zeros(length(x_points), 1);
    
    % 绘制ROC曲线
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5); % 对角线
    hold on;
    
    for i = 1:k
        if ~isnan(cv_results.auc(i)) && cv_results.auc(i) > 0.5
            % 绘制简化的ROC曲线
            x0 = 0;
            y0 = 0;
            x1 = 1 - cv_results.specificity(i);
            y1 = cv_results.recall(i);
            x2 = 1;
            y2 = 1;
            
            % 绘制折线
            plot([x0, x1, x2], [y0, y1, y2], '-', 'Color', [0.7, 0.7, 0.7, 0.3], 'LineWidth', 0.5);
            
            % 计算近似曲线下面积
            for j = 1:length(x_points)
                x = x_points(j);
                if x <= x1
                    y = y1 * x / x1;
                else
                    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
                end
                y_points(j) = y_points(j) + y / k;
            end
        end
    end
    
    % 绘制平均ROC曲线
    plot(x_points, y_points, '-', 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2], 'DisplayName', '平均ROC曲线');
    
    % 添加AUC值
    mean_auc = mean(cv_results.auc, 'omitnan');
    text(0.6, 0.2, sprintf('平均AUC = %.3f ± %.3f', mean_auc, std(cv_results.auc, 'omitnan')), ...
        'FontSize', 10, 'FontWeight', 'bold');
    
    % 设置图形属性
    xlabel('1 - 特异性', 'FontSize', 12);
    ylabel('敏感性', 'FontSize', 12);
    title('平均ROC曲线', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend('无信息线', '平均ROC曲线', 'Location', 'southeast');
    
catch ME
    % 如果ROC曲线绘制失败，显示错误消息
    text(0.5, 0.5, '无法生成ROC曲线', 'HorizontalAlignment', 'center', 'FontSize', 12);
    log_message('warning', sprintf('绘制ROC曲线失败: %s', ME.message));
end

% 调整整体布局
sgtitle('K折交叉验证性能分析', 'FontSize', 16, 'FontWeight', 'bold');
set(gcf, 'Color', 'white');

% 保存矢量图
save_figure(fig, figure_dir, 'cv_performance', 'Formats', {'svg'});

% 关闭图形
close(fig);
end

%% 创建系数稳定性图 - 新增
function create_coefficient_stability_plot(coef_stability, methods, figure_dir)
% 创建系数稳定性图
% 输入:
%   coef_stability - 系数稳定性结果
%   methods - 方法名称
%   figure_dir - 图形保存目录

% 对每种支持的方法创建图形
for m = 1:length(methods)
    method = methods{m};
    
    % 检查该方法是否有系数稳定性结果
    if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
        
        % 提取数据
        table_data = coef_stability.(method).table;
        var_list = table_data.Variable;
        coef_mean = table_data.Mean;
        coef_std = table_data.StdDev;
        coef_cv = table_data.CV;
        
        % 创建图形
        fig = figure('Name', sprintf('%s Coefficient Stability', method), 'Position', [100, 100, 1200, 800]);
        
        % 创建子图1：系数均值和标准差
        subplot(2, 1, 1);
        
        % 创建条形图
        bar_h = bar(coef_mean);
        hold on;
        
        % 添加误差线
        errorbar(1:length(coef_mean), coef_mean, coef_std, '.k');
        
        % 设置图形属性
        set(gca, 'XTick', 1:length(var_list), 'XTickLabel', var_list, 'XTickLabelRotation', 45);
        ylabel('系数值', 'FontSize', 12);
        title(sprintf('%s方法的系数均值和标准差', method), 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加零线
        line([0, length(var_list)+1], [0, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
        
        % 创建子图2：系数变异系数
        subplot(2, 1, 2);
        
        % 按变异系数大小排序
        [sorted_cv, idx] = sort(coef_cv, 'descend');
        sorted_vars = var_list(idx);
        
        % 创建条形图
        bar_h = barh(sorted_cv);
        set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);
        
        % 设置图形属性
        set(gca, 'YTick', 1:length(sorted_vars), 'YTickLabel', sorted_vars);
        xlabel('变异系数 (CV)', 'FontSize', 12);
        title(sprintf('%s方法的系数稳定性 (变异系数)', method), 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加阈值线
        line([0.5, 0.5], [0, length(sorted_vars)+1], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1);
        text(0.52, length(sorted_vars)-1, '不稳定阈值 (CV > 0.5)', 'Color', 'r', 'FontSize', 10);
        
        % 调整整体布局
        sgtitle(sprintf('%s方法的系数稳定性分析', method), 'FontSize', 16, 'FontWeight', 'bold');
        set(gcf, 'Color', 'white');
                
        % 保存矢量图
        save_figure(fig, figure_dir, sprintf('%s_coefficient_stability', method), 'Formats', {'svg'});
        
        % 关闭图形
        close(fig);
    end
end

% 创建所有方法的综合比较图
try
    % 收集所有方法的变异系数
    all_methods = {};
    all_vars = {};
    all_cvs = [];
    
    for m = 1:length(methods)
        method = methods{m};
        if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
            table_data = coef_stability.(method).table;
            
            % 只考虑截距项
            for i = 1:height(table_data)
                all_methods{end+1} = method;
                all_vars{end+1} = table_data.Variable{i};
                all_cvs(end+1) = table_data.CV(i);
            end
        end
    end
    
    % 如果有足够的数据，创建比较图
    if length(all_cvs) >= 3
        fig = figure('Name', 'Coefficient Stability Comparison', 'Position', [100, 100, 1200, 600]);
        
        % 创建散点图
        scatter(1:length(all_cvs), all_cvs, 50, 'filled', 'MarkerFaceAlpha', 0.7);
        
        % 添加方法和变量标签
        for i = 1:length(all_cvs)
            text(i, all_cvs(i) + 0.03, sprintf('%s\n%s', all_methods{i}, all_vars{i}), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'Rotation', 45);
        end
        
        % 设置图形属性
        set(gca, 'XTick', []);
        ylabel('变异系数 (CV)', 'FontSize', 12);
        title('各方法系数稳定性比较', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加阈值线
        line([0, length(all_cvs)+1], [0.5, 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1);
        text(length(all_cvs)*0.9, 0.52, '不稳定阈值 (CV > 0.5)', 'Color', 'r', 'FontSize', 10);
        
        % 调整图形
        set(gcf, 'Color', 'white');
                
        % 保存矢量图
        save_figure(fig, figure_dir, 'coefficient_stability_comparison', 'Formats', {'svg'});
        
        % 关闭图形
        close(fig);
    end
catch ME
    log_message('warning', sprintf('创建系数稳定性比较图失败: %s', ME.message));
end
end

%% 创建变量贡献图 - 新增
function create_variable_contribution_plot(var_contribution, figure_dir)
% 创建变量贡献图
% 输入:
%   var_contribution - 变量贡献分析结果
%   figure_dir - 图形保存目录



% 创建综合变量重要性图
if isfield(var_contribution, 'overall_importance')
    try
        % 提取数据
        importance_table = var_contribution.overall_importance;
        vars = importance_table.Variable;
        importance = importance_table.Normalized_Importance;
        
        % 取前10个变量
        top_n = min(10, length(vars));
        top_vars = vars(1:top_n);
        top_importance = importance(1:top_n);
        
        % 创建图形
        fig = figure('Name', 'Overall Variable Importance', 'Position', [100, 100, 1000, 600]);
        
        % 创建条形图
        barh_h = barh(top_importance);
        set(barh_h, 'FaceColor', [0.3, 0.6, 0.8]);
        
        % 设置图形属性
        set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
        xlabel('归一化重要性 (%)', 'FontSize', 12);
        title('综合变量重要性排名 (前10个变量)', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加数值标签
        for i = 1:top_n
            text(top_importance(i) + 0.5, i, sprintf('%.2f%%', top_importance(i)), ...
                'VerticalAlignment', 'middle', 'FontSize', 9);
        end
        
        % 调整图形
        set(gcf, 'Color', 'white');
        
        % 保存矢量图
        save_figure(fig, figure_dir, 'overall_variable_importance', 'Formats', {'svg'});
        
        % 关闭图形
        close(fig);
    catch ME
        log_message('warning', sprintf('创建综合变量重要性图失败: %s', ME.message));
    end
end

% 创建相关性分析图
if isfield(var_contribution, 'correlation')
    try
        % 提取数据
        corr_table = var_contribution.correlation;
        vars = corr_table.Variable;
        corr_values = corr_table.Correlation;
        partial_corr = corr_table.PartialCorr;
        
        % 按偏相关系数绝对值排序
        [~, idx] = sort(abs(partial_corr), 'descend');
        sorted_vars = vars(idx);
        sorted_corr = corr_values(idx);
        sorted_partial = partial_corr(idx);
        
        % 取前10个变量
        top_n = min(10, length(sorted_vars));
        top_vars = sorted_vars(1:top_n);
        top_corr = sorted_corr(1:top_n);
        top_partial = sorted_partial(1:top_n);
        
        % 创建图形
        fig = figure('Name', 'Correlation Analysis', 'Position', [100, 100, 1000, 600]);
        
        % 创建分组条形图
        bar_data = [top_corr, top_partial];
        bar_h = barh(bar_data);
        set(bar_h(1), 'FaceColor', [0.3, 0.6, 0.8]);
        set(bar_h(2), 'FaceColor', [0.8, 0.3, 0.3]);
        
        % 设置图形属性
        set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
        xlabel('相关系数', 'FontSize', 12);
        title('变量相关性分析 (前10个变量)', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        
        % 添加图例
        legend({'普通相关', '偏相关'}, 'Location', 'southeast');
        
        % 添加参考线
        line([0, 0], [0, top_n+1], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
        
        % 调整图形
        set(gcf, 'Color', 'white');
                
        % 保存矢量图
        save_figure(fig, figure_dir, 'correlation_analysis', 'Formats', {'svg'});

        % 关闭图形
        close(fig);
    catch ME
        log_message('warning', sprintf('创建相关性分析图失败: %s', ME.message));
    end
end

% 为各个方法创建变量贡献图
if isfield(var_contribution, 'methods')
    method_names = fieldnames(var_contribution.methods);
    
    for m = 1:length(method_names)
        method = method_names{m};
        
        % 检查该方法是否有贡献表
        if isfield(var_contribution.methods.(method), 'contribution_table')
            try
                % 提取数据
                contrib_table = var_contribution.methods.(method).contribution_table;
                
                % 创建图形
                fig = figure('Name', sprintf('%s Variable Contribution', method), 'Position', [100, 100, 1000, 600]);
                
                % 提取变量和贡献
                vars = contrib_table.Variable;
                rel_contrib = contrib_table.Relative_Contribution;
                
                % 取前10个变量
                top_n = min(10, height(contrib_table));
                top_vars = vars(1:top_n);
                top_contrib = rel_contrib(1:top_n);
                
                % 为不同方法设置不同的表现形式
                if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
                    % 回归类方法显示系数和相对贡献
                    
                    % 提取系数和方向
                    coeffs = contrib_table.Std_Coefficient(1:top_n);
                    directions = contrib_table.Effect_Direction(1:top_n);
                    
                    % 创建颜色映射
                    colors = zeros(top_n, 3);
                    for i = 1:top_n
                        if strcmp(directions{i}, '正向')
                            colors(i,:) = [0.2, 0.6, 0.8]; % 蓝色表示正向影响
                        else
                            colors(i,:) = [0.8, 0.3, 0.3]; % 红色表示负向影响
                        end
                    end
                    
                    % 创建水平条形图
                    bar_h = barh(top_contrib);
                    set(bar_h, 'FaceColor', 'flat');
                    bar_h.CData = colors;
                    
                    % 设置图形属性
                    set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
                    xlabel('相对贡献 (%)', 'FontSize', 12);
                    title(sprintf('%s方法的变量贡献分析 (前%d个变量)', method, top_n), 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                    
                    % 添加方向指示
                    for i = 1:top_n
                        if strcmp(directions{i}, '正向')
                            text(top_contrib(i) + 0.5, i, '(+)', 'VerticalAlignment', 'middle', 'FontSize', 9, 'Color', [0, 0.5, 0]);
                        else
                            text(top_contrib(i) + 0.5, i, '(-)', 'VerticalAlignment', 'middle', 'FontSize', 9, 'Color', [0.8, 0, 0]);
                        end
                    end
                    
                else
                    % 非回归类方法只显示重要性
                    bar_h = barh(top_contrib);
                    set(bar_h, 'FaceColor', [0.3, 0.6, 0.8]);
                    
                    % 设置图形属性
                    set(gca, 'YTick', 1:top_n, 'YTickLabel', top_vars);
                    xlabel('相对贡献 (%)', 'FontSize', 12);
                    title(sprintf('%s方法的变量重要性 (前%d个变量)', method, top_n), 'FontSize', 14, 'FontWeight', 'bold');
                    grid on;
                end
                
                % 添加数值标签
                for i = 1:top_n
                    text(top_contrib(i) + 0.5, i, sprintf('%.2f%%', top_contrib(i)), ...
                        'VerticalAlignment', 'middle', 'FontSize', 9);
                end
                
                % 调整图形
                set(gcf, 'Color', 'white');
                                
                % 保存矢量图
                save_figure(fig, figure_dir, [method, '_variable_contribution'], 'Formats', {'svg'});
                
                % 关闭图形
                close(fig);
            catch ME
                log_message('warning', sprintf('创建%s方法的变量贡献图失败: %s', method, ME.message));
            end
        end
    end
end
end

%% 箱线图可视化函数
function create_boxplot_visualization(results, methods, figure_dir)
    % 创建箱线图可视化
    % 输入:
    %   results - 结果结构
    %   methods - 方法名称
    %   figure_dir - 图形保存目录

    % 定义性能指标名称
    metric_names = {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC'};
    n_metrics = length(metric_names);
    n_methods = length(methods);

    % 初始化数据矩阵
    data = cell(n_methods, n_metrics);
    for i = 1:n_methods
        method = methods{i};
        if isfield(results, method) && isfield(results.(method), 'performance')
            perf = results.(method).performance;
            
            % 检查并填充每个指标的数据
            data{i, 1} = check_data(perf, 'accuracy', method, 'Accuracy');
            data{i, 2} = check_data(perf, 'sensitivity', method, 'Sensitivity');
            data{i, 3} = check_data(perf, 'specificity', method, 'Specificity');
            data{i, 4} = check_data(perf, 'precision', method, 'Precision');
            data{i, 5} = check_data(perf, 'f1_score', method, 'F1_Score');
            data{i, 6} = check_data(perf, 'auc', method, 'AUC');
        else
            % 如果方法或性能数据缺失，填充 NaN
            log_message('warning', sprintf('Method %s is missing in results or performance data', method));
            for j = 1:n_metrics
                data{i, j} = NaN;
            end
        end
    end

    % 为每个指标创建箱线图
    for i = 1:n_metrics
        metric = metric_names{i};
        
        % 提取当前指标的数据
        metric_data_cell = data(:, i);
        
        % 确定最大数据长度并填充 NaN
        max_len = 0;
        for j = 1:n_methods
            if ~isempty(metric_data_cell{j}) && ~all(isnan(metric_data_cell{j}))
                max_len = max(max_len, length(metric_data_cell{j}));
            end
        end
        
        % 如果没有有效数据，跳过
        if max_len == 0
            log_message('warning', sprintf('No valid data available for metric %s', metric));
            continue;
        end
        
        % 创建矩阵，填充 NaN 以对齐维度
        metric_data = NaN(max_len, n_methods);
        for j = 1:n_methods
            current_data = metric_data_cell{j};
            if ~isempty(current_data) && ~all(isnan(current_data))
                len = length(current_data);
                metric_data(1:len, j) = current_data;
            end
        end
        
        % 验证列数与方法数一致
        if size(metric_data, 2) ~= n_methods
            log_message('error', sprintf('metric_data has %d columns, but there are %d methods for %s', ...
                size(metric_data, 2), n_methods, metric));
            continue;
        end
        
        % 创建图形
        fig = figure('Name', sprintf('%s Boxplot', metric), 'Position', [100, 100, 1000, 600]);
        
        % 创建箱线图
        boxplot(metric_data, 'Labels', methods, 'Notch', 'on', 'Symbol', 'r+');
        
        % 设置图形属性
        title(sprintf('%s分布箱线图 - 各方法比较', metric), 'FontSize', 14, 'FontWeight', 'bold');
        ylabel(metric, 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        
        % 添加均值点
        hold on;
        means = nanmean(metric_data);
        scatter(1:n_methods, means, 100, 'filled', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
        
        % 添加均值和标准差标签
        for j = 1:n_methods
            method_data = metric_data(:, j);
            mean_val = nanmean(method_data);
            std_val = nanstd(method_data);
            text(j, max(method_data, [], 'omitnan') + 0.02, sprintf('均值: %.3f\n标准差: %.3f', mean_val, std_val), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
        end
        
        % 添加整体均值线
        overall_mean = nanmean(metric_data(:));
        plot([0.5, n_methods+0.5], [overall_mean, overall_mean], 'k--', 'LineWidth', 1.5);
        text(n_methods+0.5, overall_mean, sprintf(' 总体均值: %.3f', overall_mean), ...
            'VerticalAlignment', 'middle', 'FontSize', 9);
        
        % 调整 Y 轴范围
        ylim_current = ylim;
        ylim([ylim_current(1), ylim_current(2) + 0.1]);
        
        % 保存图形
        save_figure(fig, figure_dir, sprintf('boxplot_%s', lower(metric)), 'Formats', {'svg'});
        close(fig);
    end
end

% 辅助函数：检查和处理数据
function data_out = check_data(perf, field, method, metric_name)
    if isfield(perf, field) && ~isempty(perf.(field))
        data_out = perf.(field);
    else
        log_message('warning', sprintf('Method %s has missing or empty %s data', method, metric_name));
        data_out = NaN;
    end
end

%% 平均Precision-Recall曲线函数
function create_pr_curves(results, methods, figure_dir)
% 创建精确率-召回率曲线图
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   figure_dir - 图形保存目录

fig = figure('Name', 'Precision-Recall Curves', 'Position', [100, 100, 1000, 800]);

colors = lines(length(methods));
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
hold on;

legend_entries = cell(length(methods), 1);

% 创建标准PR曲线点
std_recall = linspace(0, 1, 100)';
avg_precision = zeros(100, length(methods));

for i = 1:length(methods)
    method = methods{i};
    
    % 计算平均精确率-召回率曲线
    all_recall = [];
    all_precision = [];
    
    % 如果存在性能结构中的预测概率，使用它们绘制PR曲线
    if isfield(results.(method).performance, 'y_pred_prob') && ...
       isfield(results.(method).performance, 'y_test')
        
        y_pred_prob = results.(method).performance.y_pred_prob;
        y_test = results.(method).performance.y_test;
        
        % 对每个Bootstrap样本计算PR曲线
        for j = 1:length(y_pred_prob)
            if ~isempty(y_pred_prob{j}) && ~isempty(y_test{j})
                [precision, recall, ~] = precision_recall_curve(y_test{j}, y_pred_prob{j});
                
                % 存储所有召回率和精确率值
                all_recall = [all_recall; recall];
                all_precision = [all_precision; precision];
            end
        end
        
        % 如果收集到足够的点，绘制平均PR曲线
        if ~isempty(all_recall) && ~isempty(all_precision)
            % 对所有样本的召回率进行排序
            [sorted_recall, idx] = sort(all_recall);
            sorted_precision = all_precision(idx);
            
            % 在标准召回率点计算平均精确率
            for k = 1:length(std_recall)
                r = std_recall(k);
                % 找出大于或等于r的最接近点
                idx = find(sorted_recall >= r, 1, 'first');
                if isempty(idx)
                    avg_precision(k, i) = 0;
                else
                    avg_precision(k, i) = sorted_precision(idx);
                end
            end
            
            % 计算平均精确率 (AP)
            ap = trapz(std_recall, avg_precision(:, i));
            
            % 绘制PR曲线
            color_idx = mod(i-1, size(colors, 1)) + 1;
            plot(std_recall, avg_precision(:, i), '-', 'Color', colors(color_idx,:), 'LineWidth', 2);
            
            % 在曲线上标记数据点
            marker_idx = mod(i-1, length(markers)) + 1;
            num_points = 10; % 均匀分布的点数
            point_idx = round(linspace(1, length(std_recall), num_points));
            plot(std_recall(point_idx), avg_precision(point_idx, i), markers{marker_idx}, ...
                'Color', colors(color_idx,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(color_idx,:));
            
            legend_entries{i} = sprintf('%s (AP=%.3f)', method, ap);
        else
            % 如果没有足够的数据点，使用性能指标中的平均值绘制单个点
            recall = results.(method).performance.avg_sensitivity;
            precision = results.(method).performance.avg_precision;
            
            color_idx = mod(i-1, size(colors, 1)) + 1;
            marker_idx = mod(i-1, length(markers)) + 1;
            
            plot(recall, precision, markers{marker_idx}, 'Color', colors(color_idx,:), ...
                'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
            
            legend_entries{i} = sprintf('%s (单点)', method);
        end
    else
        % 使用单点性能指标
        recall = results.(method).performance.avg_sensitivity;
        precision = results.(method).performance.avg_precision;
        
        color_idx = mod(i-1, size(colors, 1)) + 1;
        marker_idx = mod(i-1, length(markers)) + 1;
        
        plot(recall, precision, markers{marker_idx}, 'Color', colors(color_idx,:), ...
            'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
        
        legend_entries{i} = sprintf('%s (单点)', method);
    end
end

% 添加随机分类器的基准线
random_precision = sum([results.(methods{1}).performance.avg_sensitivity]) / length(methods);
plot([0, 1], [random_precision, random_precision], 'k--', 'LineWidth', 1.5);
legend_entries{end+1} = sprintf('随机 (Precision=%.3f)', random_precision);

% 设置图形属性
xlim([0, 1]);
ylim([0, 1]);
xlabel('召回率 (Recall)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('精确率 (Precision)', 'FontSize', 12, 'FontWeight', 'bold');
title('不同变量选择方法的精确率-召回率曲线比较', 'FontSize', 14, 'FontWeight', 'bold');
legend(legend_entries, 'Location', 'southwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);
box on;

% 保存图形
save_figure(fig, figure_dir, 'precision_recall_curves', 'Formats', {'svg'});
close(fig);
end

% 辅助函数：计算精确率-召回率曲线
function [precision, recall, thresholds] = precision_recall_curve(y_true, y_score)
% 计算精确率-召回率曲线
% 输入:
%   y_true - 真实标签
%   y_score - 预测分数或概率
% 输出:
%   precision - 精确率
%   recall - 召回率
%   thresholds - 阈值

% 获取唯一的阈值，以降序排列
thresholds = sort(unique(y_score), 'descend');

% 添加0作为最后一个阈值
thresholds = [thresholds; -Inf];

n_thresholds = length(thresholds);
precision = zeros(n_thresholds, 1);
recall = zeros(n_thresholds, 1);

for i = 1:n_thresholds
    threshold = thresholds(i);
    
    % 在当前阈值下的预测
    y_pred = y_score >= threshold;
    
    % 计算混淆矩阵元素
    TP = sum(y_pred == 1 & y_true == 1);
    FP = sum(y_pred == 1 & y_true == 0);
    FN = sum(y_pred == 0 & y_true == 1);
    
    % 计算精确率和召回率
    if TP + FP == 0
        precision(i) = 1;  % 如果没有阳性预测，精确率为1
    else
        precision(i) = TP / (TP + FP);
    end
    
    if TP + FN == 0
        recall(i) = 0;  % 如果没有真阳性，召回率为0
    else
        recall(i) = TP / (TP + FN);
    end
end

% 确保精确率-召回率曲线是单调递减的
for i = n_thresholds-1:-1:1
    precision(i) = max(precision(i), precision(i+1));
end
end

%% 混淆矩阵可视化函数
function create_confusion_matrices(results, methods, figure_dir)
    % 创建混淆矩阵可视化
    % 输入:
    %   results - 结果结构
    %   methods - 方法名称
    %   figure_dir - 图形保存目录

    % 为每种方法创建混淆矩阵
    for i = 1:length(methods)
        method = methods{i};
        
        % 获取性能指标
        performance = results.(method).performance;
        
        % 计算平均混淆矩阵
        if isfield(performance, 'y_pred') && isfield(performance, 'y_test')
            y_pred = performance.y_pred;
            y_test = performance.y_test;
            
            % 初始化混淆矩阵
            conf_matrix = zeros(2, 2);
            count = 0;
            
            % 合并所有Bootstrap样本的混淆矩阵
            for j = 1:length(y_pred)
                if ~isempty(y_pred{j}) && ~isempty(y_test{j})
                    % 计算当前样本的混淆矩阵
                    pred = y_pred{j};
                    test = y_test{j};
                    
                    TP = sum(pred == 1 & test == 1);
                    FP = sum(pred == 1 & test == 0);
                    FN = sum(pred == 0 & test == 1);
                    TN = sum(pred == 0 & test == 0);
                    
                    conf_matrix = conf_matrix + [TN, FP; FN, TP];
                    count = count + 1;
                end
            end
            
            if count > 0
                conf_matrix = conf_matrix / count;
            end
        else
            % 使用性能指标估算混淆矩阵
            sensitivity = performance.avg_sensitivity;
            specificity = performance.avg_specificity;
            precision = performance.avg_precision;
            
            % 假设测试集中正负样本比例为1:1
            TP = sensitivity * 50;
            FN = 50 - TP;
            FP = TP / precision - TP;
            TN = specificity * 50;
            
            conf_matrix = [TN, FP; FN, TP];
        end
        
        % 计算归一化混淆矩阵（按行归一化）
        conf_matrix_norm = zeros(2, 2);
        for j = 1:2
            if sum(conf_matrix(j, :)) > 0
                conf_matrix_norm(j, :) = conf_matrix(j, :) / sum(conf_matrix(j, :));
            end
        end
        
        % 创建混淆矩阵图
        fig = figure('Name', sprintf('%s Confusion Matrix', method), 'Position', [100, 100, 800, 600]);
        
        % 绘制混淆矩阵热图
        subplot(1, 2, 1);
        h = heatmap(conf_matrix, 'XLabel', '预测', 'YLabel', '实际', ...
            'XDisplayLabels', {'负类 (0)', '正类 (1)'}, 'YDisplayLabels', {'负类 (0)', '正类 (1)'});
        % 修改：直接在热图对象上设置标题
        h.Title = sprintf('%s: 原始混淆矩阵', method);
        h.FontSize = 12;
        colormap(jet);
        
        % 绘制归一化混淆矩阵热图
        subplot(1, 2, 2);
        h_norm = heatmap(conf_matrix_norm, 'XLabel', '预测', 'YLabel', '实际', ...
            'XDisplayLabels', {'负类 (0)', '正类 (1)'}, 'YDisplayLabels', {'负类 (0)', '正类 (1)'});
        % 修改：直接在热图对象上设置标题
        h_norm.Title = sprintf('%s: 归一化混淆矩阵', method);
        h_norm.FontSize = 12;
        colormap(jet);
        
        % 计算性能指标
        TN = conf_matrix(1, 1);
        FP = conf_matrix(1, 2);
        FN = conf_matrix(2, 1);
        TP = conf_matrix(2, 2);
        
        accuracy = (TP + TN) / sum(conf_matrix(:));
        sensitivity = TP / (TP + FN);
        specificity = TN / (TN + FP);
        precision = TP / (TP + FP);
        f1_score = 2 * precision * sensitivity / (precision + sensitivity);
        
        % 创建标题包含性能指标
        sgtitle(sprintf('%s 混淆矩阵分析\n准确率=%.3f, 灵敏度=%.3f, 特异性=%.3f, 精确率=%.3f, F1=%.3f', ...
            method, accuracy, sensitivity, specificity, precision, f1_score), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存图形
        save_figure(fig, figure_dir, sprintf('%s_confusion_matrix', method), 'Formats', {'svg'});
        close(fig);
    end

    % 创建所有方法的混淆矩阵比较图
    try
        % 计算每种方法的归一化混淆矩阵
        all_conf_matrices = cell(length(methods), 1);
        all_performance = zeros(length(methods), 4); % 准确率、灵敏度、特异性、精确率
        
        for i = 1:length(methods)
            method = methods{i};
            performance = results.(method).performance;
            
            if isfield(performance, 'y_pred') && isfield(performance, 'y_test')
                y_pred = performance.y_pred;
                y_test = performance.y_test;
                
                % 初始化混淆矩阵
                conf_matrix = zeros(2, 2);
                count = 0;
                
                % 合并所有Bootstrap样本的混淆矩阵
                for j = 1:length(y_pred)
                    if ~isempty(y_pred{j}) && ~isempty(y_test{j})
                        % 计算当前样本的混淆矩阵
                        pred = y_pred{j};
                        test = y_test{j};
                        
                        TP = sum(pred == 1 & test == 1);
                        FP = sum(pred == 1 & test == 0);
                        FN = sum(pred == 0 & test == 1);
                        TN = sum(pred == 0 & test == 0);
                        
                        conf_matrix = conf_matrix + [TN, FP; FN, TP];
                        count = count + 1;
                    end
                end
                
                % 计算平均混淆矩阵
                if count > 0
                    conf_matrix = conf_matrix / count;
                end
            else
                % 使用性能指标估算混淆矩阵
                sensitivity = performance.avg_sensitivity;
                specificity = performance.avg_specificity;
                precision = performance.avg_precision;
                
                % 假设测试集中正负样本比例为1:1
                TP = sensitivity * 50;
                FN = 50 - TP;
                FP = TP / precision - TP;
                TN = specificity * 50;
                
                conf_matrix = [TN, FP; FN, TP];
            end
            
            % 计算归一化混淆矩阵（按行归一化）
            conf_matrix_norm = zeros(2, 2);
            for j = 1:2
                if sum(conf_matrix(j, :)) > 0
                    conf_matrix_norm(j, :) = conf_matrix(j, :) / sum(conf_matrix(j, :));
                end
            end
            
            all_conf_matrices{i} = conf_matrix_norm;
            
            % 计算性能指标
            TN = conf_matrix(1, 1);
            FP = conf_matrix(1, 2);
            FN = conf_matrix(2, 1);
            TP = conf_matrix(2, 2);
            
            accuracy = (TP + TN) / sum(conf_matrix(:));
            sensitivity = TP / (TP + FN);
            specificity = TN / (TN + FP);
            precision = TP / (TP + FP);
            
            all_performance(i, :) = [accuracy, sensitivity, specificity, precision];
        end
        
        % 创建比较图
        fig = figure('Name', 'Confusion Matrix Comparison', 'Position', [100, 100, 1200, 900]);
        
        n_methods = length(methods);
        rows = ceil(sqrt(n_methods));
        cols = ceil(n_methods / rows);
        
        for i = 1:n_methods
            subplot(rows, cols, i);
            h = heatmap(all_conf_matrices{i}, 'XLabel', '预测', 'YLabel', '实际', ...
                'XDisplayLabels', {'0', '1'}, 'YDisplayLabels', {'0', '1'});
            % 修改：直接在热图对象上设置标题
            h.Title = sprintf('%s\n准确率=%.3f', methods{i}, all_performance(i, 1));
            h.FontSize = 9;
            colormap(jet);
        end
        
        % 添加整体标题
        sgtitle('各方法混淆矩阵比较 (归一化)', 'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存图形
        save_figure(fig, figure_dir, 'confusion_matrix_comparison', 'Formats', {'svg'});
        close(fig);
    catch ME
        log_message('warning', sprintf('创建混淆矩阵比较图失败: %s', ME.message));
    end
end

%% 修改1：修复图形保存函数 - 添加'-bestfit'选项
function create_roc_curves(results, methods, figure_dir)
    % 创建ROC曲线图
    % 输入:
    %   results - 结果结构
    %   methods - 方法名称
    %   figure_dir - 图形保存目录
    fig = figure('Name', 'ROC Curves', 'Position', [100, 100, 1000, 800]);
    
    % 禁用工具栏
    set(gcf, 'Toolbar', 'none');

    colors = lines(length(methods));
    markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
    hold on;

    legend_entries = cell(length(methods), 1);

    for i = 1:length(methods)
        method = methods{i};
        auc = results.(method).performance.avg_auc;
        sensitivity = results.(method).performance.avg_sensitivity;
        specificity = results.(method).performance.avg_specificity;
        precision = results.(method).performance.avg_precision;
        f1_score = results.(method).performance.avg_f1_score;

        fpr = 1 - specificity;
        color_idx = mod(i-1, size(colors, 1)) + 1;
        marker_idx = mod(i-1, length(markers)) + 1;

        plot(fpr, sensitivity, [markers{marker_idx}], 'Color', colors(color_idx,:), ...
            'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors(color_idx,:));
        plot([0, fpr, 1], [0, sensitivity, 1], '-', 'Color', colors(color_idx,:), 'LineWidth', 1.5);

        legend_entries{i} = sprintf('%s (AUC=%.3f, F1=%.3f)', method, auc, f1_score);
    end

    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);

    xlim([0, 1]);
    ylim([0, 1]);
    xlabel('假阳性率 (1 - 特异性)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('真阳性率 (敏感性)', 'FontSize', 12, 'FontWeight', 'bold');
    title('不同变量选择方法的ROC曲线比较', 'FontSize', 14, 'FontWeight', 'bold');
    legend([legend_entries; {'随机猜测'}], 'Location', 'southeast', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 11);
    box on;

    text(0.05, 0.95, '注: 点表示在测试集上的平均性能', 'FontSize', 9);

    set(gcf, 'Color', 'white');
    set(gca, 'TickDir', 'out');

    % 设置纸张属性
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [12 8]); % 设置为 12x8 英寸
    set(gcf, 'PaperPosition', [0 0 12 8]);

    % 保存矢量图
    save_figure(fig, figure_dir, 'roc_curves', 'Formats', {'svg'});
    close(fig);
end

%% 修改2：修复create_variable_importance_plot函数
function create_variable_importance_plot(results, methods, var_names, figure_dir)
    % 计算每个变量的平均频率
    var_freq = zeros(length(var_names), 1);
    for i = 1:length(methods)
        method = methods{i};
        var_freq = var_freq + results.(method).var_freq;
    end
    var_freq = var_freq / length(methods);

    [sorted_freq, idx] = sort(var_freq, 'descend');
    sorted_names = var_names(idx);

    ;
    fig = figure('Name', 'Variable Importance', 'Position', [100, 100, 900, 700]);
    
    % 禁用工具栏
    set(gcf, 'Toolbar', 'none');

    h = barh(sorted_freq);
    set(h, 'FaceColor', 'flat');

    colormap(autumn);
    for i = 1:length(sorted_freq)
        h.CData(i,:) = [sorted_freq(i), 0.5, 1-sorted_freq(i)];
    end

    set(gca, 'YTick', 1:length(sorted_names), 'YTickLabel', sorted_names, 'FontSize', 10);
    xlabel('选择频率', 'FontSize', 12, 'FontWeight', 'bold');
    title('不同方法中变量重要性比较', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    box on;

    for i = 1:length(sorted_freq)
        text(sorted_freq(i) + 0.03, i, sprintf('%.2f', sorted_freq(i)), ...
            'VerticalAlignment', 'middle', 'FontSize', 9);
    end

    text(0.5, length(sorted_names) + 1.5, ...
        ['注: 此图显示每个变量在', num2str(length(methods)), '种方法中的平均选择频率'], ...
        'FontSize', 9, 'HorizontalAlignment', 'center');

    set(gcf, 'Color', 'white');
    set(gca, 'TickDir', 'out');
    set(gcf, 'Position', [100, 100, 900, max(500, 150 + 30*length(sorted_names))]);

    % 设置纸张属性
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [10 8]); % 设置为 10x8 英寸
    set(gcf, 'PaperPosition', [0 0 10 8]);

    % 保存矢量图
    save_figure(fig, figure_dir, 'variable_importance', 'Formats', {'svg'});
    close(fig);
end

%% 修改3：修复create_variable_group_plot函数中的保存图形部分
function create_variable_group_plot(results, methods, var_names, figure_dir)
    % 创建变量组合可视化图
    % 输入:
    %   results - 结果结构
    %   methods - 方法名称
    %   var_names - 变量名称
    %   figure_dir - 图形保存目录

    % 对于每种方法创建图形
    for i = 1:length(methods)
        method = methods{i};
        group_perf = results.(method).group_performance;
        
        % 如果有至少3个不同的组合，则创建图
        if length(group_perf) >= 2
            % 取前10个最常见的组合
            top_n = min(10, length(group_perf));
            
            % 提取数据
            combo_labels = cell(top_n, 1);
            combo_counts = zeros(top_n, 1);
            combo_aucs = zeros(top_n, 1);
            combo_acc = zeros(top_n, 1);
            combo_sens = zeros(top_n, 1);
            combo_spec = zeros(top_n, 1);
            combo_prec = zeros(top_n, 1);    
            combo_f1 = zeros(top_n, 1);
            
            for j = 1:top_n
                combo = group_perf(j);
                var_str = sprintf('组合 %d', j);
                combo_labels{j} = var_str;
                combo_counts(j) = combo.count;
                combo_aucs(j) = combo.auc;
                combo_acc(j) = combo.accuracy;
                combo_sens(j) = combo.sensitivity;
                combo_spec(j) = combo.specificity;
                combo_prec(j) = combo.precision;
                combo_f1(j) = combo.f1_score;
            end
            
            % 创建组合性能图
            fig1 = figure('Name', sprintf('%s Variable Combinations', method), 'Position', [100, 100, 1200, 800]);
            
            % 创建子图1：组合计数
            subplot(2, 2, 1);
            bar(combo_counts, 'FaceColor', [0.3 0.6 0.9]);
            set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
            title([method, ': 变量组合出现频率'], 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('频率', 'FontSize', 10);
            grid on;
            
            % 创建子图2：组合AUC和F1分数
            subplot(2, 2, 2);
            metrics_2 = [combo_aucs, combo_f1];
            h2 = bar(metrics_2);
            set(h2(1), 'FaceColor', [0.9 0.4 0.3]);
            set(h2(2), 'FaceColor', [0.3 0.8 0.8]);
            set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
            title([method, ': 变量组合AUC和F1值'], 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('值', 'FontSize', 10);
            legend({'AUC', 'F1分数'}, 'Location', 'southwest', 'FontSize', 8);
            grid on;
            
            % 创建子图3：准确率、敏感性和特异性
            subplot(2, 2, 3);
            metrics_3 = [combo_acc, combo_sens, combo_spec];
            h3 = bar(metrics_3);
            set(h3(1), 'FaceColor', [0.3 0.8 0.3]);
            set(h3(2), 'FaceColor', [0.9 0.6 0.1]);
            set(h3(3), 'FaceColor', [0.5 0.5 0.8]);
            set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
            title([method, ': 组合性能指标1'], 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('值', 'FontSize', 10);
            legend({'准确率', '敏感性', '特异性'}, 'Location', 'southeast', 'FontSize', 8);
            grid on;
            
            % 创建子图4：精确率（新增）
            subplot(2, 2, 4);
            bar(combo_prec, 'FaceColor', [0.6 0.3 0.8]);
            set(gca, 'XTick', 1:top_n, 'XTickLabel', combo_labels, 'XTickLabelRotation', 45, 'FontSize', 9);
            title([method, ': 变量组合精确率'], 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('精确率', 'FontSize', 10);
            grid on;
            
            % 调整整体间距
            set(gcf, 'Color', 'white');
            set(fig1, 'Position', [100, 100, 1200, 800]);
            
            % 保存矢量图 - 注意这里传入方法名
            save_figure(fig1, figure_dir, '%s_variable_combinations', 'MethodName', method, 'Formats', {'svg'});
            close(fig1);
            
            % 创建组合详情图
            fig2 = figure('Name', sprintf('%s Combination Details', method), 'Position', [100, 100, 1200, 800]);
            
            % 创建矩阵显示每个组合包含哪些变量
            combo_matrix = zeros(top_n, length(var_names));
            
            for j = 1:top_n
                combo = group_perf(j);
                for k = 1:length(combo.variables)
                    var_name = combo.variables{k};
                    var_idx = find(strcmp(var_names, var_name));
                    if ~isempty(var_idx)
                        combo_matrix(j, var_idx) = 1;
                    end
                end
            end
            
            % 绘制热图
            h = heatmap(combo_matrix, 'XDisplayLabels', var_names, 'YDisplayLabels', combo_labels);
            
            % 自定义颜色映射
            colormap([1 1 1; 0.2 0.6 0.8]); % 白色和蓝色
            
            % 设置标题和标签
            h.Title = sprintf('%s: 前%d个组合的变量构成', method, top_n);
            h.XLabel = '变量';
            h.YLabel = '组合';
            h.FontSize = 10;
            
            % 保存矢量图 - 注意这里传入方法名
            save_figure(fig2, figure_dir, '%s_combination_details', 'MethodName', method, 'Formats', {'svg'});
            close(fig2);
        else
            log_message('info', sprintf('%s方法的变量组合少于3个，跳过可视化', method));
        end
    end

    % 修复创建所有方法的综合比较图部分
    try
        % 从所有方法中收集组合信息
        all_combos = struct('method', {}, 'vars', {}, 'auc', {}, 'f1', {}, 'count', {});
        for i = 1:length(methods)
            method = methods{i};
            group_perf = results.(method).group_performance;
            
            for j = 1:min(3, length(group_perf))  % 每个方法取前3个
                if group_perf(j).count >= 5 % 只考虑出现至少5次的组合
                    combo = group_perf(j);
                    var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
                    new_combo = struct('method', method, 'vars', var_str, 'auc', combo.auc, 'f1', combo.f1_score, 'count', combo.count);
                    all_combos(end+1) = new_combo;
                end
            end
        end
        
        % 如果有足够的组合，创建比较图
        if length(all_combos) >= 3
            fig3 = figure('Name', 'Top Combinations Across Methods', 'Position', [100, 100, 1000, 600]);
            
            % 提取数据
            methods_list = {all_combos.method};
            aucs = [all_combos.auc];
            f1s = [all_combos.f1];
            counts = [all_combos.count];
            vars_list = {all_combos.vars};
            
            % 创建气泡图（使用F1分数作为Y轴）
            scatter(1:length(all_combos), f1s, counts*10, aucs*50, 'filled', 'MarkerFaceAlpha', 0.7);
            colormap(jet);
            colorbar;
            
            % 添加方法标签
            text(1:length(all_combos), f1s, methods_list, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
            
            % 设置图形属性
            set(gca, 'XTick', 1:length(all_combos));
            set(gca, 'XTickLabel', vars_list, 'XTickLabelRotation', 45, 'FontSize', 9);
            xlabel('变量组合', 'FontSize', 12);
            ylabel('F1分数', 'FontSize', 12);
            title('各方法中顶级变量组合的性能比较', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            % 添加说明
            text(length(all_combos)/2, min(f1s)-0.1, '注: 气泡大小表示组合出现频率，颜色表示AUC值', ...
                'HorizontalAlignment', 'center', 'FontSize', 10);
            
            % 保存矢量图
            save_figure(fig3, figure_dir, 'top_combinations_comparison', 'Formats', {'svg'});
            close(fig3);
        end
    catch ME
        log_message('warning', sprintf('创建综合比较图时出错: %s', ME.message));
    end
end

%% 图形保存函数
function save_figure(fig, output_dir, filename_base, varargin)
    % 解析输入参数
    p = inputParser;
    addParameter(p, 'Formats', {'svg'}, @(x) iscell(x) || ischar(x));
    addParameter(p, 'DPI', 300, @isnumeric);
    addParameter(p, 'MethodName', '', @ischar);
    parse(p, varargin{:});
    
    formats = p.Results.Formats;
    dpi = p.Results.DPI;
    method_name = p.Results.MethodName;
    
    % 如果formats是字符串，转换为cell数组
    if ischar(formats)
        formats = {formats};
    end
    
    % 确保输出目录存在
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % 处理文件名中的方法名称替换
    if contains(filename_base, '%s')
        if ~isempty(method_name)
            actual_filename = sprintf(filename_base, method_name);
        else
            actual_filename = strrep(filename_base, '%s', 'unknown');
            log_message('warning', '文件名中包含%s但未提供method_name');
        end
    else
        actual_filename = filename_base;
    end
    
    % 准备图形
    set(fig, 'Color', 'white');
    
    % 检查是否存在exportgraphics函数(R2020a或更高版本)
    has_exportgraphics = exist('exportgraphics', 'file') == 2;
    
    % 初始化成功保存的格式列表
    successful_formats = {};
    
    % 保存每种格式
    for i = 1:length(formats)
        format = lower(formats{i});
        filename = fullfile(output_dir, [actual_filename '.' format]);
        
        try
            if has_exportgraphics
                % 使用exportgraphics函数(适用于R2020a或更高版本)
                switch format
                    case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
                        % 位图格式
                        exportgraphics(fig, filename, 'Resolution', dpi);
                    case {'pdf', 'eps', 'svg'}
                        % 矢量格式
                        exportgraphics(fig, filename, 'ContentType', 'vector');
                    otherwise
                        % 其他格式回退到saveas
                        saveas(fig, filename);
                end
            else
                % 回退到传统方法
                % 禁用工具栏和菜单栏
                set(fig, 'Toolbar', 'none');
                set(fig, 'MenuBar', 'none');
                
                % 根据不同的格式选择不同的保存方法
                switch format
                    case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
                        print(fig, filename, ['-d' format], ['-r' num2str(dpi)]);
                    case 'pdf'
                        print(fig, filename, '-dpdf', '-bestfit');
                    case 'eps'
                        print(fig, filename, '-depsc2');
                    case 'svg'
                        print(fig, filename, '-dsvg');
                    otherwise
                        saveas(fig, filename);
                end
            end
            
            successful_formats{end+1} = upper(format);
            
        catch ME
            % 保存失败，记录错误并尝试备用方法
            log_message('debug', ['保存' upper(format) '格式时出错: ' ME.message]);
            
            % 尝试备用方法
            try
                saveas(fig, filename);
                successful_formats{end+1} = upper(format);
            catch ME2
                log_message('warning', ['备用保存方法也失败: ' ME2.message]);
            end
        end
    end
    
    % 输出汇总日志消息
    if ~isempty(successful_formats)
        formats_str = strjoin(successful_formats, ', ');
        log_message('info', sprintf('图形已保存: %s (%s)', actual_filename, formats_str));
    else
        log_message('error', ['图形保存失败: ' actual_filename]);
    end
end

%% 创建增强综合报告 - 新增
function create_enhanced_summary_report(results, methods, var_names, cv_results, coef_stability, param_stats, var_contribution, report_dir)
% 创建增强综合比较报告，包含K折验证、系数稳定性和变量贡献分析
% 输入:
%   results - 结果结构
%   methods - 方法名称
%   var_names - 变量名称
%   cv_results - 交叉验证结果
%   coef_stability - 系数稳定性
%   param_stats - 参数统计
%   var_contribution - 变量贡献
%   report_dir - 报告保存目录

% 添加时间戳
timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

% 打开文件
file_path = fullfile(report_dir, 'enhanced_summary_report.txt');
fid = fopen(file_path, 'w');

% 写入标题
fprintf(fid, '二元逻辑回归分析增强综合报告\n');
fprintf(fid, '==============================\n');
fprintf(fid, '生成时间: %s\n', timestamp);
fprintf(fid, '系统配置: MacBook Pro 2019, i9-9980HK, 64GB RAM, 8GB 5500M\n\n');

% 1. 交叉验证结果
fprintf(fid, '1. K折交叉验证结果\n');
fprintf(fid, '----------------------\n\n');

fprintf(fid, '使用%d折交叉验证评估模型稳定性和泛化能力:\n\n', length(cv_results.accuracy));

fprintf(fid, '平均性能指标：\n');
fprintf(fid, '  - 准确率: %.3f ± %.3f\n', cv_results.avg_accuracy, cv_results.std_accuracy);
fprintf(fid, '  - 精确率: %.3f ± %.3f\n', cv_results.avg_precision, cv_results.std_precision);
fprintf(fid, '  - 召回率: %.3f ± %.3f\n', cv_results.avg_recall, cv_results.std_recall);
fprintf(fid, '  - 特异性: %.3f ± %.3f\n', cv_results.avg_specificity, cv_results.std_specificity);
fprintf(fid, '  - F1分数: %.3f ± %.3f\n', cv_results.avg_f1_score, cv_results.std_f1_score);
fprintf(fid, '  - AUC: %.3f ± %.3f\n\n', cv_results.avg_auc, cv_results.std_auc);

% 系数稳定性分析
fprintf(fid, '系数稳定性分析：\n');
coef_means = cv_results.coef_mean;
coef_stds = cv_results.coef_std;
coef_cvs = cv_results.coef_cv;

% 输出截距
fprintf(fid, '  - 截距: %.4f ± %.4f (CV = %.3f)\n', coef_means(1), coef_stds(1), coef_cvs(1));

% 找出CV较大的系数（不稳定）
unstable_idx = find(coef_cvs > 0.5);
if ~isempty(unstable_idx)
    fprintf(fid, '  - 检测到不稳定系数 (CV > 0.5):\n');
    for i = 1:length(unstable_idx)
        if unstable_idx(i) > 1 % 跳过截距
            var_idx = unstable_idx(i) - 1;
            if var_idx <= length(var_names)
                fprintf(fid, '    * %s: %.4f ± %.4f (CV = %.3f)\n', ...
                    var_names{var_idx}, coef_means(unstable_idx(i)), coef_stds(unstable_idx(i)), coef_cvs(unstable_idx(i)));
            end
        end
    end
else
    fprintf(fid, '  - 所有系数均表现稳定 (CV <= 0.5)\n');
end
fprintf(fid, '\n');

% 2. 变量选择结果
fprintf(fid, '2. 变量选择结果\n');
fprintf(fid, '----------------------\n\n');

for i = 1:length(methods)
    method = methods{i};
    selected = find(results.(method).selected_vars);
    selected_names = var_names(selected);
    
    fprintf(fid, '%s方法选择的变量：\n', method);
    for j = 1:length(selected_names)
        fprintf(fid, '  - %s (频率: %.2f)\n', selected_names{j}, results.(method).var_freq(selected(j)));
    end
    fprintf(fid, '\n');
end

% 3. 变量组合分析
fprintf(fid, '3. 变量组合分析\n');
fprintf(fid, '----------------------\n\n');

for i = 1:length(methods)
    method = methods{i};
    group_perf = results.(method).group_performance;
    
    fprintf(fid, '%s方法发现的变量组合（前5个）：\n', method);
    
    % 显示前5个最常见的组合
    top_n = min(5, length(group_perf));
    for j = 1:top_n
        combo = group_perf(j);
        var_str = strjoin(cellfun(@(x) x, combo.variables, 'UniformOutput', false), ', ');
        fprintf(fid, '  组合 #%d (出现%d次):\n', j, combo.count);
        fprintf(fid, '    变量: %s\n', var_str);
        fprintf(fid, '    性能: 准确率=%.3f, 精确率=%.3f, 召回率=%.3f, 特异性=%.3f, F1=%.3f, AUC=%.3f\n', ...
            combo.accuracy, combo.precision, combo.sensitivity, combo.specificity, combo.f1_score, combo.auc);
    end
    fprintf(fid, '\n');
end

% 4. 模型性能比较
fprintf(fid, '4. 模型性能比较\n');
fprintf(fid, '----------------------\n\n');

fprintf(fid, '方法\t\t准确率\t精确率\t召回率\t特异性\tF1分数\tAUC\n');
fprintf(fid, '------------------------------------------------------------------------\n');
for i = 1:length(methods)
    method = methods{i};
    perf = results.(method).performance;
    fprintf(fid, '%-12s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', ...
        method, perf.avg_accuracy, perf.avg_precision, perf.avg_sensitivity, ...
        perf.avg_specificity, perf.avg_f1_score, perf.avg_auc);
end
fprintf(fid, '\n');

% 5. 参数统计分析
fprintf(fid, '5. 参数统计分析\n');
fprintf(fid, '----------------------\n\n');

for i = 1:length(methods)
    method = methods{i};
    
    % 只针对回归类方法
    if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
        if isfield(param_stats, method) && isfield(param_stats.(method), 'table')
            table_data = param_stats.(method).table;
            
            fprintf(fid, '%s方法的参数统计 (基于%d个模型):\n\n', method, param_stats.(method).n_samples);
            fprintf(fid, 'Variable\t\tEstimate\t95%% CI\t\t\tp-value\tSignificance\n');
            fprintf(fid, '------------------------------------------------------------------------\n');
            
            for j = 1:height(table_data)
                % 动态查找置信区间列名
                column_names = table_data.Properties.VariableNames;
                ci_lower_col = '';
                ci_upper_col = '';
                
                % 查找包含'lower'和'upper'的列名（优先使用t分布CI）
                for col_idx = 1:length(column_names)
                    col_name = lower(column_names{col_idx});
                    if contains(col_name, 'lower') && contains(col_name, 't')
                        ci_lower_col = column_names{col_idx};
                    elseif contains(col_name, 'upper') && contains(col_name, 't')
                        ci_upper_col = column_names{col_idx};
                    end
                end
                
                % 如果未找到t分布CI，尝试BCa
                if isempty(ci_lower_col) || isempty(ci_upper_col)
                    for col_idx = 1:length(column_names)
                        col_name = lower(column_names{col_idx});
                        if contains(col_name, 'lower') && contains(col_name, 'bca')
                            ci_lower_col = column_names{col_idx};
                        elseif contains(col_name, 'upper') && contains(col_name, 'bca')
                            ci_upper_col = column_names{col_idx};
                        end
                    end
                end
                
                % 输出表格行
                if isempty(ci_lower_col) || isempty(ci_upper_col)
                    fprintf(fid, '%-15s\t%.4f\t[N/A, N/A]\t%.4f\t%s\n', ...
                        table_data.Variable{j}, table_data.Estimate(j), ...
                        table_data.p_value(j), table_data.Significance{j});
                    log_message('warning', sprintf('%s方法的参数统计表中缺少置信区间列', method));
                else
                    fprintf(fid, '%-15s\t%.4f\t[%.4f, %.4f]\t%.4f\t%s\n', ...
                        table_data.Variable{j}, table_data.Estimate(j), ...
                        table_data.(ci_lower_col)(j), table_data.(ci_upper_col)(j), ...
                        table_data.p_value(j), table_data.Significance{j});
                end
            end
            fprintf(fid, '\n');
            
            % 输出显著参数
            sig_idx = find(table_data.p_value < 0.05);
            if ~isempty(sig_idx)
                fprintf(fid, '统计显著变量 (p < 0.05):\n');
                for j = 1:length(sig_idx)
                    if ~isempty(ci_lower_col) && ~isempty(ci_upper_col)
                        fprintf(fid, '  - %s: 估计值=%.4f, 95%%CI=[%.4f, %.4f], p=%.4f %s\n', ...
                            table_data.Variable{sig_idx(j)}, table_data.Estimate(sig_idx(j)), ...
                            table_data.(ci_lower_col)(sig_idx(j)), table_data.(ci_upper_col)(sig_idx(j)), ...
                            table_data.p_value(sig_idx(j)), table_data.Significance{sig_idx(j)});
                    else
                        fprintf(fid, '  - %s: 估计值=%.4f, p=%.4f %s\n', ...
                            table_data.Variable{sig_idx(j)}, table_data.Estimate(sig_idx(j)), ...
                            table_data.p_value(sig_idx(j)), table_data.Significance{sig_idx(j)});
                    end
                end
            else
                fprintf(fid, '没有发现统计显著变量 (p < 0.05)\n');
            end
            fprintf(fid, '\n');
        else
            fprintf(fid, '%s方法没有可用的参数统计信息\n\n', method);
        end
    else
        fprintf(fid, '%s方法不适用于传统参数统计分析\n\n', method);
    end
end

% 6. 变量贡献分析
fprintf(fid, '6. 变量贡献分析\n');
fprintf(fid, '----------------------\n\n');

% 6.1 全局变量重要性
if isfield(var_contribution, 'overall_importance')
    fprintf(fid, '综合变量重要性排名:\n');
    
    importance_table = var_contribution.overall_importance;
    top_n = min(10, height(importance_table));
    
    for i = 1:top_n
        fprintf(fid, '  %d. %s (重要性: %.2f%%)\n', i, importance_table.Variable{i}, importance_table.Normalized_Importance(i));
    end
    fprintf(fid, '\n');
end

% 6.2 相关性分析
if isfield(var_contribution, 'correlation')
    fprintf(fid, '变量相关性分析:\n');
    
    corr_table = var_contribution.correlation;
    [~, idx] = sort(abs(corr_table.PartialCorr), 'descend');
    sorted_corr = corr_table(idx,:);
    top_n = min(5, height(sorted_corr));
    
    fprintf(fid, '变量\t\t相关系数\t偏相关系数\tp值\n');
    fprintf(fid, '--------------------------------------------------------\n');
    for i = 1:top_n
        fprintf(fid, '%-15s\t%.3f\t\t%.3f\t\t%.4f\n', ...
            sorted_corr.Variable{i}, sorted_corr.Correlation(i), ...
            sorted_corr.PartialCorr(i), sorted_corr.Partial_pvalue(i));
    end
    fprintf(fid, '\n');
end

% 6.3 各方法变量贡献
for i = 1:length(methods)
    method = methods{i};
    
    if isfield(var_contribution, 'methods') && isfield(var_contribution.methods, method) && ...
       isfield(var_contribution.methods.(method), 'contribution_table')
    
        contrib_table = var_contribution.methods.(method).contribution_table;
        fprintf(fid, '%s方法的变量贡献分析:\n', method);
        
        if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
            fprintf(fid, '变量\t\t\t系数\t\tp值\t\t相对贡献\t方向\n');
            fprintf(fid, '--------------------------------------------------------\n');
            
            top_n = min(5, height(contrib_table));
            for j = 1:top_n
                fprintf(fid, '%-20s\t%.4f\t\t%.4f\t\t%.2f%%\t\t%s\n', ...
                    contrib_table.Variable{j}, contrib_table.Coefficient(j), ...
                    contrib_table.p_value(j), contrib_table.Relative_Contribution(j), ...
                    contrib_table.Effect_Direction{j});
            end
        else
            fprintf(fid, '变量\t\t\t重要性\t\t相对贡献\n');
            fprintf(fid, '--------------------------------------------------------\n');
            
            top_n = min(5, height(contrib_table));
            for j = 1:top_n
                fprintf(fid, '%-20s\t%.4f\t\t%.2f%%\n', ...
                    contrib_table.Variable{j}, contrib_table.Importance(j), ...
                    contrib_table.Relative_Contribution(j));
            end
        end
        fprintf(fid, '\n');
    end
end

% 7. 最佳变量组合性能 
fprintf(fid, '7. 最佳变量组合性能\n');
fprintf(fid, '----------------------\n\n');

% 找出所有方法中性能最好的组合（基于F1分数）
best_combo = struct('method', '', 'f1', 0, 'auc', 0, 'count', 0, 'variables', {{}}, ...
    'accuracy', 0, 'precision', 0, 'sensitivity', 0, 'specificity', 0);
found_valid_combo = false;

for i = 1:length(methods)
    method = methods{i};
    group_perf = results.(method).group_performance;
    
    for j = 1:length(group_perf)
        combo = group_perf(j);
        % 只考虑出现频率较高的组合（至少5次）
        if combo.count >= 5 && combo.f1_score > best_combo.f1
            best_combo.method = method;
            best_combo.f1 = combo.f1_score;
            best_combo.auc = combo.auc;
            best_combo.count = combo.count;
            best_combo.variables = combo.variables;
            best_combo.accuracy = combo.accuracy;
            best_combo.precision = combo.precision;
            best_combo.sensitivity = combo.sensitivity;
            best_combo.specificity = combo.specificity;
            found_valid_combo = true;
        end
    end
end

fprintf(fid, '所有方法中性能最佳的变量组合（基于F1分数）：\n');
if found_valid_combo
    fprintf(fid, '  方法: %s\n', best_combo.method);
    fprintf(fid, '  出现次数: %d\n', best_combo.count);
    fprintf(fid, '  F1分数: %.3f\n', best_combo.f1);
    fprintf(fid, '  AUC: %.3f\n', best_combo.auc);
    fprintf(fid, '  准确率: %.3f\n', best_combo.accuracy);
    fprintf(fid, '  精确率: %.3f\n', best_combo.precision);
    fprintf(fid, '  召回率/敏感性: %.3f\n', best_combo.sensitivity);
    fprintf(fid, '  特异性: %.3f\n', best_combo.specificity);
    fprintf(fid, '  变量: %s\n\n', strjoin(best_combo.variables, ', '));
else
    fprintf(fid, '  未找到出现频率达到5次以上的变量组合\n\n');
end

% 8. 系数稳定性分析
fprintf(fid, '8. 系数稳定性分析\n');
fprintf(fid, '----------------------\n\n');

for i = 1:length(methods)
    method = methods{i};
    
    % 只针对回归类方法
    if any(strcmpi(method, {'stepwise', 'lasso', 'ridge', 'elasticnet'}))
        if isfield(coef_stability, method) && isfield(coef_stability.(method), 'table')
            table_data = coef_stability.(method).table;
            
            fprintf(fid, '%s方法的系数稳定性分析:\n\n', method);
            fprintf(fid, 'Variable\t\tMean\t\tStdDev\t\tCV\n');
            fprintf(fid, '--------------------------------------------------------\n');
            
            for j = 1:height(table_data)
                fprintf(fid, '%-15s\t%.4f\t\t%.4f\t\t%.3f\n', ...
                    table_data.Variable{j}, table_data.Mean(j), ...
                    table_data.StdDev(j), table_data.CV(j));
            end
            fprintf(fid, '\n');
            
            % 输出不稳定的系数
            unstable_idx = find(table_data.CV > 0.5);
            if ~isempty(unstable_idx)
                fprintf(fid, '不稳定系数 (CV > 0.5):\n');
                for j = 1:length(unstable_idx)
                    fprintf(fid, '  - %s: CV=%.3f\n', ...
                        table_data.Variable{unstable_idx(j)}, table_data.CV(unstable_idx(j)));
                end
            else
                fprintf(fid, '所有系数均表现稳定 (CV <= 0.5)\n');
            end
            fprintf(fid, '\n');
        else
            fprintf(fid, '%s方法没有可用的系数稳定性信息\n\n', method);
        end
    else
        fprintf(fid, '%s方法不适用于传统系数稳定性分析\n\n', method);
    end
end

% 9. 变量重要性
fprintf(fid, '9. 变量重要性（平均选择频率）\n');
fprintf(fid, '----------------------\n\n');

% 计算每个变量的平均频率
var_freq = zeros(length(var_names), 1);
for i = 1:length(methods)
    method = methods{i};
    var_freq = var_freq + results.(method).var_freq;
end
var_freq = var_freq / length(methods);

% 排序并输出
[sorted_freq, idx] = sort(var_freq, 'descend');
sorted_names = var_names(idx);

for i = 1:length(sorted_names)
    fprintf(fid, '%-15s\t%.3f\n', sorted_names{i}, sorted_freq(i));
end
fprintf(fid, '\n');

% 10. 并行性能统计
fprintf(fid, '10. 并行计算性能\n');
fprintf(fid, '----------------------\n\n');

fprintf(fid, '并行池配置:\n');
fprintf(fid, '  - CPU: Intel i9-9980HK (8核16线程)\n');
fprintf(fid, '  - Worker数量: 8个\n');
fprintf(fid, '  - 每个Worker线程数: 2\n');
fprintf(fid, '  - 总并行线程: 16\n');
fprintf(fid, '  - 保留线程: 0 (系统和主进程)\n\n');

% 11. 结论和推荐
fprintf(fid, '11. 结论和推荐\n');
fprintf(fid, '----------------------\n\n');

% 找出性能最好的方法（基于F1分数）
f1_values = zeros(length(methods), 1);
for i = 1:length(methods)
    f1_values(i) = results.(methods{i}).performance.avg_f1_score;
end

if all(isnan(f1_values))
    fprintf(fid, '所有方法的F1值都是NaN，无法确定最佳方法。\n\n');
else
    [max_f1, best_idx] = max(f1_values);
    best_method = methods{best_idx};
    fprintf(fid, '根据F1分数指标，%s方法表现最好，平均F1分数为%.3f。\n\n', best_method, max_f1);
end

% 推荐变量组合
fprintf(fid, '推荐变量组合：\n');
if found_valid_combo
    fprintf(fid, '  %s\n', strjoin(best_combo.variables, ', '));
    fprintf(fid, '  (来自%s方法，F1=%.3f, AUC=%.3f)\n\n', best_combo.method, best_combo.f1, best_combo.auc);
else
    fprintf(fid, '  未找到出现频率达到5次以上的变量组合\n\n');
end

% 12. 模型稳定性评价
fprintf(fid, '12. 模型稳定性评价\n');
fprintf(fid, '----------------------\n\n');

% 评估K折交叉验证性能的变异系数
cv_metrics = {'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc'};
cv_metric_names = {'准确率', '精确率', '召回率', '特异性', 'F1分数', 'AUC'};
cv_values = zeros(length(cv_metrics), 1);

for i = 1:length(cv_metrics)
    metric = cv_metrics{i};
    mean_val = mean(cv_results.(metric), 'omitnan');
    std_val = std(cv_results.(metric), 'omitnan');
    % 计算变异系数
    if mean_val > 0
        cv_values(i) = std_val / mean_val;
    else
        cv_values(i) = NaN;
    end
end

fprintf(fid, 'K折交叉验证各指标变异系数 (CV):\n');
for i = 1:length(cv_metrics)
    fprintf(fid, '  - %s: CV = %.3f\n', cv_metric_names{i}, cv_values(i));
end

% 评估模型稳定性
mean_cv = mean(cv_values, 'omitnan');
if mean_cv < 0.1
    stability_rating = '优秀';
elseif mean_cv < 0.2
    stability_rating = '良好';
elseif mean_cv < 0.3
    stability_rating = '一般';
else
    stability_rating = '较差';
end

fprintf(fid, '\n总体模型稳定性评价: %s (平均CV = %.3f)\n', stability_rating, mean_cv);

% 变量选择稳定性
best_method_freq = results.(best_method).var_freq;
high_freq_vars = best_method_freq > 0.5;
low_freq_vars = best_method_freq < 0.2 & best_method_freq > 0;

fprintf(fid, '\n变量选择稳定性 (在%s方法中):\n', best_method);
fprintf(fid, '  - 高稳定性变量 (频率 > 0.5): %d个\n', sum(high_freq_vars));
fprintf(fid, '  - 低稳定性变量 (频率 < 0.2): %d个\n', sum(low_freq_vars));

if sum(high_freq_vars) > 0
    fprintf(fid, '\n高稳定性变量名称:\n');
    high_vars = var_names(high_freq_vars);
    for i = 1:length(high_vars)
        selected_freq = best_method_freq(high_freq_vars);
        for i = 1:length(high_vars)
            fprintf(fid, '  - %s (频率: %.2f)\n', high_vars{i}, selected_freq(i));
        end
    end
end
fprintf(fid, '\n');

% 13. 总结
fprintf(fid, '13. 总结\n');
fprintf(fid, '----------------------\n\n');

fprintf(fid, '本分析在MacBook Pro 2019上运行，充分利用了i9-9980HK的多核性能\n');
fprintf(fid, '和64GB大内存优势。优化的并行配置、智能的GPU利用和高效的内存管理\n');
fprintf(fid, '确保了5种变量选择方法在多次重复实验中的高效执行。\n\n');

fprintf(fid, '本研究扩展了原有分析框架，增加了以下关键功能：\n');
fprintf(fid, '1. 结合K折交叉验证与Bootstrap方法，全面评估模型稳定性\n');
fprintf(fid, '2. 监控多种评估指标（准确率、精确率、召回率、F1分数、AUC等）\n');
fprintf(fid, '3. 分析了模型系数的稳定性，并计算了参数置信区间和p值\n');
fprintf(fid, '4. 通过多种方法评估了变量对模型的贡献\n\n');

if found_valid_combo
    fprintf(fid, '分析结果表明，%s方法选择的变量组合在预测性能上表现最优，\n', best_combo.method);
    fprintf(fid, '可作为后续模型构建的基础。推荐变量组合：%s\n\n', strjoin(best_combo.variables, ', '));
else
    fprintf(fid, '本次分析虽未找到普遍稳定的变量组合，但各方法的变量选择频率\n');
    fprintf(fid, '为后续变量筛选提供了重要参考。建议结合领域知识进行进一步筛选。\n\n');
end

fprintf(fid, '所有详细结果已保存至results文件夹，包括详细的CSV表格和可视化图形。\n\n');

fprintf(fid, '分析完成时间: %s\n', timestamp);

% 关闭文件
fclose(fid);
end
