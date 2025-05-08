classdef ParallelManager < handle
    % 并行计算管理类：专门针对i9-9980HK优化的并行计算管理
    % 包括并行池管理、任务分配和性能监控
    
    properties (Access = private)
        Config
        Logger
        ParallelPool
        ClusterProfile
        PerformanceStats
    end
    
    properties (Access = public)
        IsInitialized = false
        PoolSize
        ThreadsPerWorker
    end
    
    methods (Access = public)
        function obj = ParallelManager(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.PerformanceStats = struct();
            
            if obj.Config.UseParallelComputing
                obj.Initialize();
            end
        end
        
        function delete(obj)
            % 析构函数：清理并行池
            if obj.IsInitialized
                obj.Cleanup();
            end
        end
        
        function Initialize(obj)
            % 初始化并行计算环境
            if obj.IsInitialized
                obj.Logger.Log('info', '并行环境已初始化，跳过');
                return;
            end
            
            try
                obj.Logger.Log('info', '正在初始化并行计算环境...');
                
                % 检查现有并行池
                if ~isempty(gcp('nocreate'))
                    obj.Logger.Log('info', '使用现有并行池');
                    obj.ParallelPool = gcp;
                    obj.PoolSize = obj.ParallelPool.NumWorkers;
                    obj.ThreadsPerWorker = obj.ParallelPool.NumThreads;
                else
                    % 创建新的并行池
                    obj.CreateParallelPool();
                end
                
                % 配置并行环境
                obj.ConfigureParallelEnvironment();
                
                % 初始化性能统计
                obj.InitializePerformanceStats();
                
                obj.IsInitialized = true;
                obj.Logger.Log('info', '并行计算环境初始化完成');
                
            catch ME
                obj.Logger.LogException(ME, 'ParallelManager.Initialize');
                rethrow(ME);
            end
        end
        
        function results = RunParallelTask(obj, taskFun, taskData, taskName)
            % 运行并行任务
            % 输入:
            %   taskFun - 任务函数句柄
            %   taskData - 任务数据
            %   taskName - 任务名称（用于日志）
            % 输出:
            %   results - 任务结果
            
            if ~obj.IsInitialized
                obj.Initialize();
            end
            
            start_time = tic;
            obj.Logger.Log('info', sprintf('开始并行任务: %s', taskName));
            
            try
                % 记录任务开始
                task_id = datestr(now, 'yyyymmdd_HHMMSS');
                obj.PerformanceStats.tasks.(task_id) = struct();
                obj.PerformanceStats.tasks.(task_id).name = taskName;
                obj.PerformanceStats.tasks.(task_id).start_time = datetime('now');
                
                % 预分配结果
                n_tasks = length(taskData);
                results = cell(n_tasks, 1);
                
                % 创建进度监控
                progress_interval = max(1, round(n_tasks / 20));
                
                % 使用parfor执行任务
                parfor i = 1:n_tasks
                    try
                        results{i} = taskFun(taskData{i});
                        
                        % 定期报告进度
                        if mod(i, progress_interval) == 0
                            obj.Logger.LogProgress(i, n_tasks, sprintf('%s 进度', taskName));
                        end
                    catch task_error
                        % 记录单个任务错误但不中断整体流程
                        warning('Task %d failed: %s', i, task_error.message);
                        results{i} = [];
                    end
                end
                
                % 记录任务完成
                duration = toc(start_time);
                obj.PerformanceStats.tasks.(task_id).duration = duration;
                obj.PerformanceStats.tasks.(task_id).num_tasks = n_tasks;
                obj.PerformanceStats.tasks.(task_id).throughput = n_tasks / duration;
                
                obj.Logger.LogPerformance(taskName, duration, sprintf('%d 个子任务', n_tasks));
                
            catch ME
                obj.Logger.LogException(ME, sprintf('ParallelTask: %s', taskName));
                rethrow(ME);
            end
        end
        
        function futures = RunAsyncTask(obj, taskFun, taskData, taskName)
            % 运行异步任务
            % 输入:
            %   taskFun - 任务函数句柄
            %   taskData - 任务数据
            %   taskName - 任务名称
            % 输出:
            %   futures - 未来对象数组
            
            if ~obj.IsInitialized
                obj.Initialize();
            end
            
            obj.Logger.Log('info', sprintf('启动异步任务: %s', taskName));
            
            try
                n_tasks = length(taskData);
                futures = cell(n_tasks, 1);
                
                % 使用parfeval异步执行
                for i = 1:n_tasks
                    futures{i} = parfeval(@obj.ExecuteTask, 1, taskFun, taskData{i}, i);
                end
                
                obj.Logger.Log('info', sprintf('异步任务启动完成，%d 个子任务', n_tasks));
                
            catch ME
                obj.Logger.LogException(ME, sprintf('AsyncTask: %s', taskName));
                rethrow(ME);
            end
        end
        
        function results = WaitForAsyncResults(obj, futures, taskName)
            % 等待异步任务结果
            % 输入:
            %   futures - 未来对象数组
            %   taskName - 任务名称
            % 输出:
            %   results - 任务结果
            
            obj.Logger.Log('info', sprintf('等待异步任务完成: %s', taskName));
            
            try
                n_tasks = length(futures);
                results = cell(n_tasks, 1);
                completed = false(n_tasks, 1);
                
                % 监控进度
                start_time = tic;
                
                while ~all(completed)
                    for i = 1:n_tasks
                        if ~completed(i) && strcmp(futures{i}.State, 'finished')
                            results{i} = fetchOutputs(futures{i});
                            completed(i) = true;
                            
                            % 报告进度
                            progress = sum(completed) / n_tasks * 100;
                            obj.Logger.LogProgress(sum(completed), n_tasks, sprintf('%s 收集结果', taskName));
                        end
                    end
                    
                    % 防止CPU占用过高
                    pause(0.1);
                end
                
                duration = toc(start_time);
                obj.Logger.LogPerformance(sprintf('等待异步任务: %s', taskName), duration);
                
            catch ME
                obj.Logger.LogException(ME, sprintf('WaitForAsyncResults: %s', taskName));
                rethrow(ME);
            end
        end
        
        function Cleanup(obj)
            % 清理并行环境
            try
                obj.Logger.Log('info', '正在清理并行环境...');
                
                % 记录并行性能统计
                obj.LogPerformanceStats();
                
                % 清理临时目录
                if exist(obj.Config.ParallelClusterJobStorageLocation, 'dir')
                    rmdir(obj.Config.ParallelClusterJobStorageLocation, 's');
                end
                
                % 关闭并行池
                if ~isempty(obj.ParallelPool)
                    delete(obj.ParallelPool);
                    obj.ParallelPool = [];
                end
                
                obj.IsInitialized = false;
                obj.Logger.Log('info', '并行环境清理完成');
                
            catch ME
                obj.Logger.LogException(ME, 'ParallelManager.Cleanup');
            end
        end
        
        function stats = GetPerformanceStats(obj)
            % 获取并行性能统计信息
            stats = obj.PerformanceStats;
        end
        
        function opts = GetParallelOptions(obj)
            % 获取并行选项配置
            opts = statset('UseParallel', obj.IsInitialized, ...
                'UseSubstreams', true, ...
                'Display', 'off');
        end
    end
    
    methods (Access = private)
        function CreateParallelPool(obj)
            % 创建针对i9-9980HK优化的并行池
            try
                % 创建本地集群配置
                obj.ClusterProfile = parcluster('local');
                
                % 设置工作器数量
                obj.PoolSize = obj.Config.NumWorkers;
                obj.ThreadsPerWorker = obj.Config.NumThreadsPerWorker;
                
                % 设置并行配置
                obj.ClusterProfile.NumWorkers = obj.PoolSize;
                obj.ClusterProfile.NumThreads = obj.ThreadsPerWorker;
                
                % 配置存储位置
                obj.ClusterProfile.JobStorageLocation = obj.Config.ParallelClusterJobStorageLocation;
                
                % 优化并行池设置
                obj.ClusterProfile.AdditionalProperties.ClusterMatlabRoot = matlabroot;
                obj.ClusterProfile.AdditionalProperties.UseUniqueSubfolders = true;
                
                % 保存配置
                obj.ClusterProfile.saveProfile;
                
                % 创建并行池
                obj.ParallelPool = parpool(obj.ClusterProfile, obj.PoolSize);
                
                obj.Logger.Log('info', sprintf('创建并行池成功: %d workers x %d threads', ...
                    obj.PoolSize, obj.ThreadsPerWorker));
                
            catch ME
                obj.Logger.LogException(ME, 'CreateParallelPool');
                rethrow(ME);
            end
        end
        
        function ConfigureParallelEnvironment(obj)
            % 配置并行环境
            try
                % 设置并行池属性
                obj.ParallelPool.AutoCreate = false;
                obj.ParallelPool.IdleTimeout = 60;  % 60分钟空闲超时
                
                % 初始化工作器
                spmd
                    % 在每个工作器上设置随机数种子
                    rng(42 + labindex, 'twister');
                    
                    % 设置警告级别
                    warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');
                end
                
                % 预热并行池（执行简单操作以初始化工作器）
                obj.WarmupParallelPool();
                
            catch ME
                obj.Logger.LogException(ME, 'ConfigureParallelEnvironment');
            end
        end
        
        function WarmupParallelPool(obj)
            % 预热并行池
            obj.Logger.Log('debug', '正在预热并行池...');
            
            % 执行简单的并行操作
            dummy_data = num2cell(1:obj.PoolSize);
            
            parfor i = 1:obj.PoolSize
                % 简单的数学运算
                result = sum(rand(1000, 1000));
            end
            
            obj.Logger.Log('debug', '并行池预热完成');
        end
        
        function InitializePerformanceStats(obj)
            % 初始化性能统计
            obj.PerformanceStats = struct();
            obj.PerformanceStats.pool_creation_time = datetime('now');
            obj.PerformanceStats.pool_size = obj.PoolSize;
            obj.PerformanceStats.threads_per_worker = obj.ThreadsPerWorker;
            obj.PerformanceStats.total_threads = obj.PoolSize * obj.ThreadsPerWorker;
            obj.PerformanceStats.tasks = struct();
            obj.PerformanceStats.memory_usage = struct();
            obj.PerformanceStats.cpu_utilization = struct();
        end
        
        function result = ExecuteTask(~, taskFun, taskData, taskIndex)
            % 执行单个任务（用于parfeval）
            try
                result = taskFun(taskData);
            catch ME
                warning('Task %d failed: %s', taskIndex, ME.message);
                result = [];
            end
        end
        
        function LogPerformanceStats(obj)
            % 记录并行性能统计
            obj.Logger.CreateSection('并行性能统计');
            
            stats = obj.PerformanceStats;
            
            % 基本信息
            obj.Logger.Log('info', sprintf('并行池配置: %d workers x %d threads', ...
                stats.pool_size, stats.threads_per_worker));
            obj.Logger.Log('info', sprintf('总并行线程: %d', stats.total_threads));
            
            % 任务统计
            task_names = fieldnames(stats.tasks);
            if ~isempty(task_names)
                obj.Logger.Log('info', sprintf('执行任务数: %d', length(task_names)));
                
                total_duration = 0;
                total_tasks = 0;
                
                for i = 1:length(task_names)
                    task = stats.tasks.(task_names{i});
                    total_duration = total_duration + task.duration;
                    total_tasks = total_tasks + task.num_tasks;
                    
                    obj.Logger.Log('debug', sprintf('  任务: %s', task.name));
                    obj.Logger.Log('debug', sprintf('    持续时间: %.2f秒', task.duration));
                    obj.Logger.Log('debug', sprintf('    子任务数: %d', task.num_tasks));
                    obj.Logger.Log('debug', sprintf('    吞吐量: %.2f 任务/秒', task.throughput));
                end
                
                avg_throughput = total_tasks / total_duration;
                obj.Logger.Log('info', sprintf('平均吞吐量: %.2f 任务/秒', avg_throughput));
            end
            
            % 估计的并行加速比
            if total_duration > 0
                estimated_sequential_time = total_tasks * (total_duration / (stats.total_threads * 0.8));  % 假设80%效率
                speedup = estimated_sequential_time / total_duration;
                obj.Logger.Log('info', sprintf('估计并行加速比: %.2fx', speedup));
            end
        end
    end
end