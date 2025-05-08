classdef GPUManager < handle
    % GPU管理类：专门针对AMD GPU的智能管理
    % 自动检测数据大小，决定是否使用GPU加速
    
    properties (Access = private)
        Config
        Logger
        GPUDevice
        GPUMemoryInfo
        GPUAvailable
        TransferThreshold
    end
    
    properties (Access = public)
        IsInitialized = false
        GPUEnabled = false
        TransferCount = 0
        GPUMemoryUsed = 0
    end
    
    methods (Access = public)
        function obj = GPUManager(config, logger)
            % 构造函数
            obj.Config = config;
            obj.Logger = logger;
            obj.TransferCount = 0;
            obj.GPUMemoryUsed = 0;
            
            if obj.Config.UseGpu
                obj.Initialize();
            end
        end
        
        function delete(obj)
            % 析构函数：清理GPU内存
            if obj.IsInitialized
                obj.Cleanup();
            end
        end
        
        function Initialize(obj)
            % 初始化GPU环境
            if obj.IsInitialized
                obj.Logger.Log('info', 'GPU环境已初始化，跳过');
                return;
            end
            
            try
                obj.Logger.Log('info', '正在初始化GPU环境...');
                
                % 检查GPU可用性
                obj.CheckGPUAvailability();
                
                if obj.GPUAvailable
                    % 设置GPU设备
                    obj.SetupGPUDevice();
                    
                    % 检查内存
                    obj.CheckGPUMemory();
                    
                    % 设置数据传输阈值
                    obj.SetTransferThreshold();
                    
                    obj.GPUEnabled = true;
                    obj.Logger.Log('info', 'GPU环境初始化成功');
                else
                    obj.Logger.Log('warning', 'GPU不可用，将使用CPU计算');
                end
                
                obj.IsInitialized = true;
                
            catch ME
                obj.Logger.LogException(ME, 'GPUManager.Initialize');
                obj.GPUEnabled = false;
                obj.IsInitialized = true;  % 即使失败也标记为已初始化
            end
        end
        
        function gpu_data = ToGPU(obj, data)
            % 智能地将数据转移到GPU
            % 输入:
            %   data - 输入数据
            % 输出:
            %   gpu_data - GPU上的数据或原始数据
            
            if ~obj.GPUEnabled
                gpu_data = data;
                return;
            end
            
            try
                % 检查数据类型
                if ~isnumeric(data)
                    gpu_data = data;
                    return;
                end
                
                % 计算数据大小
                data_size = numel(data) * 8;  % 假设是double类型
                
                % 决定是否使用GPU
                if obj.ShouldUseGPU(data_size)
                    % 检查GPU内存
                    available_memory = obj.GPUDevice.AvailableMemory;
                    
                    if data_size < available_memory * 0.8  % 保留20%内存
                        gpu_data = gpuArray(data);
                        obj.TransferCount = obj.TransferCount + 1;
                        obj.GPUMemoryUsed = obj.GPUMemoryUsed + data_size;
                        
                        obj.Logger.Log('debug', sprintf('数据转移到GPU: %.2f MB', data_size / (1024^2)));
                    else
                        obj.Logger.Log('warning', 'GPU内存不足，使用CPU计算');
                        gpu_data = data;
                    end
                else
                    gpu_data = data;
                end
                
            catch ME
                obj.Logger.Log('warning', sprintf('GPU转移失败: %s，使用CPU计算', ME.message));
                gpu_data = data;
            end
        end
        
        function cpu_data = ToCPU(obj, gpu_data)
            % 将数据从GPU转移回CPU
            % 输入:
            %   gpu_data - GPU上的数据
            % 输出:
            %   cpu_data - CPU上的数据
            
            if isa(gpu_data, 'gpuArray')
                try
                    cpu_data = gather(gpu_data);
                    
                    % 更新统计信息
                    data_size = numel(cpu_data) * 8;
                    obj.Logger.Log('debug', sprintf('数据从GPU转移到CPU: %.2f MB', data_size / (1024^2)));
                    
                catch ME
                    obj.Logger.Log('warning', sprintf('GPU数据回传失败: %s', ME.message));
                    cpu_data = gpu_data;  % 如果失败，返回原始数据
                end
            else
                cpu_data = gpu_data;  % 如果不是GPU数据，直接返回
            end
        end
        
        function result = GPUCompute(obj, operation, data, varargin)
            % 在GPU上执行计算操作
            % 输入:
            %   operation - 操作函数句柄
            %   data - 输入数据
            %   varargin - 额外参数
            % 输出:
            %   result - 计算结果
            
            if ~obj.GPUEnabled
                result = operation(data, varargin{:});
                return;
            end
            
            try
                % 将数据转移到GPU
                gpu_data = obj.ToGPU(data);
                
                % 在GPU上执行操作
                start_time = tic;
                gpu_result = operation(gpu_data, varargin{:});
                gpu_duration = toc(start_time);
                
                % 将结果转移回CPU
                result = obj.ToCPU(gpu_result);
                
                obj.Logger.Log('debug', sprintf('GPU计算完成，耗时: %.4f秒', gpu_duration));
                
            catch ME
                obj.Logger.Log('warning', sprintf('GPU计算失败: %s，回退到CPU', ME.message));
                result = operation(data, varargin{:});
            end
        end
        
        function Cleanup(obj)
            % 清理GPU资源
            try
                obj.Logger.Log('info', '正在清理GPU资源...');
                
                % 清理GPU内存
                if obj.GPUEnabled
                    gpuDevice([]);  % 清空GPU
                    obj.Logger.Log('info', sprintf('GPU传输次数: %d', obj.TransferCount));
                    obj.Logger.Log('info', sprintf('总GPU内存使用: %.2f MB', obj.GPUMemoryUsed / (1024^2)));
                end
                
                obj.GPUEnabled = false;
                obj.IsInitialized = false;
                obj.Logger.Log('info', 'GPU资源清理完成');
                
            catch ME
                obj.Logger.LogException(ME, 'GPUManager.Cleanup');
            end
        end
        
        function stats = GetGPUStats(obj)
            % 获取GPU统计信息
            stats = struct();
            
            if obj.GPUAvailable
                stats.device = obj.GPUDevice;
                stats.total_memory = obj.GPUDevice.TotalMemory;
                stats.available_memory = obj.GPUDevice.AvailableMemory;
                stats.used_memory = obj.GPUDevice.TotalMemory - obj.GPUDevice.AvailableMemory;
                stats.memory_utilization = stats.used_memory / stats.total_memory * 100;
                stats.transfer_count = obj.TransferCount;
                stats.cumulative_transfer = obj.GPUMemoryUsed;
            else
                stats.device = 'None';
                stats.available = false;
            end
        end
    end
    
    methods (Access = private)
        function CheckGPUAvailability(obj)
            % 检查GPU可用性
            try
                % 检查GPU函数是否存在
                if exist('gpuArray', 'file') == 2 && gpuDeviceCount > 0
                    obj.GPUAvailable = true;
                    obj.Logger.Log('info', 'GPU可用');
                else
                    obj.GPUAvailable = false;
                    obj.Logger.Log('info', 'GPU不可用');
                end
            catch ME
                obj.GPUAvailable = false;
                obj.Logger.Log('warning', sprintf('GPU检查失败: %s', ME.message));
            end
        end
        
        function SetupGPUDevice(obj)
            % 设置GPU设备
            try
                % 获取GPU设备
                obj.GPUDevice = gpuDevice();
                
                % 记录GPU信息
                obj.Logger.Log('info', sprintf('GPU设备: %s', obj.GPUDevice.Name));
                obj.Logger.Log('info', sprintf('计算能力: %s', obj.GPUDevice.ComputeCapability));
                obj.Logger.Log('info', sprintf('总内存: %.2f GB', obj.GPUDevice.TotalMemory / 1e9));
                obj.Logger.Log('info', sprintf('可用内存: %.2f GB', obj.GPUDevice.AvailableMemory / 1e9));
                
                % 重置GPU（清理之前的数据）
                reset(obj.GPUDevice);
                
            catch ME
                obj.Logger.LogException(ME, 'SetupGPUDevice');
                obj.GPUAvailable = false;
            end
        end
        
        function CheckGPUMemory(obj)
            % 检查GPU内存状态
            try
                % 获取内存信息
                obj.GPUMemoryInfo = struct();
                obj.GPUMemoryInfo.total = obj.GPUDevice.TotalMemory;
                obj.GPUMemoryInfo.available = obj.GPUDevice.AvailableMemory;
                obj.GPUMemoryInfo.used = obj.GPUMemoryInfo.total - obj.GPUMemoryInfo.available;
                obj.GPUMemoryInfo.utilization = obj.GPUMemoryInfo.used / obj.GPUMemoryInfo.total * 100;
                
                obj.Logger.Log('debug', sprintf('GPU内存使用率: %.1f%%', obj.GPUMemoryInfo.utilization));
                
                % 警告内存使用过高
                if obj.GPUMemoryInfo.utilization > 80
                    obj.Logger.Log('warning', 'GPU内存使用率超过80%');
                end
                
            catch ME
                obj.Logger.LogException(ME, 'CheckGPUMemory');
            end
        end
        
        function SetTransferThreshold(obj)
            % 设置数据传输阈值
            % AMD GPU的传输开销相对较高，需要更高的阈值
            
            % 基础阈值
            base_threshold = obj.Config.GpuMinDataSizeThreshold;
            
            % 根据GPU内存调整阈值
            gpu_memory_gb = obj.GPUDevice.TotalMemory / 1e9;
            
            if gpu_memory_gb < 4
                % 小内存GPU，增加阈值
                obj.TransferThreshold = base_threshold * 2;
            elseif gpu_memory_gb > 8
                % 大内存GPU，可以降低阈值
                obj.TransferThreshold = base_threshold * 0.5;
            else
                obj.TransferThreshold = base_threshold;
            end
            
            obj.Logger.Log('debug', sprintf('GPU传输阈值设置为: %.2f MB', obj.TransferThreshold / (1024^2)));
        end
        
        function should_use = ShouldUseGPU(obj, data_size)
            % 决定是否应该使用GPU
            % 输入:
            %   data_size - 数据大小（字节）
            % 输出:
            %   should_use - 是否应该使用GPU
            
            % 检查基本条件
            if ~obj.GPUEnabled || data_size < obj.TransferThreshold
                should_use = false;
                return;
            end
            
            % 检查GPU内存限制
            gpu_limit = obj.Config.GpuMemoryLimit * obj.GPUDevice.TotalMemory;
            if data_size > gpu_limit
                should_use = false;
                return;
            end
            
            % 检查当前GPU内存使用情况
            obj.CheckGPUMemory();
            required_memory = data_size * 1.5;  % 考虑操作需要的额外内存
            
            if required_memory > obj.GPUMemoryInfo.available
                should_use = false;
                return;
            end
            
            % 对于特定操作类型的启发式判断
            % AMD GPU在某些操作上性能可能不如预期
            if data_size < 50 * 1024 * 1024  % 50MB
                % 小数据集，传输开销可能大于收益
                should_use = false;
                return;
            end
            
            should_use = true;
        end
        
        function MonitorGPUPerformance(obj, operation_name, duration)
            % 监控GPU操作性能
            % 输入:
            %   operation_name - 操作名称
            %   duration - 操作耗时
            
            persistent perf_history;
            if isempty(perf_history)
                perf_history = struct();
            end
            
            % 记录性能数据
            if ~isfield(perf_history, operation_name)
                perf_history.(operation_name) = [];
            end
            
            perf_history.(operation_name) = [perf_history.(operation_name), duration];
            
            % 分析性能趋势
            if length(perf_history.(operation_name)) > 10
                recent_perf = perf_history.(operation_name)(end-9:end);
                avg_perf = mean(recent_perf);
                
                obj.Logger.Log('debug', sprintf('GPU操作 %s 平均耗时: %.4f秒', operation_name, avg_perf));
                
                % 如果性能下降，提出警告
                if length(perf_history.(operation_name)) > 20
                    older_perf = mean(perf_history.(operation_name)(end-19:end-10));
                    if avg_perf > older_perf * 1.5
                        obj.Logger.Log('warning', sprintf('GPU操作 %s 性能下降', operation_name));
                    end
                end
            end
        end
    end
end