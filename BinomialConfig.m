classdef BinomialConfig < handle
    % 配置管理类：集中管理所有可调整的参数
    % 使用 handle 类确保配置在整个程序中保持一致
    
    properties (Access = public)
        % 数据预处理参数
        ExcludeRows = [6, 10, 42, 74, 124, 127, 189, 252, 285, 298, 326, 331, 339]
        ReverseItems = [12, 19, 23]
        MaxScore = 5
        TargetColumn = 29
        ValidRows = []
        
        % 变量准备参数
        VariableGroups = {
            [1, 2, 3, 4, 5, 6, 12, 19, 23],  % 组1
            [7, 8, 9],                       % 组2
            [10, 11],                        % 组3
            [13, 14],                        % 组4
            [15, 17, 18, 20, 21],            % 组5
            [22, 24],                        % 组6
            [25, 26],                        % 组7
            [27, 28]                         % 组8
        }
        


        Alpha = 0.05
        Bootstrap = 0
        CVFolds = 5
        GenerateReport = false
        GPU = false
        Intercept = true
        MethodVS = 'none'
        ModelType = 'linear'
        OutputDir = '.'
        Parallel = false
        PreProcess = 'standardize'
        RandomSeed = 42
        ReportFormat = 'html'
        SaveFormat = 'mat'
        SaveResults = false
        TestSize = 0.2
        Verbose = true




        % 多重共线性检查参数
        VifThreshold = 10
        ConditionNumberThreshold = 30
        PcaVarianceThreshold = 95
        
        % Bootstrap参数
        TrainRatio = 0.8
        NumBootstrapSamples = 100
        
        % K折交叉验证参数
        KFolds = 10
        
        % 变量选择参数
        VariableSelectionMethods = {'stepwise', 'lasso', 'ridge', 'elasticnet', 'randomforest'}
        StepwisePEnter = 0.15
        StepwisePRemove = 0.20
        LassoCrossValidationFolds = 10
        LassoAlpha = 1
        RidgeAlpha = 0.001
        ElasticNetAlpha = 0.5
        RandomForestNumTrees = 200
        
        % 模型评估参数
        SignificanceLevel = 0.05
        CoefficientStabilityThreshold = 0.5
        MinimumModelsForAnalysis = 5
        
        % 并行计算参数
        UseParallelComputing = true
        NumWorkers = []  % 自动设置
        NumThreadsPerWorker = 2
        ParallelClusterJobStorageLocation = '/tmp'
        
        % GPU设置
        UseGpu = true
        GpuMemoryLimit = 0.6  % 使用总GPU内存的60%
        GpuMinDataSizeThreshold = 5 * 1024 * 1024  % 5MB
        
        % 可视化参数
        FigureHeight = 600
        FigureWidth = 900
        ExportFormats = {'svg', 'png'}
        ExportDpi = 300
        
        % 日志参数
        LogLevel = 'info'
        LogToFile = true
        LogDirectory = 'results'
        
        % 输出参数
        OutputDirectory = 'results'
        SaveIntermediateResults = true
        
        % 系统信息
        SystemInfo = struct()
    end
    
    methods (Access = public)

        function setParameter(obj, paramName, paramValue)
            % 设置配置参数
            %
            % 参数:
            %   paramName - 参数名称
            %   paramValue - 参数值

            if isprop(obj, paramName)
                obj.(paramName) = paramValue;
            else
                warning('未知的配置参数: %s', paramName);
            end
        end

        function obj = BinomialConfig()
            % 构造函数：初始化配置
            
            % 设置系统信息
            obj.SetSystemInfo();
            
            % 计算有效行
            obj.ValidRows = setdiff(1:375, obj.ExcludeRows);
            
            % 自动设置并行参数
            obj.SetParallelParams();
            
            % 确保输出目录存在
            obj.CreateDirectories();
        end
        
        function SetSystemInfo(obj)
            % 设置系统信息
            obj.SystemInfo.CPU = 'Intel i9-9980HK (8核16线程)';
            obj.SystemInfo.Memory = '64GB RAM';
            obj.SystemInfo.GPU = 'AMD Radeon Pro 5500M 8GB';
            obj.SystemInfo.Platform = computer();
            obj.SystemInfo.MatlabVersion = version();
            obj.SystemInfo.LogicalProcessors = feature('numcores');
        end
        
        function SetParallelParams(obj)
            % 自动设置并行计算参数
            if obj.UseParallelComputing
                logical_processors = feature('numcores');
                % 针对i9-9980HK的优化配置
                if isempty(obj.NumWorkers)
                    obj.NumWorkers = min(8, logical_processors);
                end
                
                % 创建临时存储目录
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                obj.ParallelClusterJobStorageLocation = fullfile(tempdir, ...
                    ['matlab_parallel_', timestamp]);
                
                if ~exist(obj.ParallelClusterJobStorageLocation, 'dir')
                    mkdir(obj.ParallelClusterJobStorageLocation);
                end
            end
        end
        
        function CreateDirectories(obj)
            % 创建必要的目录
            dirs = {
                obj.OutputDirectory,
                fullfile(obj.OutputDirectory, 'figures'),
                fullfile(obj.OutputDirectory, 'csv'),
                fullfile(obj.OutputDirectory, 'reports'),
                obj.LogDirectory
            };
            
            for i = 1:length(dirs)
                if ~exist(dirs{i}, 'dir')
                    mkdir(dirs{i});
                end
            end
        end
        
        function SaveConfig(obj)
            % 保存配置到文件
            config_file = fullfile(obj.OutputDirectory, 'analysis_config.mat');
            config = obj.ToStruct();
            save(config_file, 'config', '-v7.3');
        end
        
        function config_struct = ToStruct(obj)
            % 将配置转换为结构体
            props = properties(obj);
            config_struct = struct();
            
            for i = 1:length(props)
                config_struct.(props{i}) = obj.(props{i});
            end
        end
        
        function LoadConfig(obj, config_file)
            % 从文件加载配置
            if exist(config_file, 'file')
                loaded = load(config_file);
                config = loaded.config;
                
                props = fields(config);
                for i = 1:length(props)
                    if isprop(obj, props{i})
                        obj.(props{i}) = config.(props{i});
                    end
                end
            else
                warning('配置文件不存在: %s', config_file);
            end
        end
        
        function DisplayConfig(obj)
            % 显示当前配置
            fprintf('\n=== 当前分析配置 ===\n');
            
            % 数据处理
            fprintf('\n数据处理参数:\n');
            fprintf('  - 排除行: [%s]\n', num2str(obj.ExcludeRows));
            fprintf('  - 反转项目: [%s]\n', num2str(obj.ReverseItems));
            fprintf('  - 目标列: %d\n', obj.TargetColumn);
            
            % 模型参数
            fprintf('\n模型参数:\n');
            fprintf('  - Bootstrap样本数: %d\n', obj.NumBootstrapSamples);
            fprintf('  - K折验证: %d\n', obj.KFolds);
            fprintf('  - VIF阈值: %.1f\n', obj.VifThreshold);
            
            % 变量选择方法
            fprintf('\n变量选择方法:\n');
            for i = 1:length(obj.VariableSelectionMethods)
                fprintf('  - %s\n', obj.VariableSelectionMethods{i});
            end
            
            % 并行设置
            fprintf('\n并行计算设置:\n');
            fprintf('  - 使用并行: %s\n', mat2str(obj.UseParallelComputing));
            fprintf('  - Worker数量: %d\n', obj.NumWorkers);
            fprintf('  - 每个Worker线程数: %d\n', obj.NumThreadsPerWorker);
            
            % 系统信息
            fprintf('\n系统信息:\n');
            fprintf('  - CPU: %s\n', obj.SystemInfo.CPU);
            fprintf('  - 内存: %s\n', obj.SystemInfo.Memory);
            fprintf('  - GPU: %s\n', obj.SystemInfo.GPU);
            
            fprintf('\n===================\n\n');
        end
    end
    
    methods (Access = private)
        % 私有辅助方法可以在这里添加
    end
end