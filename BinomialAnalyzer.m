classdef BinomialAnalyzer < handle
    % 二元逻辑回归分析器主类
    % 封装所有分析功能，提供统一的接口
    
    properties (Access = private)
        Config     % 配置对象
        Logger     % 日志对象
        Data       % 数据管理对象
        Parallel   % 并行管理对象
        GPU        % GPU管理对象
        Results    % 结果存储
        StartTime  % 分析开始时间
    end
    
    methods (Access = public)
        function obj = BinomialAnalyzer(varargin)
            % 构造函数
            % 参数可以是:
            %   - 配置文件路径
            %   - 配置对象
            %   - 空（使用默认配置）
            
            % 开始计时
            obj.StartTime = tic;
            
            % 初始化配置
            if nargin == 0
                obj.Config = BinomialConfig();
            elseif isa(varargin{1}, 'BinomialConfig')
                obj.Config = varargin{1};
            elseif ischar(varargin{1}) || isstring(varargin{1})
                obj.Config = BinomialConfig();
                obj.Config.LoadConfig(varargin{1});
            else
                error('无效的输入参数类型');
            end
            
            % 初始化日志系统
            log_file = fullfile(obj.Config.LogDirectory, 'analysis.log');
            obj.Logger = BinomialLogger(log_file, obj.Config.LogLevel);
            
            % 记录系统信息
            obj.LogSystemInfo();
            
            % 初始化其他组件
            obj.InitializeComponents();
            
            % 初始化结果存储
            obj.Results = struct();
        end
        
        function delete(obj)
            % 析构函数：清理资源
            if ~isempty(obj.Parallel)
                obj.Parallel.Cleanup();
            end
            
            if ~isempty(obj.GPU)
                obj.GPU.Cleanup();
            end
            
            total_time = toc(obj.StartTime);
            obj.Logger.Log('info', sprintf('分析完成，总耗时: %.2f秒', total_time));
        end
        
        function RunAnalysis(obj, data_file)
            % 运行完整分析流程
            % 输入:
            %   data_file - 数据文件路径
            
            try
                % 创建主要分析阶段
                obj.Logger.CreateSection('开始二元逻辑回归分析');
                
                % 1. 数据加载和预处理
                obj.LoadAndPreprocessData(data_file);
                
                % 2. 变量准备
                obj.PrepareVariables();
                
                % 3. 多重共线性检查
                obj.CheckMulticollinearity();
                
                % 4. 变量相关性分析
                obj.AnalyzeVariableCorrelations();
                
                % 5. Bootstrap抽样
                obj.GenerateBootstrapSamples();
                
                % 6. K折交叉验证
                obj.PerformCrossValidation();
                
                % 7. 变量选择
                obj.PerformVariableSelection();
                
                % 8. 系数稳定性监控
                obj.MonitorCoefficientStability();
                
                % 9. 参数统计分析
                obj.CalculateParameterStatistics();
                
                % 10. 变量贡献分析
                obj.EvaluateVariableContribution();
                
                % 11. 残差分析
                obj.PerformResidualAnalysis();
                
                % 12. 创建可视化
                obj.CreateVisualizations();
                
                % 13. 保存结果
                obj.SaveResults();
                
                % 14. 生成报告
                obj.GenerateReport();
                
                obj.Logger.CreateSection('分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'RunAnalysis');
                rethrow(ME);
            end
        end
        
        function CustomAnalysis(obj, data_file, analysis_steps)
            % 运行自定义分析步骤
            % 输入:
            %   data_file - 数据文件路径
            %   analysis_steps - 要执行的分析步骤数组
            
            try
                obj.Logger.CreateSection('开始自定义分析');
                
                % 总是需要先加载数据
                if ~isfield(obj.Results, 'data_raw')
                    obj.LoadAndPreprocessData(data_file);
                end
                
                % 执行指定的分析步骤
                for i = 1:length(analysis_steps)
                    step = analysis_steps{i};
                    
                    switch lower(step)
                        case 'preprocess'
                            obj.PrepareVariables();
                        case 'multicollinearity'
                            obj.CheckMulticollinearity();
                        case 'correlation'
                            obj.AnalyzeVariableCorrelations();
                        case 'bootstrap'
                            obj.GenerateBootstrapSamples();
                        case 'crossvalidation'
                            obj.PerformCrossValidation();
                        case 'variableselection'
                            obj.PerformVariableSelection();
                        case 'coefficientstability'
                            obj.MonitorCoefficientStability();
                        case 'parameterstats'
                            obj.CalculateParameterStatistics();
                        case 'contribution'
                            obj.EvaluateVariableContribution();
                        case 'residual'
                            obj.PerformResidualAnalysis();
                        case 'visualization'
                            obj.CreateVisualizations();
                        case 'save'
                            obj.SaveResults();
                        case 'report'
                            obj.GenerateReport();
                        otherwise
                            obj.Logger.Log('warning', sprintf('未知的分析步骤: %s', step));
                    end
                end
                
                obj.Logger.CreateSection('自定义分析完成');
                
            catch ME
                obj.Logger.LogException(ME, 'CustomAnalysis');
                rethrow(ME);
            end
        end
        
        function result = GetResults(obj, component)
            % 获取特定组件的结果
            % 输入:
            %   component - 组件名称（可选，默认返回所有结果）
            
            if nargin < 2
                result = obj.Results;
            else
                if isfield(obj.Results, component)
                    result = obj.Results.(component);
                else
                    warning('结果中不存在组件: %s', component);
                    result = [];
                end
            end
        end
        
        function config = GetConfig(obj)
            % 获取当前配置
            config = obj.Config;
        end
        
        function logger = GetLogger(obj)
            % 获取日志对象
            logger = obj.Logger;
        end
        
        function ShowProgress(obj, message)
            % 显示进度信息
            obj.Logger.Log('info', message);
        end
    end
    
    methods (Access = private)
        function InitializeComponents(obj)
            % 初始化各个组件
            
            % 初始化数据管理器
            obj.Data = DataManager(obj.Config, obj.Logger);
            
            % 初始化并行计算管理器
            obj.Parallel = ParallelManager(obj.Config, obj.Logger);
            
            % 初始化GPU管理器
            obj.GPU = GPUManager(obj.Config, obj.Logger);
            
            % 验证必要的工具箱
            obj.ValidateToolboxes();
        end
        
        function ValidateToolboxes(obj)
            % 验证必要的MATLAB工具箱
            required_toolboxes = {
                'statistics_toolbox',
                'optimization_toolbox'
            };
            
            for i = 1:length(required_toolboxes)
                if ~license('test', required_toolboxes{i})
                    error('需要安装 %s', required_toolboxes{i});
                end
            end
            
            obj.Logger.Log('info', '所有必要的工具箱验证成功');
        end
        
        function LogSystemInfo(obj)
            % 记录系统信息
            obj.Logger.CreateSection('系统配置信息');
            
            info = obj.Config.SystemInfo;
            obj.Logger.Log('info', sprintf('CPU: %s', info.CPU));
            obj.Logger.Log('info', sprintf('内存: %s', info.Memory));
            obj.Logger.Log('info', sprintf('GPU: %s', info.GPU));
            obj.Logger.Log('info', sprintf('平台: %s', info.Platform));
            obj.Logger.Log('info', sprintf('MATLAB版本: %s', info.MatlabVersion));
            obj.Logger.Log('info', sprintf('逻辑处理器数: %d', info.LogicalProcessors));
            
            % 显示配置
            obj.Config.DisplayConfig();
        end
        
        function LoadAndPreprocessData(obj, data_file)
            % 加载和预处理数据
            obj.Logger.CreateSection('数据加载和预处理');
            
            start_time = tic;
            try
                % 加载数据
                [data_raw, success, msg] = obj.Data.LoadData(data_file);
                if ~success
                    error(msg);
                end
                
                % 预处理数据
                [data_processed, valid_rows] = obj.Data.PreprocessData(data_raw);
                
                % 保存到结果
                obj.Results.data_raw = data_raw;
                obj.Results.data_processed = data_processed;
                obj.Results.valid_rows = valid_rows;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('数据加载和预处理', duration, ...
                    sprintf('样本数: %d, 变量数: %d', size(data_processed, 1), size(data_processed, 2)));
                
            catch ME
                obj.Logger.LogException(ME, '数据加载和预处理');
                rethrow(ME);
            end
        end
        
        function PrepareVariables(obj)
            % 准备自变量和因变量
            obj.Logger.CreateSection('变量准备');
            
            start_time = tic;
            try
                [X, y, var_names, group_means] = obj.Data.PrepareVariables(obj.Results.data_processed);
                
                % 保存到结果
                obj.Results.X = X;
                obj.Results.y = y;
                obj.Results.var_names = var_names;
                obj.Results.group_means = group_means;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('变量准备', duration, ...
                    sprintf('自变量数: %d', size(X, 2)));
                
            catch ME
                obj.Logger.LogException(ME, '变量准备');
                rethrow(ME);
            end
        end
        
        function CheckMulticollinearity(obj)
            % 检查多重共线性
            obj.Logger.CreateSection('多重共线性检查');
            
            start_time = tic;
            try
                collinearity_checker = CollinearityChecker(obj.Config, obj.Logger);
                [X_final, vif_values, removed_vars] = collinearity_checker.Check(obj.Results.X, obj.Results.var_names);
                
                % 保存结果
                obj.Results.X_final = X_final;
                obj.Results.vif_values = vif_values;
                obj.Results.removed_vars = removed_vars;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('多重共线性检查', duration, ...
                    sprintf('最终自变量数: %d', size(X_final, 2)));
                
            catch ME
                obj.Logger.LogException(ME, '多重共线性检查');
                rethrow(ME);
            end
        end
        
        function AnalyzeVariableCorrelations(obj)
            % 分析变量相关性
            obj.Logger.CreateSection('变量相关性分析');
            
            start_time = tic;
            try
                correlation_analyzer = CorrelationAnalyzer(obj.Config, obj.Logger);
                pca_results = correlation_analyzer.Analyze(obj.Results.X_final, obj.Results.var_names(~obj.Results.removed_vars));
                
                % 保存结果
                obj.Results.pca_results = pca_results;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('变量相关性分析', duration);
                
            catch ME
                obj.Logger.LogException(ME, '变量相关性分析');
                rethrow(ME);
            end
        end
        
        function GenerateBootstrapSamples(obj)
            % 生成Bootstrap样本
            obj.Logger.CreateSection('Bootstrap抽样');
            
            start_time = tic;
            try
                bootstrap_sampler = BootstrapSampler(obj.Config, obj.Logger);
                [train_indices, test_indices] = bootstrap_sampler.Sample(obj.Results.y);
                
                % 保存结果
                obj.Results.train_indices = train_indices;
                obj.Results.test_indices = test_indices;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('Bootstrap抽样', duration, ...
                    sprintf('生成了 %d 个训练/测试集', length(train_indices)));
                
            catch ME
                obj.Logger.LogException(ME, 'Bootstrap抽样');
                rethrow(ME);
            end
        end
        
        function PerformCrossValidation(obj)
            % 执行K折交叉验证
            obj.Logger.CreateSection('K折交叉验证');
            
            start_time = tic;
            try
                cv_validator = CrossValidator(obj.Config, obj.Logger);
                cv_results = cv_validator.Validate(obj.Results.X_final, obj.Results.y, obj.Results.var_names(~obj.Results.removed_vars));
                
                % 保存结果
                obj.Results.cv_results = cv_results;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('K折交叉验证', duration, ...
                    sprintf('K=%d', obj.Config.KFolds));
                
            catch ME
                obj.Logger.LogException(ME, 'K折交叉验证');
                rethrow(ME);
            end
        end
        
        function PerformVariableSelection(obj)
            % 执行变量选择
            obj.Logger.CreateSection('变量选择');
            
            start_time = tic;
            try
                variable_selector = VariableSelector(obj.Config, obj.Logger, obj.Parallel);
                results = variable_selector.SelectVariables(obj.Results.X_final, obj.Results.y, ...
                    obj.Results.train_indices, obj.Results.test_indices, obj.Results.var_names(~obj.Results.removed_vars));
                
                % 保存结果
                obj.Results.variable_selection = results;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('变量选择', duration);
                
            catch ME
                obj.Logger.LogException(ME, '变量选择');
                rethrow(ME);
            end
        end
        
        function MonitorCoefficientStability(obj)
            % 监控系数稳定性
            obj.Logger.CreateSection('系数稳定性监控');
            
            start_time = tic;
            try
                stability_monitor = CoefficientStabilityMonitor(obj.Config, obj.Logger);
                coef_stability = stability_monitor.Monitor(obj.Results.variable_selection, ...
                    obj.Config.VariableSelectionMethods, obj.Results.var_names(~obj.Results.removed_vars));
                
                % 保存结果
                obj.Results.coef_stability = coef_stability;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('系数稳定性监控', duration);
                
            catch ME
                obj.Logger.LogException(ME, '系数稳定性监控');
                rethrow(ME);
            end
        end
        
        function CalculateParameterStatistics(obj)
            % 计算参数统计
            obj.Logger.CreateSection('参数统计分析');
            
            start_time = tic;
            try
                param_stats_calculator = ParameterStatisticsCalculator(obj.Config, obj.Logger);
                param_stats = param_stats_calculator.Calculate(obj.Results.variable_selection, ...
                    obj.Config.VariableSelectionMethods, obj.Results.var_names(~obj.Results.removed_vars));
                
                % 保存结果
                obj.Results.param_stats = param_stats;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('参数统计分析', duration);
                
            catch ME
                obj.Logger.LogException(ME, '参数统计分析');
                rethrow(ME);
            end
        end
        
        function EvaluateVariableContribution(obj)
            % 评估变量贡献
            obj.Logger.CreateSection('变量贡献分析');
            
            start_time = tic;
            try
                contribution_evaluator = VariableContributionEvaluator(obj.Config, obj.Logger);
                var_contribution = contribution_evaluator.Evaluate(obj.Results.X_final, obj.Results.y, ...
                    obj.Results.variable_selection, obj.Config.VariableSelectionMethods, obj.Results.var_names(~obj.Results.removed_vars));
                
                % 保存结果
                obj.Results.var_contribution = var_contribution;
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('变量贡献分析', duration);
                
            catch ME
                obj.Logger.LogException(ME, '变量贡献分析');
                rethrow(ME);
            end
        end
        
        function PerformResidualAnalysis(obj)
            % 执行残差分析
            obj.Logger.CreateSection('残差分析');
            
            start_time = tic;
            try
                residual_analyzer = ResidualAnalyzer(obj.Config, obj.Logger);
                residual_analyzer.Analyze(obj.Results.variable_selection, obj.Config.VariableSelectionMethods);
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('残差分析', duration);
                
            catch ME
                obj.Logger.LogException(ME, '残差分析');
                rethrow(ME);
            end
        end
        
        function CreateVisualizations(obj)
            % 创建可视化图表
            obj.Logger.CreateSection('创建可视化图表');
            
            start_time = tic;
            try
                visualizer = ResultVisualizer(obj.Config, obj.Logger);
                visualizer.CreateAllVisualization(obj.Results, obj.Config.VariableSelectionMethods);
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('可视化创建', duration);
                
            catch ME
                obj.Logger.LogException(ME, '创建可视化图表');
                rethrow(ME);
            end
        end
        
        function SaveResults(obj)
            % 保存结果
            obj.Logger.CreateSection('保存分析结果');
            
            start_time = tic;
            try
                result_saver = ResultSaver(obj.Config, obj.Logger);
                result_saver.SaveAll(obj.Results, obj.Config.VariableSelectionMethods);
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('结果保存', duration);
                
            catch ME
                obj.Logger.LogException(ME, '保存结果');
                rethrow(ME);
            end
        end
        
        function GenerateReport(obj)
            % 生成分析报告
            obj.Logger.CreateSection('生成分析报告');
            
            start_time = tic;
            try
                report_generator = ReportGenerator(obj.Config, obj.Logger);
                report_generator.Generate(obj.Results, obj.Config.VariableSelectionMethods);
                
                duration = toc(start_time);
                obj.Logger.LogPerformance('报告生成', duration);
                
            catch ME
                obj.Logger.LogException(ME, '生成报告');
                rethrow(ME);
            end
        end
    end
end