function [results, models] = Binomial(X, y, varargin)
% Binomial - 高级统计分析主函数
%
% 该函数是Binomial分析系统的主入口，集成了所有模块的功能，
% 可进行预处理、变量选择、模型构建、统计分析、可视化等操作。
%
% 语法:
%   [results, models] = Binomial(X, y)
%   [results, models] = Binomial(X, y, 'Name', Value, ...)
%
% 必需输入参数:
%   X - 自变量矩阵 [n x p]，其中n是观测数，p是变量数
%   y - 因变量向量 [n x 1]
%
% 可选名值对参数:
%   'Config' - BinomialConfig对象或配置文件路径
%   'LogLevel' - 日志级别 {'DEBUG', 'INFO', 'WARN', 'ERROR'}
%   'LogFile' - 日志文件路径
%   'PreProcess' - 预处理选项 {'standardize', 'normalize', 'robust', 'none'}
%   'VariableNames' - 变量名称元胞数组，长度为p
%   'Intercept' - 是否包含截距项 {true, false}
%   'MethodVS' - 变量选择方法 {'stepwise', 'lasso', 'elasticnet', 'genetic', 'none'}
%   'ModelType' - 模型类型 {'linear', 'logistic', 'poisson', 'robust'}
%   'CVFolds' - 交叉验证折数，默认为5
%   'Bootstrap' - 自助法迭代次数，默认为0（不使用）
%   'Alpha' - 显著性水平，默认为0.05
%   'TestSize' - 测试集比例，默认为0.2
%   'RandomSeed' - 随机数种子，默认为当前时间戳
%   'GPU' - 是否使用GPU计算 {true, false}
%   'Parallel' - 是否使用并行计算 {true, false}
%   'NumWorkers' - 并行计算的工作进程数
%   'OutputDir' - 输出文件保存目录
%   'GenerateReport' - 是否生成分析报告 {true, false}
%   'ReportFormat' - 报告格式 {'html', 'pdf', 'markdown', 'all'}
%   'SaveResults' - 是否保存结果 {true, false}
%   'SaveFormat' - 结果保存格式 {'mat', 'csv', 'excel', 'all'}
%   'Verbose' - 是否显示详细信息 {true, false}
%
% 输出参数:
%   results - 分析结果结构体，包含所有统计分析结果
%   models - 模型结构体数组，包含所有构建的模型
%
% 示例:
%   % 基本用法
%   [results, models] = Binomial(X, y);
%
%   % 高级用法
%   [results, models] = Binomial(X, y, 'PreProcess', 'standardize', ...
%       'MethodVS', 'lasso', 'CVFolds', 10, 'Bootstrap', 1000, ...
%       'GenerateReport', true, 'SaveResults', true);
%
% 参见:
%   BinomialConfig, BinomialLogger, BinomialAnalyzer, VariableSelector, 
%   CrossValidator, BootstrapSampler, ResultVisualizer, ResultSaver, 
%   ReportGenerator

% 初始化基础系统
tidx = tic;

% 处理输入参数
p = inputParser;
p.addRequired('X', @(x) isnumeric(x) && ismatrix(x));
p.addRequired('y', @(x) isnumeric(x) && (isvector(x) || ismatrix(x) && size(x, 2) == 1));
p.addParameter('Config', [], @(x) isempty(x) || isa(x, 'BinomialConfig') || ischar(x));
p.addParameter('LogLevel', 'INFO', @ischar);
p.addParameter('LogFile', '', @ischar);
p.addParameter('PreProcess', 'standardize', @ischar);
p.addParameter('VariableNames', {}, @iscellstr);
p.addParameter('Intercept', true, @islogical);
p.addParameter('MethodVS', 'none', @ischar);
p.addParameter('ModelType', 'linear', @ischar);
p.addParameter('CVFolds', 5, @(x) isnumeric(x) && isscalar(x) && x > 1);
p.addParameter('Bootstrap', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('Alpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
p.addParameter('TestSize', 0.2, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 1);
p.addParameter('RandomSeed', round(sum(clock)), @(x) isnumeric(x) && isscalar(x));
p.addParameter('GPU', false, @islogical);
p.addParameter('Parallel', false, @islogical);
p.addParameter('NumWorkers', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('OutputDir', pwd, @ischar);
p.addParameter('GenerateReport', false, @islogical);
p.addParameter('ReportFormat', 'html', @ischar);
p.addParameter('SaveResults', false, @islogical);
p.addParameter('SaveFormat', 'mat', @ischar);
p.addParameter('Verbose', true, @islogical);

% 解析输入参数
p.parse(X, y, varargin{:});
opts = p.Results;

% 设置随机数种子
rng(opts.RandomSeed);

% 创建配置对象
if isempty(opts.Config)
    config = BinomialConfig();
else
    if ischar(opts.Config)
        % 从配置文件加载
        config = BinomialConfig(opts.Config);
    else
        % 使用提供的配置对象
        config = opts.Config;
    end
end

% 更新配置（命令行参数优先级高于配置文件）
params = p.Results;
paramNames = fieldnames(params);
for i = 1:length(paramNames)
    paramName = paramNames{i};
    if ~ismember(paramName, {'X', 'y', 'Config'}) && ~isempty(params.(paramName))
        if isprop(config, paramName)
            config.(paramName) = params.(paramName);
        else
            warning('未知的配置参数: %s', paramName);
        end
    end
end

% 初始化日志系统
logger = BinomialLogger.getLogger('Binomial');
logger.setLevel(opts.LogLevel);
if ~isempty(opts.LogFile)
    logger.addFileHandler(opts.LogFile);
end

logger.info('Binomial分析系统 v1.0.0');
logger.info('分析开始时间: %s', datestr(now));

% 检查输入数据
[n, p] = size(X);
if length(y) ~= n
    error('输入错误: X和y的样本数不匹配，X有%d行，而y有%d个元素。', n, length(y));
end

logger.info('输入数据: %d 个观测值, %d 个变量', n, p);

% 确保y是列向量
y = y(:);

% 变量名称处理
if isempty(opts.VariableNames)
    variableNames = arrayfun(@(i) sprintf('X%d', i), 1:p, 'UniformOutput', false);
else
    if length(opts.VariableNames) ~= p
        warning('变量名称数量与变量数不匹配，使用默认名称');
        variableNames = arrayfun(@(i) sprintf('X%d', i), 1:p, 'UniformOutput', false);
    else
        variableNames = opts.VariableNames;
    end
end

% 输出目录处理
if ~exist(opts.OutputDir, 'dir')
    [success, ~] = mkdir(opts.OutputDir);
    if ~success
        logger.warn('无法创建输出目录 %s，将使用当前目录', opts.OutputDir);
        opts.OutputDir = pwd;
    end
end

% 初始化模块对象
try
    % 数据管理模块
    dataManager = DataManager(config, logger);
    
    % GPU管理模块
    gpuManager = GPUManager(config, logger);
    if opts.GPU
        if gpuManager.isGPUAvailable()
            logger.info('GPU可用，将使用GPU加速计算');
        else
            logger.warn('请求GPU计算，但GPU不可用，将使用CPU');
            opts.GPU = false;
        end
    end
    
    % 并行计算管理模块
    parallelManager = ParallelManager(config, logger);
    if opts.Parallel
        if parallelManager.isParallelAvailable()
            if opts.NumWorkers > 0
                parallelManager.startParallel(opts.NumWorkers);
            else
                parallelManager.startParallel();
            end
            logger.info('已启动并行计算池，使用 %d 个工作进程', parallelManager.getNumWorkers());
        else
            logger.warn('请求并行计算，但并行工具箱不可用，将使用串行计算');
            opts.Parallel = false;
        end
    end
    
    % 设置处理后的数据
    dataManager.setData(X, y, variableNames);
    
    % 数据预处理
    switch lower(opts.PreProcess)
        case 'standardize'
            dataManager.standardize();
            logger.info('数据已标准化 (均值=0, 标准差=1)');
        case 'normalize'
            dataManager.normalize();
            logger.info('数据已归一化 (范围 [0,1])');
        case 'robust'
            dataManager.robustScale();
            logger.info('数据已稳健缩放 (中位数=0, IQR=1)');
        case 'none'
            logger.info('跳过数据预处理');
        otherwise
            logger.warn('未知的预处理方法 "%s"，跳过预处理', opts.PreProcess);
    end
    
    % 多重共线性检查
    colChecker = CollinearityChecker(config, logger);
    colChecker.check(dataManager.getX());
    
    % 获取预处理后的数据
    X_proc = dataManager.getX();
    y_proc = dataManager.getY();
    
    % 数据划分（如果需要）
    if opts.TestSize > 0
        dataManager.splitData(opts.TestSize);
        X_train = dataManager.getTrainX();
        y_train = dataManager.getTrainY();
        X_test = dataManager.getTestX();
        y_test = dataManager.getTestY();
        logger.info('数据已划分: 训练集 %d 个样本, 测试集 %d 个样本', ...
            size(X_train, 1), size(X_test, 1));
    else
        X_train = X_proc;
        y_train = y_proc;
        X_test = [];
        y_test = [];
    end
    
    % 变量选择
    variableSelector = VariableSelector(logger);
    switch lower(opts.MethodVS)
        case 'stepwise'
            selectedVars = variableSelector.runStepwise(X_train, y_train);
            logger.info('已完成逐步回归变量选择，选择了 %d 个变量', length(selectedVars));
        case 'lasso'
            selectedVars = variableSelector.runLasso(X_train, y_train);
            logger.info('已完成LASSO变量选择，选择了 %d 个变量', length(selectedVars));
        case 'elasticnet'
            selectedVars = variableSelector.runElasticNet(X_train, y_train);
            logger.info('已完成Elastic Net变量选择，选择了 %d 个变量', length(selectedVars));
        case 'genetic'
            selectedVars = variableSelector.runGeneticAlgorithm(X_train, y_train);
            logger.info('已完成遗传算法变量选择，选择了 %d 个变量', length(selectedVars));
        case 'none'
            selectedVars = 1:size(X_train, 2);
            logger.info('未执行变量选择，使用所有 %d 个变量', length(selectedVars));
        otherwise
            logger.warn('未知的变量选择方法 "%s"，使用所有变量', opts.MethodVS);
            selectedVars = 1:size(X_train, 2);
    end
    
    % 更新数据集，只使用选中的变量
    X_selected = X_train(:, selectedVars);
    variableNamesSelected = variableNames(selectedVars);
    if ~isempty(X_test)
        X_test_selected = X_test(:, selectedVars);
    else
        X_test_selected = [];
    end
    
    % 相关性分析
    corrAnalyzer = CorrelationAnalyzer(logger);
    corrAnalyzer.analyze(X_selected, y_train, variableNamesSelected);
    
    % 主分析器
    analyzer = BinomialAnalyzer(logger);
    
    % Bootstrap采样（如果需要）
    if opts.Bootstrap > 0
        bootstrapper = BootstrapSampler(logger);
        bootstrapper.setSamples(opts.Bootstrap);
        bootstrapper.generateSamples(X_selected, y_train);
        logger.info('已完成Bootstrap采样，生成 %d 个样本', opts.Bootstrap);
    end
    
    % 交叉验证
    if opts.CVFolds > 1
        crossValidator = CrossValidator(logger);
        crossValidator.setFolds(opts.CVFolds);
        crossValidator.generateFolds(X_selected, y_train);
        logger.info('已完成 %d 折交叉验证划分', opts.CVFolds);
    end
    
    % 模型拟合
    switch lower(opts.ModelType)
        case 'linear'
            % 线性回归模型
            model = analyzer.fitLinearModel(X_selected, y_train, opts.Intercept);
            logger.info('已拟合线性回归模型: R² = %.4f', model.rsquared);
            
            % 如果启用了交叉验证
            if opts.CVFolds > 1
                cvResults = crossValidator.validate(@(X, y) analyzer.fitLinearModel(X, y, opts.Intercept));
                logger.info('交叉验证完成: 平均 R² = %.4f, 平均 RMSE = %.4f', ...
                    mean(cvResults.rsquared), mean(cvResults.rmse));
            end
            
            % 如果启用了Bootstrap
            if opts.Bootstrap > 0
                bootResults = bootstrapper.analyze(@(X, y) analyzer.fitLinearModel(X, y, opts.Intercept));
                logger.info('Bootstrap分析完成: 系数标准误 = %.4f, 平均 R² = %.4f', ...
                    mean(bootResults.stdErrors), mean(bootResults.rsquared));
            end
            
        case 'logistic'
            % 逻辑回归模型
            model = analyzer.fitLogisticModel(X_selected, y_train, opts.Intercept);
            logger.info('已拟合逻辑回归模型: 准确率 = %.4f', model.accuracy);
            
            % 如果启用了交叉验证
            if opts.CVFolds > 1
                cvResults = crossValidator.validate(@(X, y) analyzer.fitLogisticModel(X, y, opts.Intercept));
                logger.info('交叉验证完成: 平均准确率 = %.4f, 平均 AUC = %.4f', ...
                    mean(cvResults.accuracy), mean(cvResults.auc));
            end
            
            % 如果启用了Bootstrap
            if opts.Bootstrap > 0
                bootResults = bootstrapper.analyze(@(X, y) analyzer.fitLogisticModel(X, y, opts.Intercept));
                logger.info('Bootstrap分析完成: 系数标准误 = %.4f, 平均准确率 = %.4f', ...
                    mean(bootResults.stdErrors), mean(bootResults.accuracy));
            end
            
        case 'poisson'
            % 泊松回归模型
            model = analyzer.fitPoissonModel(X_selected, y_train, opts.Intercept);
            logger.info('已拟合泊松回归模型: 偏差 = %.4f', model.deviance);
            
            % 如果启用了交叉验证
            if opts.CVFolds > 1
                cvResults = crossValidator.validate(@(X, y) analyzer.fitPoissonModel(X, y, opts.Intercept));
                logger.info('交叉验证完成: 平均偏差 = %.4f', mean(cvResults.deviance));
            end
            
            % 如果启用了Bootstrap
            if opts.Bootstrap > 0
                bootResults = bootstrapper.analyze(@(X, y) analyzer.fitPoissonModel(X, y, opts.Intercept));
                logger.info('Bootstrap分析完成: 系数标准误 = %.4f', mean(bootResults.stdErrors));
            end
            
        case 'robust'
            % 稳健回归模型
            model = analyzer.fitRobustModel(X_selected, y_train, opts.Intercept);
            logger.info('已拟合稳健回归模型: R² = %.4f', model.rsquared);
            
            % 如果启用了交叉验证
            if opts.CVFolds > 1
                cvResults = crossValidator.validate(@(X, y) analyzer.fitRobustModel(X, y, opts.Intercept));
                logger.info('交叉验证完成: 平均 R² = %.4f, 平均 RMSE = %.4f', ...
                    mean(cvResults.rsquared), mean(cvResults.rmse));
            end
            
            % 如果启用了Bootstrap
            if opts.Bootstrap > 0
                bootResults = bootstrapper.analyze(@(X, y) analyzer.fitRobustModel(X, y, opts.Intercept));
                logger.info('Bootstrap分析完成: 系数标准误 = %.4f, 平均 R² = %.4f', ...
                    mean(bootResults.stdErrors), mean(bootResults.rsquared));
            end
            
        otherwise
            error('不支持的模型类型: %s', opts.ModelType);
    end
    
    % 系数稳定性监控
    coefMonitor = CoefficientStabilityMonitor(logger);
    if opts.Bootstrap > 0
        coefMonitor.initialize(model.coefficients, variableNamesSelected);
        for i = 1:length(bootResults.coefficients)
            coefMonitor.addCoefficients(bootResults.coefficients{i});
        end
        stabilityResults = coefMonitor.analyzeStability();
        logger.info('系数稳定性分析完成: 平均稳定性得分 = %.4f, 不稳定系数比例 = %.2f%%', ...
            stabilityResults.meanStabilityScore, stabilityResults.percentUnstable);
    end
    
    % 参数统计分析
    paramStats = ParameterStatisticsCalculator(logger);
    paramStats.calculate(X_selected, y_train, model.coefficients, variableNamesSelected);
    paramReport = paramStats.generateSummaryReport();
    logger.info('参数统计分析完成: 模型 R² = %.4f, 调整 R² = %.4f', ...
        paramReport.ModelQuality.R2, paramReport.ModelQuality.AdjustedR2);
    
    % 变量贡献评估
    varContrib = VariableContributionEvaluator(logger);
    varContrib.evaluate(X_selected, y_train, model.coefficients, variableNamesSelected);
    contribReport = varContrib.generateContributionReport();
    logger.info('变量贡献评估完成');
    
    % 残差分析
    residAnalyzer = ResidualAnalyzer(logger);
    residAnalyzer.analyze(X_selected, y_train, model.coefficients, model.residuals);
    residReport = residAnalyzer.generateResidualReport();
    logger.info('残差分析完成: 正态性(p值) = %.4f', residReport.normalityTests.jarqueBera.pValue);
    
    % 模型评估（在测试集上，如果有）
    if ~isempty(X_test_selected) && ~isempty(y_test)
        switch lower(opts.ModelType)
            case {'linear', 'robust'}
                y_pred = X_test_selected * model.coefficients;
                if opts.Intercept && isfield(model, 'intercept')
                    y_pred = y_pred + model.intercept;
                end
                test_rmse = sqrt(mean((y_test - y_pred).^2));
                test_r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);
                logger.info('测试集评估: RMSE = %.4f, R² = %.4f', test_rmse, test_r2);
                
                % 将测试集结果添加到模型中
                model.test_rmse = test_rmse;
                model.test_r2 = test_r2;
                model.test_predictions = y_pred;
                
            case 'logistic'
                probs = 1 ./ (1 + exp(-(X_test_selected * model.coefficients)));
                if opts.Intercept && isfield(model, 'intercept')
                    probs = 1 ./ (1 + exp(-(X_test_selected * model.coefficients + model.intercept)));
                end
                y_pred = probs > 0.5;
                
                test_accuracy = mean(y_pred == y_test);
                
                % 计算AUC
                [~, ~, ~, test_auc] = perfcurve(y_test, probs, 1);
                
                logger.info('测试集评估: 准确率 = %.4f, AUC = %.4f', test_accuracy, test_auc);
                
                % 将测试集结果添加到模型中
                model.test_accuracy = test_accuracy;
                model.test_auc = test_auc;
                model.test_predictions = y_pred;
                model.test_probabilities = probs;
                
            case 'poisson'
                log_mu = X_test_selected * model.coefficients;
                if opts.Intercept && isfield(model, 'intercept')
                    log_mu = log_mu + model.intercept;
                end
                mu = exp(log_mu);
                
                test_deviance = 2 * sum(y_test .* log(y_test ./ mu) - (y_test - mu));
                
                logger.info('测试集评估: 偏差 = %.4f', test_deviance);
                
                % 将测试集结果添加到模型中
                model.test_deviance = test_deviance;
                model.test_predictions = mu;
        end
    end
    
    % 结果可视化
    if config.getParameter('Visualize')
        visualizer = ResultVisualizer(logger);
        
        % 设置保存路径
        visualizer.setSavePath(opts.OutputDir);
        
        % 准备结果数据
        vizData = struct();
        vizData.coefficients = model.coefficients;
        vizData.standardErrors = paramReport.ParameterTable.StdError;
        vizData.variableNames = variableNamesSelected;
        
        if isfield(model, 'residuals')
            vizData.residuals = model.residuals;
        end
        
        if isfield(model, 'fitted')
            vizData.observed = y_train;
            vizData.predicted = model.fitted;
        end
        
        if isfield(paramReport, 'ModelQuality')
            vizData.modelQuality = paramReport.ModelQuality;
        end
        
        if isfield(residReport, 'diagnostics')
            vizData.diagnostics = residReport.diagnostics;
        end
        
        if isfield(residReport, 'outliers')
            vizData.outliers = residReport.outliers;
        end
        
        % 设置模型结果数据
        visualizer.setModelResults(vizData);
        
        % 生成图形
        visualizer.plotCoefficientEstimates(model.coefficients, paramReport.ParameterTable.StdError, variableNamesSelected);
        visualizer.plotVariableImportance(abs(varContrib.standardizedCoefficients), variableNamesSelected);
        
        if isfield(model, 'fitted') && isfield(model, 'residuals')
            visualizer.plotModelFit(y_train, model.fitted);
        end
        
        if isfield(residReport, 'residuals')
            visualizer.plotResidualDiagnostics(residReport.residuals, residReport.leverage, residReport.cookDistance);
        end
        
        if isfield(vizData, 'observed') && isfield(vizData, 'predicted')
            visualizer.plotModelFit(vizData.observed, vizData.predicted);
        end
        
        if isfield(corrAnalyzer, 'correlationMatrix')
            visualizer.plotCorrelationHeatmap(corrAnalyzer.correlationMatrix, variableNamesSelected);
        end
        
        % 保存所有图形
        visualizer.saveAllFigures('png');
        logger.info('已生成并保存图形文件');
    end
    
    % 保存结果
    if opts.SaveResults
        resultSaver = ResultSaver(logger);
        resultSaver.setSaveDirectory(opts.OutputDir);
        
        % 准备结果数据
        resultsData = struct();
        resultsData.model = model;
        resultsData.modelType = opts.ModelType;
        resultsData.variableNames = variableNamesSelected;
        resultsData.selectedVariables = selectedVars;
        resultsData.coefficients = model.coefficients;
        
        if opts.CVFolds > 1
            resultsData.crossValidation = cvResults;
        end
        
        if opts.Bootstrap > 0
            resultsData.bootstrap = bootResults;
            resultsData.stabilityResults = stabilityResults;
        end
        
        resultsData.parameterStats = paramReport;
        resultsData.variableContribution = contribReport;
        resultsData.residualAnalysis = residReport;
        
        % 添加原始和预处理数据
        resultsData.rawData = struct('X', X, 'y', y, 'variableNames', variableNames);
        resultsData.processedData = struct('X', X_proc, 'y', y_proc);
        
        if ~isempty(X_test)
            resultsData.testData = struct('X', X_test, 'y', y_test);
            if isfield(model, 'test_predictions')
                resultsData.testPredictions = model.test_predictions;
            end
        end
        
        % 保存不同格式的结果
        formats = strsplit(lower(opts.SaveFormat), ',');
        
        for i = 1:length(formats)
            format = strtrim(formats{i});
            switch format
                case 'mat'
                    resultSaver.saveToMAT(resultsData);
                case 'csv'
                    % 将主要结果保存为CSV
                    coefTable = array2table([model.coefficients, paramReport.ParameterTable.StdError, ...
                        paramReport.ParameterTable.tStat, paramReport.ParameterTable.pValue], ...
                        'VariableNames', {'Coefficient', 'StdError', 'tStat', 'pValue'});
                    coefTable.Variable = variableNamesSelected';
                    resultSaver.saveToCSV(coefTable, [resultSaver.saveName '_coefficients']);
                    
                    % 保存拟合值和残差
                    if isfield(model, 'fitted') && isfield(model, 'residuals')
                        fitResidTable = array2table([y_train, model.fitted, model.residuals], ...
                            'VariableNames', {'Observed', 'Fitted', 'Residuals'});
                        resultSaver.saveToCSV(fitResidTable, [resultSaver.saveName '_fit_residuals']);
                    end
                case 'excel'
                    % 将多个表格保存在同一个Excel文件中
                    resultSaver.saveToExcel(resultsData);
                case 'all'
                    resultSaver.saveToMAT(resultsData);
                    
                    coefTable = array2table([model.coefficients, paramReport.ParameterTable.StdError, ...
                        paramReport.ParameterTable.tStat, paramReport.ParameterTable.pValue], ...
                        'VariableNames', {'Coefficient', 'StdError', 'tStat', 'pValue'});
                    coefTable.Variable = variableNamesSelected';
                    resultSaver.saveToCSV(coefTable, [resultSaver.saveName '_coefficients']);
                    
                    if isfield(model, 'fitted') && isfield(model, 'residuals')
                        fitResidTable = array2table([y_train, model.fitted, model.residuals], ...
                            'VariableNames', {'Observed', 'Fitted', 'Residuals'});
                        resultSaver.saveToCSV(fitResidTable, [resultSaver.saveName '_fit_residuals']);
                    end
                    
                    resultSaver.saveToExcel(resultsData);
                    resultSaver.saveToJSON(resultsData);
            end
        end
        
        logger.info('已保存分析结果文件');
    end
    
    % 生成报告
    if opts.GenerateReport
        reportGen = ReportGenerator(logger);
        reportGen.setReportDirectory(opts.OutputDir);
        
        % 准备报告数据
        reportData = struct();
        reportData.modelInfo = struct(...
            'modelType', opts.ModelType, ...
            'observations', size(X_train, 1), ...
            'variables', length(selectedVars), ...
            'quality', struct());
        
        % 添加模型质量指标
        if isfield(paramReport, 'ModelQuality')
            reportData.modelInfo.quality = paramReport.ModelQuality;
        end
        
        % 添加系数信息
        reportData.coefficients = model.coefficients;
        reportData.standardErrors = paramReport.ParameterTable.StdError;
        reportData.variableNames = variableNamesSelected;
        reportData.observations = size(X_train, 1);
        
        % 添加变量重要性信息
        if isfield(varContrib, 'standardizedCoefficients')
            reportData.importance = abs(varContrib.standardizedCoefficients);
        end
        
        % 添加诊断信息
        if isfield(residReport, 'normalityTests')
            reportData.diagnostics.normalityTest = struct(...
                'method', 'Jarque-Bera', ...
                'statistic', residReport.normalityTests.jarqueBera.statistic, ...
                'pValue', residReport.normalityTests.jarqueBera.pValue, ...
                'conclusion', conditional(residReport.normalityTests.jarqueBera.isNormal, ...
                '残差符合正态分布', '残差不符合正态分布'));
        end
        
        if isfield(residReport, 'autocorrelationTests')
            reportData.diagnostics.autocorrelationTest = struct(...
                'method', 'Durbin-Watson', ...
                'statistic', residReport.autocorrelationTests.durbinWatson.statistic, ...
                'pValue', residReport.autocorrelationTests.durbinWatson.pValue, ...
                'conclusion', conditional(~residReport.autocorrelationTests.durbinWatson.hasAutocorrelation, ...
                '残差无显著自相关性', '残差存在自相关性'));
        end
        
        if isfield(residReport, 'heteroskedasticityTests')
            reportData.diagnostics.heteroskedasticityTest = struct(...
                'method', 'Breusch-Pagan', ...
                'statistic', residReport.heteroskedasticityTests.breuschPagan.statistic, ...
                'pValue', residReport.heteroskedasticityTests.breuschPagan.pValue, ...
                'conclusion', conditional(~residReport.heteroskedasticityTests.breuschPagan.hasHeteroskedasticity, ...
                '残差方差齐性良好', '残差存在异方差性'));
        end
        
        if isfield(residReport, 'outliers') && isfield(residReport.outliers, 'consensus')
            reportData.diagnostics.outliers = residReport.outliers.consensus;
        end
        
        if isfield(model, 'residuals') && isfield(model, 'fitted')
            reportData.observed = y_train;
            reportData.predicted = model.fitted;
            reportData.residuals = model.residuals;
        end
        
        % 添加结论和建议
        conclusions = {};
        
        % 模型拟合度结论
        if isfield(paramReport, 'ModelQuality') && isfield(paramReport.ModelQuality, 'R2')
            r2 = paramReport.ModelQuality.R2;
            if r2 > 0.8
                conclusions{end+1} = sprintf('模型拟合度非常好 (R² = %.4f)，能很好地解释因变量的变异。', r2);
            elseif r2 > 0.6
                conclusions{end+1} = sprintf('模型拟合度良好 (R² = %.4f)，能较好地解释因变量的变异。', r2);
            elseif r2 > 0.4
                conclusions{end+1} = sprintf('模型拟合度一般 (R² = %.4f)，能解释部分因变量的变异。', r2);
            else
                conclusions{end+1} = sprintf('模型拟合度较差 (R² = %.4f)，解释因变量变异的能力有限。', r2);
            end
        end
        
        % 变量重要性结论
        if ~isempty(contribReport.topVariables.names)
            topVarStr = strjoin(contribReport.topVariables.names(1:min(3, length(contribReport.topVariables.names))), '、');
            conclusions{end+1} = sprintf('最重要的变量是%s，对模型有显著贡献。', topVarStr);
        end
        
        % 残差诊断结论
        if isfield(reportData, 'diagnostics')
            diagnoses = {};
            
            if isfield(reportData.diagnostics, 'normalityTest') && ...
                    ~isfield(reportData.diagnostics.normalityTest, 'isNormal')
                diagnoses{end+1} = '残差不符合正态分布';
            end
            
            if isfield(reportData.diagnostics, 'autocorrelationTest') && ...
                    isfield(reportData.diagnostics.autocorrelationTest, 'hasAutocorrelation') && ...
                    reportData.diagnostics.autocorrelationTest.hasAutocorrelation
                diagnoses{end+1} = '残差存在自相关性';
            end
            
            if isfield(reportData.diagnostics, 'heteroskedasticityTest') && ...
                    isfield(reportData.diagnostics.heteroskedasticityTest, 'hasHeteroskedasticity') && ...
                    reportData.diagnostics.heteroskedasticityTest.hasHeteroskedasticity
                diagnoses{end+1} = '残差存在异方差性';
            end
            
            if isfield(reportData.diagnostics, 'outliers') && ...
                    length(reportData.diagnostics.outliers) > 0
                diagnoses{end+1} = sprintf('存在%d个异常点', length(reportData.diagnostics.outliers));
            end
            
            if ~isempty(diagnoses)
                conclusions{end+1} = sprintf('模型诊断发现以下问题：%s，建议进一步改进模型。', strjoin(diagnoses, '、'));
            else
                conclusions{end+1} = '模型诊断未发现明显问题，残差表现良好。';
            end
        end
        
        % 增加模型使用建议
        if isfield(model, 'coefficients') && ~isempty(model.coefficients)
            conclusions{end+1} = '在实际应用中，应注意模型系数的置信区间和预测的不确定性。';
        end
        
        if opts.Bootstrap > 0 && isfield(stabilityResults, 'percentUnstable') && stabilityResults.percentUnstable > 20
            conclusions{end+1} = sprintf('Bootstrap分析显示%.1f%%的系数不稳定，表明模型对数据变化较敏感。', stabilityResults.percentUnstable);
        end
        
        reportData.conclusions = conclusions;
        
        % 添加警告信息（如果有）
        warnings = {};
        
        if isfield(colChecker, 'vif') && any(colChecker.vif > 10)
            highVifVars = find(colChecker.vif > 10);
            if ~isempty(highVifVars)
                highVifNames = variableNames(highVifVars);
                warnings{end+1} = sprintf('变量%s存在严重多重共线性 (VIF > 10)，可能影响参数估计的稳定性。', strjoin(highVifNames, '、'));
            end
        end
        
        if ~isempty(warnings)
            reportData.warnings = warnings;
        end
        
        % 设置报告数据
        reportGen.setResultData(reportData);
        
        % 根据设置生成报告
        formats = strsplit(lower(opts.ReportFormat), ',');
        
        for i = 1:length(formats)
            format = strtrim(formats{i});
            switch format
                case 'html'
                    reportGen.generateHTMLReport();
                    reportGen.saveHTMLReport();
                case 'pdf'
                    reportGen.generatePDFReport();
                case 'markdown'
                    reportGen.generateMarkdownReport();
                case 'all'
                    reportGen.generateHTMLReport();
                    reportGen.saveHTMLReport();
                    reportGen.generateMarkdownReport();
                    reportGen.generatePDFReport();
            end
        end
        
        logger.info('已生成分析报告');
    end
    
    % 准备返回结果
    results = struct();
    results.model = model;
    results.modelType = opts.ModelType;
    results.variableNames = variableNamesSelected;
    results.selectedVariables = selectedVars;
    results.coefficients = model.coefficients;
    
    if opts.CVFolds > 1
        results.crossValidation = cvResults;
    end
    
    if opts.Bootstrap > 0
        results.bootstrap = bootResults;
        results.stabilityResults = stabilityResults;
    end
    
    results.parameterStats = paramReport;
    results.variableContribution = contribReport;
    results.residualAnalysis = residReport;
    
    % 收集所有模型
    models = model;
    
    % 关闭并行池（如果已启动）
    if opts.Parallel && parallelManager.isPoolOpen()
        parallelManager.stopParallel();
        logger.info('已关闭并行计算池');
    end
    
    % 计算总运行时间
    totalTime = toc(tidx);
    logger.info('分析完成！总运行时间: %.2f 秒', totalTime);
    
catch ME
    % 错误处理
    logger.error('分析过程中发生错误: %s', ME.message);
    logger.error('错误详细信息: %s', getReport(ME));
    
    % 尝试关闭并行池（如果已启动）
    try
        if exist('parallelManager', 'var') && parallelManager.isPoolOpen()
            parallelManager.stopParallel();
            logger.info('已关闭并行计算池');
        end
    catch
        % 忽略关闭并行池时的错误
    end
    
    % 重新抛出错误
    rethrow(ME);
end

end

% 辅助函数
function result = conditional(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end