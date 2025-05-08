classdef ReportGenerator < handle
    % ReportGenerator - 分析报告生成模块
    %
    % 该类负责根据模型分析结果生成详细的分析报告，
    % 支持HTML、PDF等多种格式，包含文本说明、表格和图形。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   resultData - 分析结果数据
    %   reportTitle - 报告标题
    %   reportSubtitle - 报告副标题
    %   author - 报告作者
    %   datestamp - 报告日期
    %   reportDirectory - 报告保存目录
    %   templateDirectory - 模板目录
    %   figures - 报告中包含的图形
    %   htmlReport - HTML 报告内容
    %   reportFiles - 已生成的报告文件信息
    
    properties
        logger               % 日志记录器
        resultData           % 分析结果数据
        reportTitle          % 报告标题
        reportSubtitle       % 报告副标题
        author               % 报告作者
        datestamp            % 报告日期
        reportDirectory      % 报告保存目录
        templateDirectory    % 模板目录
        figures              % 报告图形
        htmlReport           % HTML报告内容
        reportFiles          % 已生成报告文件信息
    end
    
    methods
        function obj = ReportGenerator(logger)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('ReportGenerator');
            else
                obj.logger = logger;
            end
            
            % 初始化属性
            obj.reportTitle = 'Binomial 分析报告';
            obj.reportSubtitle = '详细结果与解释';
            obj.author = 'Binomial 分析系统';
            obj.datestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            obj.reportDirectory = pwd;
            obj.templateDirectory = fullfile(fileparts(which('ReportGenerator')), 'templates');
            obj.figures = {};
            obj.htmlReport = '';
            obj.reportFiles = struct('type', {}, 'path', {}, 'size', {}, 'timestamp', {});
            
            obj.logger.info('报告生成模块已初始化');
        end
        
        function setResultData(obj, resultData)
            % 设置分析结果数据
            %
            % 参数:
            %   resultData - 包含分析结果的结构体
            
            obj.resultData = resultData;
            obj.logger.debug('已设置分析结果数据');
        end
        
        function setReportDetails(obj, title, subtitle, author)
            % 设置报告详细信息
            %
            % 参数:
            %   title - 报告标题
            %   subtitle - 报告副标题（可选）
            %   author - 报告作者（可选）
            
            if nargin >= 2 && ~isempty(title)
                obj.reportTitle = title;
            end
            
            if nargin >= 3 && ~isempty(subtitle)
                obj.reportSubtitle = subtitle;
            end
            
            if nargin >= 4 && ~isempty(author)
                obj.author = author;
            end
            
            obj.logger.debug('已设置报告详细信息');
        end
        
        function setReportDirectory(obj, directory)
            % 设置报告保存目录
            %
            % 参数:
            %   directory - 保存目录路径
            
            % 检查目录是否存在，不存在则创建
            if ~exist(directory, 'dir')
                [success, msg] = mkdir(directory);
                if ~success
                    obj.logger.error('创建目录失败: %s，错误: %s', directory, msg);
                    return;
                end
            end
            
            obj.reportDirectory = directory;
            obj.logger.info('报告保存目录已设置为: %s', directory);
        end
        
        function addFigure(obj, fig, title, description)
            % 添加图形到报告
            %
            % 参数:
            %   fig - 图形句柄
            %   title - 图形标题
            %   description - 图形描述（可选）
            
            if nargin < 4
                description = '';
            end
            
            newFig = struct();
            newFig.handle = fig;
            newFig.title = title;
            newFig.description = description;
            
            % 添加图形
            figIndex = length(obj.figures) + 1;
            obj.figures{figIndex} = newFig;
            
            obj.logger.debug('已添加图形到报告: %s', title);
        end
        
        function generateHTMLReport(obj)
            % 生成HTML格式的分析报告
            
            if isempty(obj.resultData)
                obj.logger.error('没有分析结果数据，无法生成报告');
                return;
            end
            
            % 初始化HTML内容
            html = obj.getHTMLHeader();
            
            % 添加报告标题和元信息
            html = [html, obj.getHTMLTitleSection()];
            
            % 添加摘要部分
            html = [html, obj.getHTMLSummarySection()];
            
            % 添加模型信息部分
            html = [html, obj.getHTMLModelSection()];
            
            % 添加变量重要性部分
            html = [html, obj.getHTMLVariableImportanceSection()];
            
            % 添加诊断部分
            html = [html, obj.getHTMLDiagnosticsSection()];
            
            % 添加结论和建议部分
            html = [html, obj.getHTMLConclusionSection()];
            
            % 添加附录（如果有）
            html = [html, obj.getHTMLAppendixSection()];
            
            % 添加页脚
            html = [html, obj.getHTMLFooter()];
            
            % 保存HTML内容
            obj.htmlReport = html;
            
            obj.logger.info('HTML报告内容已生成');
        end
        
        function [reportPath, fileInfo] = saveHTMLReport(obj, filename)
            % 保存HTML报告到文件
            %
            % 参数:
            %   filename - 文件名（可选，默认为 'binomial_report.html'）
            %
            % 返回值:
            %   reportPath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(filename)
                filename = 'binomial_report.html';
            end
            
            % 确保文件名有.html扩展名
            if ~endsWith(filename, '.html')
                filename = [filename, '.html'];
            end
            
            % 如果HTML内容为空，先生成
            if isempty(obj.htmlReport)
                obj.generateHTMLReport();
            end
            
            % 构建完整文件路径
            reportPath = fullfile(obj.reportDirectory, filename);
            
            try
                % 写入文件
                fid = fopen(reportPath, 'w');
                fprintf(fid, '%s', obj.htmlReport);
                fclose(fid);
                
                % 获取文件信息
                fileInfo = dir(reportPath);
                
                % 记录已生成文件的信息
                index = length(obj.reportFiles) + 1;
                obj.reportFiles(index).type = 'HTML';
                obj.reportFiles(index).path = reportPath;
                obj.reportFiles(index).size = fileInfo.bytes;
                obj.reportFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存HTML报告: %s (%.2f KB)', reportPath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('保存HTML报告失败: %s', ME.message);
                reportPath = '';
                fileInfo = [];
            end
        end
        
        function [reportPath, fileInfo] = generateMarkdownReport(obj, filename)
            % 生成Markdown格式的分析报告并保存
            %
            % 参数:
            %   filename - 文件名（可选，默认为 'binomial_report.md'）
            %
            % 返回值:
            %   reportPath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if isempty(obj.resultData)
                obj.logger.error('没有分析结果数据，无法生成报告');
                reportPath = '';
                fileInfo = [];
                return;
            end
            
            if nargin < 2 || isempty(filename)
                filename = 'binomial_report.md';
            end
            
            % 确保文件名有.md扩展名
            if ~endsWith(filename, '.md')
                filename = [filename, '.md'];
            end
            
            % 构建完整文件路径
            reportPath = fullfile(obj.reportDirectory, filename);
            
            try
                % 打开文件
                fid = fopen(reportPath, 'w');
                
                % 写入标题和元信息
                fprintf(fid, '# %s\n\n', obj.reportTitle);
                fprintf(fid, '## %s\n\n', obj.reportSubtitle);
                fprintf(fid, '**作者:** %s  \n', obj.author);
                fprintf(fid, '**日期:** %s  \n\n', obj.datestamp);
                fprintf(fid, '---\n\n');
                
                % 写入摘要部分
                fprintf(fid, '## 摘要\n\n');
                
                if isfield(obj.resultData, 'summary')
                    fprintf(fid, '%s\n\n', obj.resultData.summary);
                else
                    fprintf(fid, '本报告包含了使用Binomial分析工具进行的统计分析结果。\n\n');
                end
                
                % 写入模型信息部分
                fprintf(fid, '## 模型信息\n\n');
                
                if isfield(obj.resultData, 'modelInfo')
                    modelInfo = obj.resultData.modelInfo;
                    
                    fprintf(fid, '### 模型概述\n\n');
                    
                    if isfield(modelInfo, 'modelType')
                        fprintf(fid, '- **模型类型:** %s\n', modelInfo.modelType);
                    end
                    
                    if isfield(modelInfo, 'observations')
                        fprintf(fid, '- **观测数量:** %d\n', modelInfo.observations);
                    end
                    
                    if isfield(modelInfo, 'variables')
                        fprintf(fid, '- **变量数量:** %d\n', modelInfo.variables);
                    end
                    
                    fprintf(fid, '\n');
                    
                    % 写入模型质量指标
                    if isfield(modelInfo, 'quality')
                        quality = modelInfo.quality;
                        
                        fprintf(fid, '### 模型质量指标\n\n');
                        fprintf(fid, '| 指标 | 值 |\n');
                        fprintf(fid, '|-----|-----|\n');
                        
                        if isfield(quality, 'R2')
                            fprintf(fid, '| R² | %.4f |\n', quality.R2);
                        end
                        
                        if isfield(quality, 'AdjustedR2')
                            fprintf(fid, '| 调整R² | %.4f |\n', quality.AdjustedR2);
                        end
                        
                        if isfield(quality, 'RMSE')
                            fprintf(fid, '| RMSE | %.4f |\n', quality.RMSE);
                        end
                        
                        if isfield(quality, 'AIC')
                            fprintf(fid, '| AIC | %.2f |\n', quality.AIC);
                        end
                        
                        if isfield(quality, 'BIC')
                            fprintf(fid, '| BIC | %.2f |\n', quality.BIC);
                        end
                        
                        fprintf(fid, '\n');
                    end
                else
                    fprintf(fid, '未提供模型信息。\n\n');
                end
                
                % 写入系数部分
                fprintf(fid, '## 系数估计\n\n');
                
                if isfield(obj.resultData, 'coefficients') && isfield(obj.resultData, 'standardErrors')
                    coefficients = obj.resultData.coefficients;
                    standardErrors = obj.resultData.standardErrors;
                    
                    fprintf(fid, '| 变量 | 系数 | 标准误 | t值 | p值 |\n');
                    fprintf(fid, '|------|------|--------|-----|------|\n');
                    
                    for i = 1:length(coefficients)
                        varName = '';
                        if isfield(obj.resultData, 'variableNames') && length(obj.resultData.variableNames) >= i
                            varName = obj.resultData.variableNames{i};
                        else
                            varName = sprintf('变量%d', i);
                        end
                        
                        tStat = coefficients(i) / standardErrors(i);
                        pValue = 2 * (1 - tcdf(abs(tStat), obj.resultData.observations - length(coefficients)));
                        
                        fprintf(fid, '| %s | %.4f | %.4f | %.4f | %.4f |\n', ...
                            varName, coefficients(i), standardErrors(i), tStat, pValue);
                    end
                    
                    fprintf(fid, '\n');
                else
                    fprintf(fid, '未提供系数信息。\n\n');
                end
                
                % 写入变量重要性部分
                fprintf(fid, '## 变量重要性\n\n');
                
                if isfield(obj.resultData, 'importance')
                    importance = obj.resultData.importance;
                    
                    % 按重要性排序
                    [sortedImp, idx] = sort(importance, 'descend');
                    
                    fprintf(fid, '| 变量 | 重要性 |\n');
                    fprintf(fid, '|------|--------|\n');
                    
                    for i = 1:length(sortedImp)
                        varIdx = idx(i);
                        varName = '';
                        if isfield(obj.resultData, 'variableNames') && length(obj.resultData.variableNames) >= varIdx
                            varName = obj.resultData.variableNames{varIdx};
                        else
                            varName = sprintf('变量%d', varIdx);
                        end
                        
                        fprintf(fid, '| %s | %.4f |\n', varName, sortedImp(i));
                    end
                    
                    fprintf(fid, '\n');
                elseif isfield(obj.resultData, 'standardizedCoefficients')
                    stdCoefs = abs(obj.resultData.standardizedCoefficients);
                    
                    % 按绝对值排序
                    [sortedCoefs, idx] = sort(stdCoefs, 'descend');
                    
                    fprintf(fid, '| 变量 | 标准化系数 |\n');
                    fprintf(fid, '|------|------------|\n');
                    
                    for i = 1:length(sortedCoefs)
                        varIdx = idx(i);
                        varName = '';
                        if isfield(obj.resultData, 'variableNames') && length(obj.resultData.variableNames) >= varIdx
                            varName = obj.resultData.variableNames{varIdx};
                        else
                            varName = sprintf('变量%d', varIdx);
                        end
                        
                        fprintf(fid, '| %s | %.4f |\n', varName, sortedCoefs(i));
                    end
                    
                    fprintf(fid, '\n');
                else
                    fprintf(fid, '未提供变量重要性信息。\n\n');
                end
                
                % 写入诊断部分
                fprintf(fid, '## 模型诊断\n\n');
                
                if isfield(obj.resultData, 'diagnostics')
                    diagnostics = obj.resultData.diagnostics;
                    
                    if isfield(diagnostics, 'normalityTest')
                        fprintf(fid, '### 残差正态性检验\n\n');
                        
                        test = diagnostics.normalityTest;
                        fprintf(fid, '- **检验方法:** %s\n', test.method);
                        fprintf(fid, '- **统计量:** %.4f\n', test.statistic);
                        fprintf(fid, '- **p值:** %.4f\n', test.pValue);
                        fprintf(fid, '- **结论:** %s\n\n', test.conclusion);
                    end
                    
                    if isfield(diagnostics, 'autocorrelationTest')
                        fprintf(fid, '### 残差自相关检验\n\n');
                        
                        test = diagnostics.autocorrelationTest;
                        fprintf(fid, '- **检验方法:** %s\n', test.method);
                        fprintf(fid, '- **统计量:** %.4f\n', test.statistic);
                        fprintf(fid, '- **p值:** %.4f\n', test.pValue);
                        fprintf(fid, '- **结论:** %s\n\n', test.conclusion);
                    end
                    
                    if isfield(diagnostics, 'heteroskedasticityTest')
                        fprintf(fid, '### 残差异方差检验\n\n');
                        
                        test = diagnostics.heteroskedasticityTest;
                        fprintf(fid, '- **检验方法:** %s\n', test.method);
                        fprintf(fid, '- **统计量:** %.4f\n', test.statistic);
                        fprintf(fid, '- **p值:** %.4f\n', test.pValue);
                        fprintf(fid, '- **结论:** %s\n\n', test.conclusion);
                    end
                    
                    if isfield(diagnostics, 'outliers') && ~isempty(diagnostics.outliers)
                        fprintf(fid, '### 异常值检测\n\n');
                        
                        outliers = diagnostics.outliers;
                        fprintf(fid, '- **检测到的异常值数量:** %d\n', length(outliers));
                        fprintf(fid, '- **异常值索引:** ');
                        
                        for i = 1:min(10, length(outliers))
                            fprintf(fid, '%d', outliers(i));
                            if i < min(10, length(outliers))
                                fprintf(fid, ', ');
                            end
                        end
                        
                        if length(outliers) > 10
                            fprintf(fid, '等 (共%d个)\n\n', length(outliers));
                        else
                            fprintf(fid, '\n\n');
                        end
                    end
                else
                    fprintf(fid, '未提供诊断信息。\n\n');
                end
                
                % 写入结论和建议部分
                fprintf(fid, '## 结论和建议\n\n');
                
                if isfield(obj.resultData, 'conclusions')
                    conclusions = obj.resultData.conclusions;
                    
                    for i = 1:length(conclusions)
                        fprintf(fid, '- %s\n', conclusions{i});
                    end
                else
                    fprintf(fid, '基于分析结果，请自行解释和得出结论。\n');
                end
                
                fprintf(fid, '\n');
                
                % 写入页脚
                fprintf(fid, '---\n\n');
                fprintf(fid, '*本报告由Binomial分析系统自动生成于 %s*\n', obj.datestamp);
                
                % 关闭文件
                fclose(fid);
                
                % 获取文件信息
                fileInfo = dir(reportPath);
                
                % 记录已生成文件的信息
                index = length(obj.reportFiles) + 1;
                obj.reportFiles(index).type = 'Markdown';
                obj.reportFiles(index).path = reportPath;
                obj.reportFiles(index).size = fileInfo.bytes;
                obj.reportFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存Markdown报告: %s (%.2f KB)', reportPath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('生成Markdown报告失败: %s', ME.message);
                reportPath = '';
                fileInfo = [];
            end
        end
        
        function [reportPath, fileInfo] = generatePDFReport(obj, filename)
            % 生成PDF格式的分析报告
            %
            % 参数:
            %   filename - 文件名（可选，默认为 'binomial_report.pdf'）
            %
            % 返回值:
            %   reportPath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(filename)
                filename = 'binomial_report.pdf';
            end
            
            % 确保文件名有.pdf扩展名
            if ~endsWith(filename, '.pdf')
                filename = [filename, '.pdf'];
            end
            
            % 构建完整文件路径
            reportPath = fullfile(obj.reportDirectory, filename);
            
            try
                % 首先创建HTML报告
                if isempty(obj.htmlReport)
                    obj.generateHTMLReport();
                end
                
                % 尝试使用MATLAB的打印功能生成PDF
                tempHtmlFile = fullfile(obj.reportDirectory, 'temp_report.html');
                fid = fopen(tempHtmlFile, 'w');
                fprintf(fid, '%s', obj.htmlReport);
                fclose(fid);
                
                % 使用系统命令或第三方工具转换为PDF
                % 这里使用简单的方法：如果有可用的web浏览器打印功能，可以使用它
                % 注意：此方法可能需要进一步改进或使用其他工具
                
                % 尝试使用MATLAB的webwrite功能
                try
                    % 创建临时文件并打开
                    web(tempHtmlFile);
                    message = sprintf(['HTML报告已在浏览器中打开，请手动打印为PDF\n', ...
                        '保存PDF到: %s'], reportPath);
                    uiwait(msgbox(message, '生成PDF报告', 'modal'));
                    
                    % 用户手动保存后，检查文件是否存在
                    if exist(reportPath, 'file')
                        fileInfo = dir(reportPath);
                        
                        % 记录已生成文件的信息
                        index = length(obj.reportFiles) + 1;
                        obj.reportFiles(index).type = 'PDF';
                        obj.reportFiles(index).path = reportPath;
                        obj.reportFiles(index).size = fileInfo.bytes;
                        obj.reportFiles(index).timestamp = datestr(now);
                        
                        obj.logger.info('已保存PDF报告: %s (%.2f KB)', reportPath, fileInfo.bytes/1024);
                    else
                        obj.logger.warn('未找到PDF报告，可能用户取消了保存操作');
                        reportPath = '';
                        fileInfo = [];
                    end
                catch
                    obj.logger.warn('无法在浏览器中打开HTML报告，请手动将HTML转换为PDF');
                    reportPath = '';
                    fileInfo = [];
                end
                
                % 删除临时HTML文件
                if exist(tempHtmlFile, 'file')
                    delete(tempHtmlFile);
                end
            catch ME
                obj.logger.error('生成PDF报告失败: %s', ME.message);
                reportPath = '';
                fileInfo = [];
            end
        end
    end
    
    methods (Access = private)
        function html = getHTMLHeader(obj)
            % 生成HTML报告的头部
            
            html = ['<!DOCTYPE html>\n', ...
                '<html lang="zh-CN">\n', ...
                '<head>\n', ...
                '  <meta charset="UTF-8">\n', ...
                '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n', ...
                sprintf('  <title>%s</title>\n', obj.reportTitle), ...
                '  <style>\n', ...
                '    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; }\n', ...
                '    .container { max-width: 1200px; margin: 0 auto; }\n', ...
                '    header { text-align: center; margin-bottom: 30px; }\n', ...
                '    h1 { color: #2c3e50; }\n', ...
                '    h2 { color: #3498db; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 10px; }\n', ...
                '    h3 { color: #2980b9; }\n', ...
                '    table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n', ...
                '    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n', ...
                '    th { background-color: #f2f2f2; }\n', ...
                '    tr:nth-child(even) { background-color: #f9f9f9; }\n', ...
                '    .meta-info { color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }\n', ...
                '    .summary { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 20px; }\n', ...
                '    .conclusion { background-color: #eafaf1; padding: 15px; border-left: 4px solid #2ecc71; margin: 20px 0; }\n', ...
                '    .warning { background-color: #fcf3cf; padding: 15px; border-left: 4px solid #f1c40f; margin: 20px 0; }\n', ...
                '    .error { background-color: #fadbd8; padding: 15px; border-left: 4px solid #e74c3c; margin: 20px 0; }\n', ...
                '    .figure { text-align: center; margin: 20px 0; }\n', ...
                '    .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; }\n', ...
                '    .figure-caption { font-style: italic; color: #7f8c8d; margin-top: 5px; }\n', ...
                '    footer { margin-top: 50px; text-align: center; font-size: 0.8em; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 20px; }\n', ...
                '  </style>\n', ...
                '</head>\n', ...
                '<body>\n', ...
                '<div class="container">\n'];
        end
        
        function html = getHTMLTitleSection(obj)
            % 生成HTML报告的标题部分
            
            html = ['<header>\n', ...
                sprintf('  <h1>%s</h1>\n', obj.reportTitle), ...
                sprintf('  <h2>%s</h2>\n', obj.reportSubtitle), ...
                '  <div class="meta-info">\n', ...
                sprintf('    <p><strong>作者:</strong> %s</p>\n', obj.author), ...
                sprintf('    <p><strong>日期:</strong> %s</p>\n', obj.datestamp), ...
                '  </div>\n', ...
                '</header>\n'];
        end
        
        function html = getHTMLSummarySection(obj)
            % 生成HTML报告的摘要部分
            
            html = ['<section>\n', ...
                '  <h2>摘要</h2>\n', ...
                '  <div class="summary">\n'];
            
            if isfield(obj.resultData, 'summary')
                html = [html, sprintf('    <p>%s</p>\n', obj.resultData.summary)];
            else
                html = [html, '    <p>本报告包含了使用Binomial分析工具进行的统计分析结果。</p>\n'];
            end
            
            html = [html, '  </div>\n</section>\n'];
            
            return;
        end
        
        function html = getHTMLModelSection(obj)
            % 生成HTML报告的模型信息部分
            
            html = ['<section>\n', ...
                '  <h2>模型信息</h2>\n'];
            
            if isfield(obj.resultData, 'modelInfo')
                modelInfo = obj.resultData.modelInfo;
                
                html = [html, '  <h3>模型概述</h3>\n', ...
                    '  <table>\n', ...
                    '    <tr><th>属性</th><th>值</th></tr>\n'];
                
                if isfield(modelInfo, 'modelType')
                    html = [html, sprintf('    <tr><td>模型类型</td><td>%s</td></tr>\n', modelInfo.modelType)];
                end
                
                if isfield(modelInfo, 'observations')
                    html = [html, sprintf('    <tr><td>观测数量</td><td>%d</td></tr>\n', modelInfo.observations)];
                end
                
                if isfield(modelInfo, 'variables')
                    html = [html, sprintf('    <tr><td>变量数量</td><td>%d</td></tr>\n', modelInfo.variables)];
                end
                
                html = [html, '  </table>\n'];
                
                % 添加模型质量指标
                if isfield(modelInfo, 'quality')
                    quality = modelInfo.quality;
                    
                    html = [html, '  <h3>模型质量指标</h3>\n', ...
                        '  <table>\n', ...
                        '    <tr><th>指标</th><th>值</th></tr>\n'];
                    
                    if isfield(quality, 'R2')
                        html = [html, sprintf('    <tr><td>R²</td><td>%.4f</td></tr>\n', quality.R2)];
                    end
                    
                    if isfield(quality, 'AdjustedR2')
                        html = [html, sprintf('    <tr><td>调整R²</td><td>%.4f</td></tr>\n', quality.AdjustedR2)];
                    end
                    
                    if isfield(quality, 'RMSE')
                        html = [html, sprintf('    <tr><td>RMSE</td><td>%.4f</td></tr>\n', quality.RMSE)];
                    end
                    
                    if isfield(quality, 'AIC')
                        html = [html, sprintf('    <tr><td>AIC</td><td>%.2f</td></tr>\n', quality.AIC)];
                    end
                    
                    if isfield(quality, 'BIC')
                        html = [html, sprintf('    <tr><td>BIC</td><td>%.2f</td></tr>\n', quality.BIC)];
                    end
                    
                    html = [html, '  </table>\n'];
                end
                
                % 添加系数信息
                if isfield(obj.resultData, 'coefficients') && isfield(obj.resultData, 'standardErrors')
                    coefficients = obj.resultData.coefficients;
                    standardErrors = obj.resultData.standardErrors;
                    
                    html = [html, '  <h3>系数估计</h3>\n', ...
                        '  <table>\n', ...
                        '    <tr><th>变量</th><th>系数</th><th>标准误</th><th>t值</th><th>p值</th><th>显著性</th></tr>\n'];
                    
                    for i = 1:length(coefficients)
                        varName = '';
                        if isfield(obj.resultData, 'variableNames') && length(obj.resultData.variableNames) >= i
                            varName = obj.resultData.variableNames{i};
                        else
                            varName = sprintf('变量%d', i);
                        end
                        
                        tStat = coefficients(i) / standardErrors(i);
                        pValue = 2 * (1 - tcdf(abs(tStat), obj.resultData.observations - length(coefficients)));
                        
                        % 显著性标记
                        if pValue < 0.001
                            significance = '***';
                        elseif pValue < 0.01
                            significance = '**';
                        elseif pValue < 0.05
                            significance = '*';
                        elseif pValue < 0.1
                            significance = '.';
                        else
                            significance = '';
                        end
                        
                        html = [html, sprintf('    <tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%s</td></tr>\n', ...
                            varName, coefficients(i), standardErrors(i), tStat, pValue, significance)];
                    end
                    
                    html = [html, '  </table>\n', ...
                        '  <p><small>显著性代码: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, . p&lt;0.1</small></p>\n'];
                end
            else
                html = [html, '  <p>未提供模型信息。</p>\n'];
            end
            
            html = [html, '</section>\n'];
        end
        
        function html = getHTMLVariableImportanceSection(obj)
            % 生成HTML报告的变量重要性部分
            
            html = ['<section>\n', ...
                '  <h2>变量重要性</h2>\n'];
            
            if isfield(obj.resultData, 'importance')
                importance = obj.resultData.importance;
                
                % 按重要性排序
                [sortedImp, idx] = sort(importance, 'descend');
                
                html = [html, '  <table>\n', ...
                    '    <tr><th>变量</th><th>重要性</th><th>相对重要性</th></tr>\n'];
                
                for i = 1:length(sortedImp)
                    varIdx = idx(i);
                    varName = '';
                    if isfield(obj.resultData, 'variableNames') && length(obj.resultData.variableNames) >= varIdx
                        varName = obj.resultData.variableNames{varIdx};
                    else
                        varName = sprintf('变量%d', varIdx);
                    end
                    
                    % 计算相对重要性（百分比）
                    relativeImp = 100 * sortedImp(i) / sortedImp(1);
                    
                    % 创建条形图单元格
                    barWidth = round(relativeImp);
                    barCell = sprintf('<div style="background-color: #3498db; width: %d%%; height: 20px;"></div>', barWidth);
                    
                    html = [html, sprintf('    <tr><td>%s</td><td>%.4f</td><td>%s</td></tr>\n', ...
                        varName, sortedImp(i), barCell)];
                end
                
                html = [html, '  </table>\n'];
            elseif isfield(obj.resultData, 'standardizedCoefficients')
                stdCoefs = abs(obj.resultData.standardizedCoefficients);
                
                % 按绝对值排序
                [sortedCoefs, idx] = sort(stdCoefs, 'descend');
                
                html = [html, '  <table>\n', ...
                    '    <tr><th>变量</th><th>标准化系数(绝对值)</th><th>相对重要性</th></tr>\n'];
                
                for i = 1:length(sortedCoefs)
                    varIdx = idx(i);
                    varName = '';
                    if isfield(obj.resultData, 'variableNames') && length(obj.resultData.variableNames) >= varIdx
                        varName = obj.resultData.variableNames{varIdx};
                    else
                        varName = sprintf('变量%d', varIdx);
                    end
                    
                    % 计算相对重要性（百分比）
                    relativeImp = 100 * sortedCoefs(i) / sortedCoefs(1);
                    
                    % 创建条形图单元格
                    barWidth = round(relativeImp);
                    barCell = sprintf('<div style="background-color: #3498db; width: %d%%; height: 20px;"></div>', barWidth);
                    
                    html = [html, sprintf('    <tr><td>%s</td><td>%.4f</td><td>%s</td></tr>\n', ...
                        varName, sortedCoefs(i), barCell)];
                end
                
                html = [html, '  </table>\n'];
            else
                html = [html, '  <p>未提供变量重要性信息。</p>\n'];
            end
            
            html = [html, '</section>\n'];
        end
        
        function html = getHTMLDiagnosticsSection(obj)
            % 生成HTML报告的诊断部分
            
            html = ['<section>\n', ...
                '  <h2>模型诊断</h2>\n'];
            
            if isfield(obj.resultData, 'diagnostics')
                diagnostics = obj.resultData.diagnostics;
                
                if isfield(diagnostics, 'normalityTest')
                    test = diagnostics.normalityTest;
                    
                    html = [html, '  <h3>残差正态性检验</h3>\n', ...
                        '  <table>\n', ...
                        '    <tr><th>检验方法</th><th>统计量</th><th>p值</th><th>结论</th></tr>\n', ...
                        sprintf('    <tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%s</td></tr>\n', ...
                        test.method, test.statistic, test.pValue, test.conclusion), ...
                        '  </table>\n'];
                end
                
                if isfield(diagnostics, 'autocorrelationTest')
                    test = diagnostics.autocorrelationTest;
                    
                    html = [html, '  <h3>残差自相关检验</h3>\n', ...
                        '  <table>\n', ...
                        '    <tr><th>检验方法</th><th>统计量</th><th>p值</th><th>结论</th></tr>\n', ...
                        sprintf('    <tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%s</td></tr>\n', ...
                        test.method, test.statistic, test.pValue, test.conclusion), ...
                        '  </table>\n'];
                end
                
                if isfield(diagnostics, 'heteroskedasticityTest')
                    test = diagnostics.heteroskedasticityTest;
                    
                    html = [html, '  <h3>残差异方差检验</h3>\n', ...
                        '  <table>\n', ...
                        '    <tr><th>检验方法</th><th>统计量</th><th>p值</th><th>结论</th></tr>\n', ...
                        sprintf('    <tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%s</td></tr>\n', ...
                        test.method, test.statistic, test.pValue, test.conclusion), ...
                        '  </table>\n'];
                end
                
                if isfield(diagnostics, 'outliers') && ~isempty(diagnostics.outliers)
                    outliers = diagnostics.outliers;
                    
                    html = [html, '  <h3>异常值检测</h3>\n', ...
                        sprintf('  <p>检测到 %d 个异常值。</p>\n', length(outliers))];
                    
                    if length(outliers) <= 20
                        html = [html, '  <p>异常值索引: '];
                        
                        for i = 1:length(outliers)
                            html = [html, sprintf('%d', outliers(i))];
                            if i < length(outliers)
                                html = [html, ', '];
                            end
                        end
                        
                        html = [html, '</p>\n'];
                    else
                        html = [html, sprintf('  <p>异常值数量过多 (%d个)，此处不列出所有索引。</p>\n', length(outliers))];
                    end
                    
                    if isfield(diagnostics, 'outlierDetails') && istable(diagnostics.outlierDetails)
                        details = diagnostics.outlierDetails;
                        html = [html, '  <h4>主要异常值详情</h4>\n', ...
                            '  <table>\n', ...
                            '    <tr><th>索引</th>'];
                        
                        % 添加表头
                        for col = 1:width(details)
                            html = [html, sprintf('<th>%s</th>', details.Properties.VariableNames{col})];
                        end
                        html = [html, '</tr>\n'];
                        
                        % 添加表格内容（最多显示10行）
                        maxRows = min(10, height(details));
                        for row = 1:maxRows
                            html = [html, '    <tr><td>', num2str(row), '</td>'];
                            
                            for col = 1:width(details)
                                val = details{row, col};
                                if isnumeric(val)
                                    html = [html, sprintf('<td>%.4f</td>', val)];
                                elseif ischar(val)
                                    html = [html, sprintf('<td>%s</td>', val)];
                                else
                                    html = [html, '<td>?</td>'];
                                end
                            end
                            
                            html = [html, '</tr>\n'];
                        end
                        
                        html = [html, '  </table>\n'];
                        
                        if height(details) > maxRows
                            html = [html, sprintf('  <p><small>表格仅显示前 %d 行异常值。</small></p>\n', maxRows)];
                        end
                    end
                end
            else
                html = [html, '  <p>未提供诊断信息。</p>\n'];
            end
            
            html = [html, '</section>\n'];
        end
        
        function html = getHTMLConclusionSection(obj)
            % 生成HTML报告的结论部分
            
            html = ['<section>\n', ...
                '  <h2>结论和建议</h2>\n', ...
                '  <div class="conclusion">\n'];
            
            if isfield(obj.resultData, 'conclusions') && ~isempty(obj.resultData.conclusions)
                conclusions = obj.resultData.conclusions;
                
                html = [html, '    <ul>\n'];
                for i = 1:length(conclusions)
                    html = [html, sprintf('      <li>%s</li>\n', conclusions{i})];
                end
                html = [html, '    </ul>\n'];
            else
                html = [html, '    <p>基于分析结果，请自行解释和得出结论。</p>\n'];
            end
            
            html = [html, '  </div>\n'];
            
            % 添加警告信息（如果有）
            if isfield(obj.resultData, 'warnings') && ~isempty(obj.resultData.warnings)
                warnings = obj.resultData.warnings;
                
                html = [html, '  <div class="warning">\n', ...
                    '    <h3>注意事项</h3>\n', ...
                    '    <ul>\n'];
                
                for i = 1:length(warnings)
                    html = [html, sprintf('      <li>%s</li>\n', warnings{i})];
                end
                
                html = [html, '    </ul>\n', ...
                    '  </div>\n'];
            end
            
            html = [html, '</section>\n'];
        end
        
        function html = getHTMLAppendixSection(obj)
            % 生成HTML报告的附录部分
            
            % 如果没有附录内容，返回空字符串
            if ~isfield(obj.resultData, 'appendix') || isempty(obj.resultData.appendix)
                html = '';
                return;
            end
            
            appendix = obj.resultData.appendix;
            
            html = ['<section>\n', ...
                '  <h2>附录</h2>\n'];
            
            % 根据附录内容类型添加不同的部分
            if isstruct(appendix)
                fields = fieldnames(appendix);
                
                for i = 1:length(fields)
                    field = fields{i};
                    content = appendix.(field);
                    
                    html = [html, sprintf('  <h3>%s</h3>\n', field)];
                    
                    if istable(content)
                        % 如果是表格，创建HTML表格
                        html = [html, '  <table>\n', ...
                            '    <tr>'];
                        
                        % 添加表头
                        for col = 1:width(content)
                            html = [html, sprintf('<th>%s</th>', content.Properties.VariableNames{col})];
                        end
                        html = [html, '</tr>\n'];
                        
                        % 添加表格内容（最多显示50行）
                        maxRows = min(50, height(content));
                        for row = 1:maxRows
                            html = [html, '    <tr>'];
                            
                            for col = 1:width(content)
                                val = content{row, col};
                                if isnumeric(val)
                                    html = [html, sprintf('<td>%.4g</td>', val)];
                                elseif ischar(val)
                                    html = [html, sprintf('<td>%s</td>', val)];
                                else
                                    html = [html, '<td>?</td>'];
                                end
                            end
                            
                            html = [html, '</tr>\n'];
                        end
                        
                        html = [html, '  </table>\n'];
                        
                        if height(content) > maxRows
                            html = [html, sprintf('  <p><small>表格仅显示前 %d 行。</small></p>\n', maxRows)];
                        end
                    elseif isnumeric(content) && ismatrix(content)
                        % 如果是数值矩阵，创建HTML表格
                        html = [html, sprintf('  <p>数值矩阵: [%d x %d]</p>\n', size(content, 1), size(content, 2))];
                        
                        if numel(content) <= 1000
                            html = [html, '  <table>\n'];
                            
                            % 添加表格内容
                            for row = 1:size(content, 1)
                                html = [html, '    <tr>'];
                                
                                for col = 1:size(content, 2)
                                    html = [html, sprintf('<td>%.4g</td>', content(row, col))];
                                end
                                
                                html = [html, '</tr>\n'];
                            end
                            
                            html = [html, '  </table>\n'];
                        else
                            html = [html, '  <p>矩阵过大，无法完全显示。</p>\n'];
                        end
                    elseif ischar(content)
                        % 如果是字符串，直接添加
                        html = [html, sprintf('  <p>%s</p>\n', content)];
                    elseif iscell(content)
                        % 如果是元胞数组，创建列表
                        html = [html, '  <ul>\n'];
                        
                        for j = 1:min(100, length(content))
                            item = content{j};
                            if ischar(item)
                                html = [html, sprintf('    <li>%s</li>\n', item)];
                            elseif isnumeric(item) && isscalar(item)
                                html = [html, sprintf('    <li>%g</li>\n', item)];
                            else
                                html = [html, '    <li>[复杂数据]</li>\n'];
                            end
                        end
                        
                        html = [html, '  </ul>\n'];
                        
                        if length(content) > 100
                            html = [html, sprintf('  <p><small>列表仅显示前 100 项，共 %d 项。</small></p>\n', length(content))];
                        end
                    else
                        % 其他类型，显示类型信息
                        html = [html, sprintf('  <p>[%s 类型数据，无法直接显示]</p>\n', class(content))];
                    end
                end
            elseif iscell(appendix)
                % 如果是元胞数组，假设每个元素都是可添加的部分
                for i = 1:length(appendix)
                    item = appendix{i};
                    
                    if isstruct(item) && isfield(item, 'title') && isfield(item, 'content')
                        html = [html, sprintf('  <h3>%s</h3>\n', item.title)];
                        
                        if ischar(item.content)
                            html = [html, sprintf('  <p>%s</p>\n', item.content)];
                        elseif istable(item.content)
                            % 处理表格（类似上面的代码）
                            html = [html, '  <table>\n', ...
                                '    <tr>'];
                            
                            for col = 1:width(item.content)
                                html = [html, sprintf('<th>%s</th>', item.content.Properties.VariableNames{col})];
                            end
                            html = [html, '</tr>\n'];
                            
                            maxRows = min(50, height(item.content));
                            for row = 1:maxRows
                                html = [html, '    <tr>'];
                                
                                for col = 1:width(item.content)
                                    val = item.content{row, col};
                                    if isnumeric(val)
                                        html = [html, sprintf('<td>%.4g</td>', val)];
                                    elseif ischar(val)
                                        html = [html, sprintf('<td>%s</td>', val)];
                                    else
                                        html = [html, '<td>?</td>'];
                                    end
                                end
                                
                                html = [html, '</tr>\n'];
                            end
                            
                            html = [html, '  </table>\n'];
                        else
                            html = [html, '  <p>[复杂数据]</p>\n'];
                        end
                    end
                end
            end
            
            html = [html, '</section>\n'];
        end
        
        function html = getHTMLFooter(obj)
            % 生成HTML报告的页脚
            
            html = ['<footer>\n', ...
                sprintf('  <p>本报告由Binomial分析系统自动生成于 %s</p>\n', obj.datestamp), ...
                '</footer>\n', ...
                '</div>\n', ...
                '</body>\n', ...
                '</html>\n'];
        end
    end
    end