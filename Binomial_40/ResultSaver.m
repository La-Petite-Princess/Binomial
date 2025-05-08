classdef ResultSaver < handle
    % ResultSaver - 模型结果保存模块
    %
    % 该类负责将模型分析的各种结果保存为文件，
    % 支持MAT、CSV、Excel、JSON等多种格式，方便后续分析和共享。
    %
    % 属性:
    %   logger - 日志记录器对象
    %   resultData - 待保存的结果数据
    %   saveDirectory - 保存目录
    %   sessionId - 会话标识符
    %   saveName - 保存文件的基本名称
    %   saveTime - 保存时间
    %   savedFiles - 已保存文件的信息
    
    properties
        logger              % 日志记录器
        resultData          % 结果数据结构体
        saveDirectory       % 保存目录
        sessionId           % 会话标识符
        saveName            % 保存文件基本名称
        saveTime            % 保存时间
        savedFiles          % 已保存文件信息
    end
    
    methods
        function obj = ResultSaver(logger)
            % 构造函数
            %
            % 参数:
            %   logger - BinomialLogger实例
            
            if nargin < 1 || isempty(logger)
                obj.logger = BinomialLogger.getLogger('ResultSaver');
            else
                obj.logger = logger;
            end
            
            % 初始化属性
            obj.saveDirectory = pwd;
            obj.sessionId = datestr(now, 'yyyymmdd_HHMMSS');
            obj.saveName = ['binomial_results_', obj.sessionId];
            obj.saveTime = now;
            obj.savedFiles = struct('name', {}, 'path', {}, 'format', {}, 'size', {}, 'timestamp', {});
            
            obj.logger.info('结果保存模块已初始化，会话ID: %s', obj.sessionId);
        end
        
        function setResultData(obj, resultData)
            % 设置要保存的结果数据
            %
            % 参数:
            %   resultData - 包含分析结果的结构体或对象
            
            obj.resultData = resultData;
            obj.logger.debug('已设置结果数据');
        end
        
        function setSaveDirectory(obj, directory)
            % 设置保存目录
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
            
            obj.saveDirectory = directory;
            obj.logger.info('保存目录已设置为: %s', directory);
        end
        
        function setSaveName(obj, baseName)
            % 设置保存文件的基本名称
            %
            % 参数:
            %   baseName - 文件基本名称（不含扩展名）
            
            obj.saveName = baseName;
            obj.logger.debug('保存文件基本名称已设置为: %s', baseName);
        end
        
        function [filePath, fileInfo] = saveToMAT(obj, data, filename)
            % 保存数据为MAT文件
            %
            % 参数:
            %   data - 要保存的数据（可选，默认为obj.resultData）
            %   filename - 文件名（可选，默认为obj.saveName）
            %
            % 返回值:
            %   filePath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(data)
                data = obj.resultData;
            end
            
            if nargin < 3 || isempty(filename)
                filename = obj.saveName;
            end
            
            % 确保文件名有.mat扩展名
            if ~endsWith(filename, '.mat')
                filename = [filename, '.mat'];
            end
            
            % 构建完整文件路径
            filePath = fullfile(obj.saveDirectory, filename);
            
            try
                % 保存数据
                save(filePath, 'data');
                
                % 获取文件信息
                fileInfo = dir(filePath);
                
                % 记录已保存文件的信息
                index = length(obj.savedFiles) + 1;
                obj.savedFiles(index).name = filename;
                obj.savedFiles(index).path = filePath;
                obj.savedFiles(index).format = 'MAT';
                obj.savedFiles(index).size = fileInfo.bytes;
                obj.savedFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存MAT文件: %s (%.2f KB)', filePath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('保存MAT文件失败: %s', ME.message);
                filePath = '';
                fileInfo = [];
            end
        end
        
        function [filePath, fileInfo] = saveToCSV(obj, data, filename, variableNames)
            % 保存数据为CSV文件
            %
            % 参数:
            %   data - 要保存的数据（可选，默认为obj.resultData）
            %   filename - 文件名（可选，默认为obj.saveName）
            %   variableNames - 变量名称元胞数组（可选）
            %
            % 返回值:
            %   filePath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(data)
                data = obj.resultData;
            end
            
            if nargin < 3 || isempty(filename)
                filename = obj.saveName;
            end
            
            % 确保文件名有.csv扩展名
            if ~endsWith(filename, '.csv')
                filename = [filename, '.csv'];
            end
            
            % 构建完整文件路径
            filePath = fullfile(obj.saveDirectory, filename);
            
            try
                % 将数据转换为表格形式
                if istable(data)
                    T = data;
                elseif isnumeric(data)
                    % 如果是数值矩阵，转换为表格
                    if nargin < 4 || isempty(variableNames)
                        % 生成默认变量名
                        variableNames = arrayfun(@(i) sprintf('Var%d', i), ...
                            1:size(data, 2), 'UniformOutput', false);
                    end
                    T = array2table(data, 'VariableNames', variableNames);
                elseif isstruct(data)
                    % 如果是结构体，提取字段并创建表格
                    % 只处理简单字段（非结构体、非元胞数组）
                    fields = fieldnames(data);
                    validFields = {};
                    validData = {};
                    
                    for i = 1:length(fields)
                        field = fields{i};
                        value = data.(field);
                        
                        if isnumeric(value) && ismatrix(value) && size(value, 1) == 1
                            % 标量或向量
                            validFields{end+1} = field;
                            validData{end+1} = value(:)';
                        elseif ischar(value)
                            % 字符串
                            validFields{end+1} = field;
                            validData{end+1} = {value};
                        elseif islogical(value) && isscalar(value)
                            % 逻辑标量
                            validFields{end+1} = field;
                            validData{end+1} = value;
                        end
                    end
                    
                    if isempty(validFields)
                        obj.logger.warn('找不到适合保存为CSV的字段');
                        filePath = '';
                        fileInfo = [];
                        return;
                    end
                    
                    % 创建表格
                    T = table(validData{:}, 'VariableNames', validFields);
                else
                    obj.logger.error('不支持的数据类型');
                    filePath = '';
                    fileInfo = [];
                    return;
                end
                
                % 保存表格为CSV
                writetable(T, filePath);
                
                % 获取文件信息
                fileInfo = dir(filePath);
                
                % 记录已保存文件的信息
                index = length(obj.savedFiles) + 1;
                obj.savedFiles(index).name = filename;
                obj.savedFiles(index).path = filePath;
                obj.savedFiles(index).format = 'CSV';
                obj.savedFiles(index).size = fileInfo.bytes;
                obj.savedFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存CSV文件: %s (%.2f KB)', filePath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('保存CSV文件失败: %s', ME.message);
                filePath = '';
                fileInfo = [];
            end
        end
        
        function [filePath, fileInfo] = saveToExcel(obj, data, filename, sheetName)
            % 保存数据为Excel文件
            %
            % 参数:
            %   data - 要保存的数据（可选，默认为obj.resultData）
            %   filename - 文件名（可选，默认为obj.saveName）
            %   sheetName - 工作表名称（可选，默认为'Results'）
            %
            % 返回值:
            %   filePath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(data)
                data = obj.resultData;
            end
            
            if nargin < 3 || isempty(filename)
                filename = obj.saveName;
            end
            
            if nargin < 4 || isempty(sheetName)
                sheetName = 'Results';
            end
            
            % 确保文件名有.xlsx扩展名
            if ~endsWith(filename, '.xlsx')
                filename = [filename, '.xlsx'];
            end
            
            % 构建完整文件路径
            filePath = fullfile(obj.saveDirectory, filename);
            
            try
                % 将数据转换为可写入Excel的形式
                if istable(data)
                    % 直接写入表格
                    writetable(data, filePath, 'Sheet', sheetName);
                elseif isnumeric(data)
                    % 如果是数值数据，直接写入
                    writematrix(data, filePath, 'Sheet', sheetName);
                elseif isstruct(data)
                    % 如果是结构体，创建多个工作表
                    fields = fieldnames(data);
                    
                    for i = 1:length(fields)
                        field = fields{i};
                        value = data.(field);
                        
                        currentSheet = ['Sheet_', field];
                        
                        if istable(value)
                            writetable(value, filePath, 'Sheet', currentSheet);
                        elseif isnumeric(value) && ismatrix(value)
                            writematrix(value, filePath, 'Sheet', currentSheet);
                        elseif ischar(value) || iscellstr(value)
                            writecell({value}, filePath, 'Sheet', currentSheet);
                        elseif iscell(value) && ~isempty(value)
                            writecell(value, filePath, 'Sheet', currentSheet);
                        else
                            obj.logger.warn('跳过不支持的字段: %s', field);
                        end
                    end
                else
                    obj.logger.error('不支持的数据类型');
                    filePath = '';
                    fileInfo = [];
                    return;
                end
                
                % 获取文件信息
                fileInfo = dir(filePath);
                
                % 记录已保存文件的信息
                index = length(obj.savedFiles) + 1;
                obj.savedFiles(index).name = filename;
                obj.savedFiles(index).path = filePath;
                obj.savedFiles(index).format = 'Excel';
                obj.savedFiles(index).size = fileInfo.bytes;
                obj.savedFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存Excel文件: %s (%.2f KB)', filePath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('保存Excel文件失败: %s', ME.message);
                filePath = '';
                fileInfo = [];
            end
        end
        
        function [filePath, fileInfo] = saveToJSON(obj, data, filename)
            % 保存数据为JSON文件
            %
            % 参数:
            %   data - 要保存的数据（可选，默认为obj.resultData）
            %   filename - 文件名（可选，默认为obj.saveName）
            %
            % 返回值:
            %   filePath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(data)
                data = obj.resultData;
            end
            
            if nargin < 3 || isempty(filename)
                filename = obj.saveName;
            end
            
            % 确保文件名有.json扩展名
            if ~endsWith(filename, '.json')
                filename = [filename, '.json'];
            end
            
            % 构建完整文件路径
            filePath = fullfile(obj.saveDirectory, filename);
            
            try
                % 将数据转换为JSON
                jsonStr = jsonencode(data);
                
                % 使用更可读的格式（为了可读性，增加缩进）
                jsonStr = obj.prettifyJSON(jsonStr);
                
                % 写入文件
                fid = fopen(filePath, 'w');
                fprintf(fid, '%s', jsonStr);
                fclose(fid);
                
                % 获取文件信息
                fileInfo = dir(filePath);
                
                % 记录已保存文件的信息
                index = length(obj.savedFiles) + 1;
                obj.savedFiles(index).name = filename;
                obj.savedFiles(index).path = filePath;
                obj.savedFiles(index).format = 'JSON';
                obj.savedFiles(index).size = fileInfo.bytes;
                obj.savedFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存JSON文件: %s (%.2f KB)', filePath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('保存JSON文件失败: %s', ME.message);
                filePath = '';
                fileInfo = [];
            end
        end
        
        function [filePath, fileInfo] = saveToTXT(obj, data, filename)
            % 保存数据为纯文本文件
            %
            % 参数:
            %   data - 要保存的数据（可选，默认为obj.resultData）
            %   filename - 文件名（可选，默认为obj.saveName）
            %
            % 返回值:
            %   filePath - 保存的文件路径
            %   fileInfo - 文件信息结构体
            
            if nargin < 2 || isempty(data)
                data = obj.resultData;
            end
            
            if nargin < 3 || isempty(filename)
                filename = obj.saveName;
            end
            
            % 确保文件名有.txt扩展名
            if ~endsWith(filename, '.txt')
                filename = [filename, '.txt'];
            end
            
            % 构建完整文件路径
            filePath = fullfile(obj.saveDirectory, filename);
            
            try
                % 打开文件
                fid = fopen(filePath, 'w');
                
                % 写入标题
                fprintf(fid, '===================================================\n');
                fprintf(fid, '                  分析结果报告\n');
                fprintf(fid, '===================================================\n');
                fprintf(fid, '生成时间: %s\n', datestr(now));
                fprintf(fid, '---------------------------------------------------\n\n');
                
                % 根据数据类型写入内容
                if isstruct(data)
                    fields = fieldnames(data);
                    for i = 1:length(fields)
                        field = fields{i};
                        value = data.(field);
                        
                        fprintf(fid, '## %s\n', field);
                        
                        if isnumeric(value) && isscalar(value)
                            fprintf(fid, '%g\n\n', value);
                        elseif isnumeric(value) && ismatrix(value) && numel(value) <= 100
                            fprintf(fid, '[\n');
                            for row = 1:size(value, 1)
                                fprintf(fid, '  ');
                                fprintf(fid, '%g ', value(row, :));
                                fprintf(fid, ']\n\n');
                            end
                        elseif ischar(value)
                            fprintf(fid, '%s\n\n', value);
                        elseif iscell(value) && ~isempty(value) && ischar(value{1})
                            for j = 1:length(value)
                                fprintf(fid, '- %s\n', value{j});
                            end
                            fprintf(fid, '\n');
                        elseif istable(value) && size(value, 1) <= 100
                            % 表格的列名
                            fprintf(fid, '| ');
                            for col = 1:width(value)
                                fprintf(fid, '%s | ', value.Properties.VariableNames{col});
                            end
                            fprintf(fid, '\n');
                            
                            % 分隔线
                            fprintf(fid, '| ');
                            for col = 1:width(value)
                                fprintf(fid, '--- | ');
                            end
                            fprintf(fid, '\n');
                            
                            % 表格内容
                            for row = 1:height(value)
                                fprintf(fid, '| ');
                                for col = 1:width(value)
                                    val = value{row, col};
                                    if isnumeric(val)
                                        fprintf(fid, '%g | ', val);
                                    elseif ischar(val)
                                        fprintf(fid, '%s | ', val);
                                    else
                                        fprintf(fid, '? | ');
                                    end
                                end
                                fprintf(fid, '\n');
                            end
                            fprintf(fid, '\n');
                        else
                            fprintf(fid, '[复杂数据结构，无法以文本形式显示]\n\n');
                        end
                    end
                elseif istable(data)
                    % 表格的列名
                    fprintf(fid, '| ');
                    for col = 1:width(data)
                        fprintf(fid, '%s | ', data.Properties.VariableNames{col});
                    end
                    fprintf(fid, '\n');
                    
                    % 分隔线
                    fprintf(fid, '| ');
                    for col = 1:width(data)
                        fprintf(fid, '--- | ');
                    end
                    fprintf(fid, '\n');
                    
                    % 表格内容（限制行数以避免过大）
                    maxRows = min(100, height(data));
                    for row = 1:maxRows
                        fprintf(fid, '| ');
                        for col = 1:width(data)
                            val = data{row, col};
                            if isnumeric(val)
                                fprintf(fid, '%g | ', val);
                            elseif ischar(val)
                                fprintf(fid, '%s | ', val);
                            else
                                fprintf(fid, '? | ');
                            end
                        end
                        fprintf(fid, '\n');
                    end
                    
                    if height(data) > maxRows
                        fprintf(fid, '\n[表格过大，只显示前 %d 行]\n', maxRows);
                    end
                elseif isnumeric(data)
                    % 数值矩阵
                    fprintf(fid, '数值矩阵: [%d x %d]\n\n', size(data, 1), size(data, 2));
                    
                    if numel(data) <= 1000
                        for row = 1:size(data, 1)
                            for col = 1:size(data, 2)
                                fprintf(fid, '%g\t', data(row, col));
                            end
                            fprintf(fid, '\n');
                        end
                    else
                        fprintf(fid, '[矩阵过大，无法完全显示]\n');
                    end
                else
                    fprintf(fid, '[不支持的数据类型]\n');
                end
                
                % 关闭文件
                fclose(fid);
                
                % 获取文件信息
                fileInfo = dir(filePath);
                
                % 记录已保存文件的信息
                index = length(obj.savedFiles) + 1;
                obj.savedFiles(index).name = filename;
                obj.savedFiles(index).path = filePath;
                obj.savedFiles(index).format = 'TXT';
                obj.savedFiles(index).size = fileInfo.bytes;
                obj.savedFiles(index).timestamp = datestr(now);
                
                obj.logger.info('已保存TXT文件: %s (%.2f KB)', filePath, fileInfo.bytes/1024);
            catch ME
                obj.logger.error('保存TXT文件失败: %s', ME.message);
                filePath = '';
                fileInfo = [];
            end
        end
        
        function saveResults(obj, formats, data)
            % 一次性保存多种格式的结果
            %
            % 参数:
            %   formats - 格式字符串元胞数组，如 {'mat', 'csv', 'xlsx'}
            %   data - 要保存的数据（可选，默认为obj.resultData）
            
            if nargin < 3 || isempty(data)
                data = obj.resultData;
            end
            
            if isempty(data)
                obj.logger.error('没有数据可以保存');
                return;
            end
            
            if nargin < 2 || isempty(formats)
                formats = {'mat'};  % 默认保存为MAT文件
            end
            
            % 如果formats是字符串，转换为元胞数组
            if ischar(formats)
                formats = {formats};
            end
            
            % 遍历每种格式进行保存
            savedCount = 0;
            for i = 1:length(formats)
                format = lower(formats{i});
                switch format
                    case {'mat', 'matlab'}
                        [filePath, ~] = obj.saveToMAT(data);
                        if ~isempty(filePath)
                            savedCount = savedCount + 1;
                        end
                    case {'csv'}
                        [filePath, ~] = obj.saveToCSV(data);
                        if ~isempty(filePath)
                            savedCount = savedCount + 1;
                        end
                    case {'excel', 'xlsx', 'xls'}
                        [filePath, ~] = obj.saveToExcel(data);
                        if ~isempty(filePath)
                            savedCount = savedCount + 1;
                        end
                    case {'json'}
                        [filePath, ~] = obj.saveToJSON(data);
                        if ~isempty(filePath)
                            savedCount = savedCount + 1;
                        end
                    case {'txt', 'text'}
                        [filePath, ~] = obj.saveToTXT(data);
                        if ~isempty(filePath)
                            savedCount = savedCount + 1;
                        end
                    otherwise
                        obj.logger.warn('不支持的格式: %s', format);
                end
            end
            
            obj.logger.info('已保存 %d 个文件', savedCount);
        end
        
        function filesInfo = getSavedFilesInfo(obj)
            % 获取已保存文件的信息
            %
            % 返回值:
            %   filesInfo - 包含文件信息的结构体数组
            
            filesInfo = obj.savedFiles;
        end
    end
    
    methods (Static)
        function jsonStr = prettifyJSON(jsonStr)
            % 美化JSON字符串，添加缩进和换行
            %
            % 参数:
            %   jsonStr - 原始JSON字符串
            %
            % 返回值:
            %   jsonStr - 美化后的JSON字符串
            
            % 注意：这是一个简单的美化处理，不处理复杂的嵌套结构
            
            % 替换花括号
            jsonStr = strrep(jsonStr, '{', '{\n');
            jsonStr = strrep(jsonStr, '}', '\n}');
            
            % 替换方括号
            jsonStr = strrep(jsonStr, '[', '[\n');
            jsonStr = strrep(jsonStr, ']', '\n]');
            
            % 替换逗号
            jsonStr = strrep(jsonStr, ',', ',\n');
            
            % 添加缩进
            lines = strsplit(jsonStr, '\n');
            indentLevel = 0;
            indentStr = '  ';  % 两个空格的缩进
            
            for i = 1:length(lines)
                line = lines{i};
                
                % 调整缩进级别
                if ~isempty(line) && (line(1) == '}' || line(1) == ']')
                    indentLevel = max(0, indentLevel - 1);
                end
                
                % 添加缩进
                if ~isempty(line)
                    lines{i} = [repmat(indentStr, 1, indentLevel), line];
                end
                
                % 调整下一行的缩进级别
                if ~isempty(line) && (line(1) == '{' || line(1) == '[')
                    indentLevel = indentLevel + 1;
                end
            end
            
            % 重新组合成字符串
            jsonStr = strjoin(lines, '\n');
        end
        
        function data = loadFromMAT(filePath)
            % 从MAT文件加载数据
            %
            % 参数:
            %   filePath - MAT文件路径
            %
            % 返回值:
            %   data - 加载的数据
            
            try
                loaded = load(filePath);
                if isfield(loaded, 'data')
                    data = loaded.data;
                else
                    % 如果没有data字段，返回整个结构体
                    data = loaded;
                end
            catch ME
                error('加载MAT文件失败: %s', ME.message);
            end
        end
        
        function saveTableToCsv(table, filePath)
            % 将表格保存为CSV文件
            %
            % 参数:
            %   table - 要保存的表格
            %   filePath - 保存文件路径
            
            try
                writetable(table, filePath);
            catch ME
                error('保存CSV文件失败: %s', ME.message);
            end
        end
    end
end