classdef BinomialLogger < handle
    % BinomialLogger - 日志记录系统
    %
    % 该类提供了统一的日志记录功能，支持不同级别的日志输出，
    % 可以将日志内容输出到控制台和文件，方便调试和追踪。
    %
    % 可用的日志级别包括:
    %   DEBUG   - 详细的调试信息
    %   INFO    - 一般的信息性消息
    %   WARN    - 警告信息
    %   ERROR   - 错误信息
    %
    % 示例:
    %   logger = BinomialLogger.getLogger('ModuleName');
    %   logger.info('这是一条信息');
    %   logger.error('发生错误: %s', '错误详情');
    
    properties (Constant)
        % 日志级别常量
        LEVEL_DEBUG = 1;
        LEVEL_INFO = 2;
        LEVEL_WARN = 3;
        LEVEL_ERROR = 4;
    end
    
    properties
        name            % 日志记录器名称
        level           % 日志级别
        fileHandlers    % 文件处理器列表
        useConsole      % 是否输出到控制台
        useTimestamp    % 是否显示时间戳
        useLoggerName   % 是否显示记录器名称
        dateFormat      % 日期格式
    end
    
    properties (Access = private, Constant)
        % 单例实例映射表
        instances = containers.Map();
        
        % 级别名称映射
        levelNames = {'DEBUG', 'INFO', 'WARN', 'ERROR'};
        
        % 级别颜色映射
        levelColors = {
            [0, 0.5, 0]     % DEBUG - 绿色
            [0, 0, 1]       % INFO - 蓝色
            [1, 0.5, 0]     % WARN - 橙色
            [1, 0, 0]       % ERROR - 红色
        };
    end
    
    methods
        function obj = BinomialLogger(name)
            % 构造函数，创建一个日志记录器实例
            %
            % 参数:
            %   name - 日志记录器名称
            
            obj.name = name;
            obj.level = obj.LEVEL_INFO;  % 默认级别为INFO
            obj.fileHandlers = {};
            obj.useConsole = true;       % 默认输出到控制台
            obj.useTimestamp = true;     % 默认显示时间戳
            obj.useLoggerName = true;    % 默认显示记录器名称
            obj.dateFormat = 'yyyy-mm-dd HH:MM:SS.FFF';  % 默认日期格式
        end
        
        function setLevel(obj, levelName)
            % 设置日志级别
            %
            % 参数:
            %   levelName - 级别名称 {'DEBUG', 'INFO', 'WARN', 'ERROR'}
            
            switch upper(levelName)
                case 'DEBUG'
                    obj.level = obj.LEVEL_DEBUG;
                case 'INFO'
                    obj.level = obj.LEVEL_INFO;
                case 'WARN'
                    obj.level = obj.LEVEL_WARN;
                case 'ERROR'
                    obj.level = obj.LEVEL_ERROR;
                otherwise
                    warning('无效的日志级别: %s，使用默认级别 INFO', levelName);
                    obj.level = obj.LEVEL_INFO;
            end
        end
        
        function addFileHandler(obj, filename, append)
            % 添加文件处理器
            %
            % 参数:
            %   filename - 日志文件路径
            %   append - 是否追加模式 (true/false, 默认为true)
            
            if nargin < 3
                append = true;
            end
            
            % 检查文件路径
            [folder, ~, ~] = fileparts(filename);
            if ~isempty(folder) && ~exist(folder, 'dir')
                try
                    mkdir(folder);
                catch ME
                    warning('无法创建日志目录: %s, %s', folder, ME.message);
                    return;
                end
            end
            
            % 创建文件处理器
            handler = struct('filename', filename, 'append', append);
            
            % 添加到处理器列表
            obj.fileHandlers{end+1} = handler;
            
            % 写入日志文件头
            try
                if append && exist(filename, 'file')
                    fid = fopen(filename, 'a');
                else
                    fid = fopen(filename, 'w');
                    fprintf(fid, '# 日志开始时间: %s\n', datestr(now, obj.dateFormat));
                    fprintf(fid, '# 级别 | 时间戳 | 记录器 | 消息\n');
                    fprintf(fid, '# ----------------------------------------\n');
                end
                
                if fid ~= -1
                    fclose(fid);
                end
            catch ME
                warning('%s', sprintf('初始化日志文件失败: %s', ME.message));
            end
        end
        
        function removeFileHandler(obj, filename)
            % 移除文件处理器
            %
            % 参数:
            %   filename - 要移除的日志文件路径
            
            for i = length(obj.fileHandlers):-1:1
                if strcmp(obj.fileHandlers{i}.filename, filename)
                    obj.fileHandlers(i) = [];
                    return;
                end
            end
        end
        
        function setConsoleOutput(obj, useConsole)
            % 设置是否输出到控制台
            %
            % 参数:
            %   useConsole - 布尔值，true表示输出到控制台
            
            obj.useConsole = logical(useConsole);
        end
        
        function setTimestampDisplay(obj, useTimestamp)
            % 设置是否显示时间戳
            %
            % 参数:
            %   useTimestamp - 布尔值，true表示显示时间戳
            
            obj.useTimestamp = logical(useTimestamp);
        end
        
        function setLoggerNameDisplay(obj, useLoggerName)
            % 设置是否显示记录器名称
            %
            % 参数:
            %   useLoggerName - 布尔值，true表示显示记录器名称
            
            obj.useLoggerName = logical(useLoggerName);
        end
        
        function setDateFormat(obj, format)
            % 设置日期格式
            %
            % 参数:
            %   format - 日期格式字符串
            
            try
                % 测试格式是否有效
                datestr(now, format);
                obj.dateFormat = format;
            catch
                warning('无效的日期格式: %s，保持原有格式', format);
            end
        end
        
        function debug(obj, message, varargin)
            % 记录DEBUG级别日志
            %
            % 参数:
            %   message - 日志消息，可包含格式说明符
            %   varargin - 格式说明符对应的值
            
            obj.log(obj.LEVEL_DEBUG, message, varargin{:});
        end
        
        function info(obj, message, varargin)
            % 记录INFO级别日志
            %
            % 参数:
            %   message - 日志消息，可包含格式说明符
            %   varargin - 格式说明符对应的值
            
            obj.log(obj.LEVEL_INFO, message, varargin{:});
        end
        
        function warn(obj, message, varargin)
            % 记录WARN级别日志
            %
            % 参数:
            %   message - 日志消息，可包含格式说明符
            %   varargin - 格式说明符对应的值
            
            obj.log(obj.LEVEL_WARN, message, varargin{:});
        end
        
        function error(obj, message, varargin)
            % 记录ERROR级别日志
            %
            % 参数:
            %   message - 日志消息，可包含格式说明符
            %   varargin - 格式说明符对应的值
            
            obj.log(obj.LEVEL_ERROR, message, varargin{:});
        end
    end
    
    methods (Access = protected)
        function log(obj, level, message, varargin)
            % 记录日志
            %
            % 参数:
            %   level - 日志级别
            %   message - 日志消息，可包含格式说明符
            %   varargin - 格式说明符对应的值
            
            % 检查日志级别
            if level < obj.level
                return;
            end
            
            % 格式化消息
            try
                if ~isempty(varargin)
                    formattedMessage = sprintf(message, varargin{:});
                else
                    formattedMessage = message;
                end
            catch
                formattedMessage = sprintf('格式化日志消息失败: %s', message);
            end
            
            % 获取当前时间
            timestamp = datestr(now, obj.dateFormat);
            
            % 获取级别名称
            levelName = obj.levelNames{level};
            
            % 构建完整日志消息
            logEntry = '';
            
            if obj.useTimestamp
                logEntry = sprintf('[%s] ', timestamp);
            end
            
            logEntry = sprintf('%s[%s] ', logEntry, levelName);
            
            if obj.useLoggerName
                logEntry = sprintf('%s[%s] ', logEntry, obj.name);
            end
            
            logEntry = [logEntry, formattedMessage];
            
            % 输出到控制台
            if obj.useConsole
                switch level
                    case obj.LEVEL_DEBUG
                        fprintf('<strong>%s</strong>\n', logEntry);
                    case {obj.LEVEL_INFO, obj.LEVEL_WARN, obj.LEVEL_ERROR}
                        fprintf('<strong><font color="%s">%s</font></strong>\n', ...
                            rgb2hex(obj.levelColors{level}), logEntry);
                end
            end
            
            % 写入日志文件
            for i = 1:length(obj.fileHandlers)
                handler = obj.fileHandlers{i};
                
                try
                    if handler.append && exist(handler.filename, 'file')
                        fid = fopen(handler.filename, 'a');
                    else
                        fid = fopen(handler.filename, 'w');
                    end
                    
                    if fid ~= -1
                        fprintf(fid, '%s | %s | %s | %s\n', ...
                            levelName, timestamp, obj.name, formattedMessage);
                        fclose(fid);
                    else
                        if obj.useConsole
                            warning('%s', sprintf('无法打开日志文件: %s', handler.filename));
                        end
                    end
                catch ME
                    if obj.useConsole
                        warning('%s', sprintf('写入日志文件失败: %s', ME.message));
                    end
                end
            end
        end
    end
    
    methods (Static)
        function logger = getLogger(name)
            % 获取日志记录器实例
            %
            % 参数:
            %   name - 日志记录器名称
            %
            % 返回值:
            %   logger - BinomialLogger实例
            
            persistent instances;
            
            if isempty(instances)
                instances = containers.Map();
            end
            
            if ~isKey(instances, name)
                logger = BinomialLogger(name);
                instances(name) = logger;
            else
                logger = instances(name);
            end
        end
    end
end

function hex = rgb2hex(rgb)
    % 将RGB颜色转换为十六进制颜色代码
    %
    % 参数:
    %   rgb - RGB颜色向量 [r, g, b]，取值范围为[0, 1]
    %
    % 返回值:
    %   hex - 十六进制颜色代码字符串
    
    r = round(rgb(1) * 255);
    g = round(rgb(2) * 255);
    b = round(rgb(3) * 255);
    
    hex = sprintf('#%02X%02X%02X', r, g, b);
end