classdef BinomialLogger < handle
    % 日志管理类：负责所有日志记录功能
    % 支持多种日志级别、文件和控制台输出
    
    properties (Access = private)
        LogFile
        LogLevel
        FigureSaveLevel
        Console
        LogBuffer
        BufferSize
    end
    
    properties (Constant)
        LEVELS = struct(...
            'debug', 0, ...
            'info', 1, ...
            'warning', 2, ...
            'error', 3 ...
        )
        
        LEVEL_NAMES = {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
        
        % ANSI颜色代码
        COLORS = struct(...
            'debug', '\033[0;34m', ...  % 蓝色
            'info', '\033[0;32m', ...   % 绿色
            'warning', '\033[1;33m', ...% 黄色
            'error', '\033[1;31m', ...  % 红色
            'reset', '\033[0m' ...      % 重置
        )
    end
    
    methods (Access = public)
        function obj = BinomialLogger(log_file, log_level, figure_save_level)
            % 构造函数
            % 输入:
            %   log_file - 日志文件路径
            %   log_level - 日志级别 ('debug', 'info', 'warning', 'error')
            %   figure_save_level - 图形保存日志级别（可选）
            
            if nargin < 2
                log_level = 'info';
            end
            
            if nargin < 3
                % 图形保存日志级别比一般日志级别高一级
                switch lower(log_level)
                    case 'debug'
                        figure_save_level = 'info';
                    case 'info'
                        figure_save_level = 'warning';
                    otherwise
                        figure_save_level = log_level;
                end
            end
            
            obj.LogFile = log_file;
            obj.LogLevel = lower(log_level);
            obj.FigureSaveLevel = lower(figure_save_level);
            obj.Console = true;
            obj.BufferSize = 100;
            obj.LogBuffer = cell(obj.BufferSize, 1);
            
            % 初始化日志文件
            obj.InitializeLogFile();
            
            % 记录启动信息
            obj.Log('info', '=== 日志系统初始化 ===');
            obj.Log('info', sprintf('日志级别: %s (图形保存级别: %s)', ...
                upper(obj.LogLevel), upper(obj.FigureSaveLevel)));
        end
        
        function delete(obj)
            % 析构函数：确保缓冲区内容写入文件
            obj.FlushBuffer();
        end
        
        function Log(obj, level, message)
            % 记录日志
            % 输入:
            %   level - 日志级别
            %   message - 日志信息
            
            level = lower(level);
            
            % 检查是否应记录此级别的日志
            if ~obj.ShouldLog(level)
                return;
            end
            
            % 生成时间戳
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
            
            % 获取调用函数信息
            stack = dbstack('-completenames');
            if length(stack) > 1
                % 跳过当前函数，获取调用者信息
                caller = stack(2);
                [~, caller_func, ~] = fileparts(caller.name);
                caller_info = sprintf('%s:%d', caller_func, caller.line);
            else
                caller_info = 'unknown';
            end
            
            % 格式化日志消息
            level_name = upper(level);
            formatted_message = sprintf('[%s] [%s] [%s] %s', ...
                timestamp, level_name, caller_info, message);
            
            % 输出到控制台（带颜色）
            if obj.Console && obj.ShouldOutputToConsole(level)
                if ispc
                    % Windows系统不支持ANSI颜色
                    fprintf('%s\n', formatted_message);
                else
                    % Unix系统支持ANSI颜色
                    color = obj.COLORS.(level);
                    reset = obj.COLORS.reset;
                    fprintf('%s%s%s\n', color, formatted_message, reset);
                end
            end
            
            % 添加到缓冲区
            obj.AddToBuffer(formatted_message);
        end
        
        function LogPerformance(obj, operation, duration, additional_info)
            % 记录性能信息
            % 输入:
            %   operation - 操作名称
            %   duration - 持续时间（秒）
            %   additional_info - 额外信息（可选）
            
            if nargin < 4
                additional_info = '';
            end
            
            if ~isempty(additional_info)
                message = sprintf('性能: %s 完成，耗时: %.2f秒 (%s)', ...
                    operation, duration, additional_info);
            else
                message = sprintf('性能: %s 完成，耗时: %.2f秒', ...
                    operation, duration);
            end
            
            obj.Log('info', message);
        end
        
        function LogMemoryUsage(obj)
            % 记录内存使用情况
            if ispc
                % Windows系统
                [~, sys_mem] = memory();
                total_mem = sys_mem.TotalPhys / (1024^3);  % GB
                used_mem = (sys_mem.TotalPhys - sys_mem.AvailPhys) / (1024^3);  % GB
                free_mem = sys_mem.AvailPhys / (1024^3);  % GB
                
                obj.Log('debug', sprintf('内存使用: 总计%.2fGB, 已用%.2fGB, 可用%.2fGB', ...
                    total_mem, used_mem, free_mem));
            else
                % Unix系统
                feature('memstats');  % 可能需要root权限
            end
        end
        
        function LogProgress(obj, current, total, description)
            % 记录进度信息（进度条式）
            % 输入:
            %   current - 当前进度
            %   total - 总进度
            %   description - 描述信息
            
            percentage = current / total * 100;
            progress_bar_width = 50;
            filled_width = round(progress_bar_width * current / total);
            bar = ['[' repmat('=', filled_width, 1) repmat(' ', progress_bar_width - filled_width, 1) ']'];
            
            message = sprintf('%s %s %.1f%% (%d/%d)', description, bar, percentage, current, total);
            
            % 使用回车符实现覆盖输出（仅适用于console）
            if obj.Console
                fprintf('\r%s', message);
                if current == total
                    fprintf('\n'); % 完成时换行
                end
            end
            
            % 只在特定进度点记录到文件
            if mod(current, max(1, round(total / 10))) == 0 || current == total
                obj.Log('debug', message);
            end
        end
        
        function LogException(obj, exception, context)
            % 记录异常信息
            % 输入:
            %   exception - MException对象
            %   context - 上下文信息
            
            obj.Log('error', sprintf('异常发生在 %s: %s', context, exception.message));
            
            % 记录堆栈跟踪
            for i = 1:length(exception.stack)
                frame = exception.stack(i);
                obj.Log('error', sprintf('  在 %s (行 %d): %s', ...
                    frame.name, frame.line, frame.file));
            end
            
            % 如果有cause，也记录它
            if ~isempty(exception.cause)
                obj.Log('error', '原因:');
                for i = 1:length(exception.cause)
                    obj.LogException(exception.cause{i}, '嵌套异常');
                end
            end
        end
        
        function SetLogLevel(obj, level)
            % 设置日志级别
            obj.LogLevel = lower(level);
            obj.Log('info', sprintf('日志级别已更改为: %s', upper(level)));
        end
        
        function SetFigureSaveLevel(obj, level)
            % 设置图形保存日志级别
            obj.FigureSaveLevel = lower(level);
            obj.Log('info', sprintf('图形保存日志级别已更改为: %s', upper(level)));
        end
        
        function CreateSection(obj, title)
            % 创建日志分节
            separator = repmat('=', 1, 60);
            obj.Log('info', separator);
            obj.Log('info', sprintf(' %s ', title));
            obj.Log('info', separator);
        end
        
        function Summary = GetSummary(obj)
            % 获取日志摘要
            Summary = struct();
            Summary.LogFile = obj.LogFile;
            Summary.LogLevel = obj.LogLevel;
            Summary.FigureSaveLevel = obj.FigureSaveLevel;
            
            % 统计不同级别的日志数量
            if exist(obj.LogFile, 'file')
                fid = fopen(obj.LogFile, 'r');
                if fid ~= -1
                    log_content = fread(fid, '*char')';
                    fclose(fid);
                    
                    % 计算各级别出现次数
                    for i = 1:length(obj.LEVEL_NAMES)
                        level_name = obj.LEVEL_NAMES{i};
                        count = length(regexp(log_content, ['\[' level_name '\]']));
                        Summary.(sprintf('%sCount', level_name)) = count;
                    end
                end
            end
        end
    end
    
    methods (Access = private)
        function InitializeLogFile(obj)
            % 初始化日志文件
            [dir_path, ~, ~] = fileparts(obj.LogFile);
            if ~exist(dir_path, 'dir')
                mkdir(dir_path);
            end
            
            % 如果文件已存在，备份旧文件
            if exist(obj.LogFile, 'file')
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                [dir_path, name, ext] = fileparts(obj.LogFile);
                backup_file = fullfile(dir_path, sprintf('%s_%s%s', name, timestamp, ext));
                copyfile(obj.LogFile, backup_file);
            end
            
            % 创建新文件
            fid = fopen(obj.LogFile, 'w');
            if fid == -1
                error('无法创建日志文件: %s', obj.LogFile);
            end
            fclose(fid);
        end
        
        function should_log = ShouldLog(obj, level)
            % 判断是否应记录此级别的日志
            should_log = obj.LEVELS.(level) >= obj.LEVELS.(obj.LogLevel);
        end
        
        function should_output = ShouldOutputToConsole(obj, level)
            % 判断是否应输出到控制台
            should_output = obj.Console && obj.ShouldLog(level);
        end
        
        function AddToBuffer(obj, message)
            % 添加消息到缓冲区
            % 使用循环缓冲区
            persistent buffer_index;
            if isempty(buffer_index)
                buffer_index = 1;
            end
            
            obj.LogBuffer{buffer_index} = message;
            buffer_index = mod(buffer_index, obj.BufferSize) + 1;
            
            % 定期刷新缓冲区
            if buffer_index == 1
                obj.FlushBuffer();
            end
        end
        
        function FlushBuffer(obj)
            % 将缓冲区内容写入文件
            try
                fid = fopen(obj.LogFile, 'a');
                if fid ~= -1
                    for i = 1:obj.BufferSize
                        if ~isempty(obj.LogBuffer{i})
                            fprintf(fid, '%s\n', obj.LogBuffer{i});
                        end
                    end
                    fclose(fid);
                    % 清空缓冲区
                    obj.LogBuffer = cell(obj.BufferSize, 1);
                end
            catch ME
                warning('写入日志文件失败: %s', ME.message);
            end
        end
    end
end