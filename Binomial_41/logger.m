%% logger.m - 日志模块
classdef logger
    methods(Static)
        function log_message(level, message)
            % 记录日志消息
            % 输入:
            %   level - 日志级别 ('debug', 'info', 'warning', 'error')
            %   message - 日志消息
            
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
        
        function log_system_info()
            % 记录系统信息
            
            logger.log_message('info', '系统配置:');
            logger.log_message('info', sprintf('- CPU: Intel i9-9980HK (8核16线程)'));
            logger.log_message('info', sprintf('- 内存: 64GB RAM'));
            logger.log_message('info', sprintf('- GPU: AMD Radeon Pro 5500M 8GB'));
        end
    end
end
