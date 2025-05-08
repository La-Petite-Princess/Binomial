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

%% utils.m - 工具函数模块
classdef utils
    methods(Static)
        function check_toolboxes()
            % 检查必要的工具箱是否已安装
            if ~license('test', 'statistics_toolbox')
                error('需要安装 Statistics and Machine Learning Toolbox');
            end
        end
        
        function data_gpu = toGPU(data)
            % 将数据转移到GPU(如果支持) - 为AMD 5500M GPU优化
            % 输入:
            %   data - 输入数据
            % 输出:
            %   data_gpu - GPU上的数据或原始数据
                
            persistent gpuAvailable gpuMemLimit;
                
            % 只检查一次GPU可用性
            if isempty(gpuAvailable)
                gpuAvailable = (exist('gpuArray', 'file') == 2) && gpuDeviceCount > 0;
                
                if gpuAvailable
                    gpu = gpuDevice();
                    % 为AMD GPU设置更保守的内存限制（总显存的60%）
                    gpuMemLimit = 0.6 * gpu.AvailableMemory;
                    
                    logger.log_message('info', sprintf('GPU可用: %s, 总内存: %.2f GB, 可用内存: %.2f GB', ...
                        gpu.Name, gpu.TotalMemory/1e9, gpu.AvailableMemory/1e9));
                else
                    gpuMemLimit = 0;
                    logger.log_message('info', 'GPU不可用，使用CPU计算');
                end
            end
                
            % 智能决策：根据数据大小和传输开销决定是否使用GPU
            if gpuAvailable
                try
                    % 计算数据大小(字节)
                    dataSize = numel(data) * 8; % 假设是双精度数据
                    
                    % 最小阈值：太小的数据传输开销大于收益(5MB)
                    minThreshold = 5 * 1024 * 1024;
                    
                    % 如果数据太大或太小，不使用GPU
                    if dataSize > gpuMemLimit || dataSize < minThreshold
                        data_gpu = data;
                        return;
                    end
                    
                    % 使用GPU
                    data_gpu = gpuArray(data);
                catch ME
                    logger.log_message('warning', sprintf('GPU转换失败: %s，使用CPU计算', ME.message));
                    data_gpu = data;
                end
            else
                data_gpu = data;
            end
        end
        
        function save_figure(fig, output_dir, filename_base, varargin)
            % 保存图形为多种格式
            % 输入:
            %   fig - 图形句柄
            %   output_dir - 输出目录
            %   filename_base - 文件名基础
            %   varargin - 可选参数
            
            % 解析输入参数
            p = inputParser;
            addParameter(p, 'Formats', {'svg'}, @(x) iscell(x) || ischar(x));
            addParameter(p, 'DPI', 300, @isnumeric);
            addParameter(p, 'MethodName', '', @ischar);
            parse(p, varargin{:});
            
            formats = p.Results.Formats;
            dpi = p.Results.DPI;
            method_name = p.Results.MethodName;
            
            % 如果formats是字符串，转换为cell数组
            if ischar(formats)
                formats = {formats};
            end
            
            % 确保输出目录存在
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            % 处理文件名中的方法名称替换
            if contains(filename_base, '%s')
                if ~isempty(method_name)
                    actual_filename = sprintf(filename_base, method_name);
                else
                    actual_filename = strrep(filename_base, '%s', 'unknown');
                    logger.log_message('warning', '文件名中包含%s但未提供method_name');
                end
            else
                actual_filename = filename_base;
            end
            
            % 准备图形
            set(fig, 'Color', 'white');
            
            % 检查是否存在exportgraphics函数(R2020a或更高版本)
            has_exportgraphics = exist('exportgraphics', 'file') == 2;
            
            % 初始化成功保存的格式列表
            successful_formats = {};
            
            % 保存每种格式
            for i = 1:length(formats)
                format = lower(formats{i});
                filename = fullfile(output_dir, [actual_filename '.' format]);
                
                try
                    if has_exportgraphics
                        % 使用exportgraphics函数(适用于R2020a或更高版本)
                        switch format
                            case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
                                % 位图格式
                                exportgraphics(fig, filename, 'Resolution', dpi);
                            case {'pdf', 'eps', 'svg'}
                                % 矢量格式
                                exportgraphics(fig, filename, 'ContentType', 'vector');
                            otherwise
                                % 其他格式回退到saveas
                                saveas(fig, filename);
                        end
                    else
                        % 回退到传统方法
                        % 禁用工具栏和菜单栏
                        set(fig, 'Toolbar', 'none');
                        set(fig, 'MenuBar', 'none');
                        
                        % 根据不同的格式选择不同的保存方法
                        switch format
                            case {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
                                print(fig, filename, ['-d' format], ['-r' num2str(dpi)]);
                            case 'pdf'
                                print(fig, filename, '-dpdf', '-bestfit');
                            case 'eps'
                                print(fig, filename, '-depsc2');
                            case 'svg'
                                print(fig, filename, '-dsvg');
                            otherwise
                                saveas(fig, filename);
                        end
                    end
                    
                    successful_formats{end+1} = upper(format);
                    
                catch ME
                    % 保存失败，记录错误并尝试备用方法
                    logger.log_message('debug', ['保存' upper(format) '格式时出错: ' ME.message]);
                    
                    % 尝试备用方法
                    try
                        saveas(fig, filename);
                        successful_formats{end+1} = upper(format);
                    catch ME2
                        logger.log_message('warning', ['备用保存方法也失败: ' ME2.message]);
                    end
                end
            end
            
            % 输出汇总日志消息
            if ~isempty(successful_formats)
                formats_str = strjoin(successful_formats, ', ');
                logger.log_message('info', sprintf('图形已保存: %s (%s)', actual_filename, formats_str));
            else
                logger.log_message('error', ['图形保存失败: ' actual_filename]);
            end
        end
    end
end