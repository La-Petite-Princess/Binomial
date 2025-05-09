%% parallel_module.m - 并行计算模块
classdef parallel_module
    methods(Static)
        function pool = setup_parallel_pool()
            % 检查是否已存在并行池
            if isempty(gcp('nocreate'))
                logger.log_message('info', '正在创建并行计算池...');
                try
                    % 获取集群对象
                    c = parcluster('local');
                    
                    % 检查NumWorkers属性
                    if c.NumWorkers == 0
                        % 设置最大工作进程数
                        c.NumWorkers = min(feature('numcores'), 512);
                        logger.log_message('info', sprintf('将NumWorkers设置为%d', c.NumWorkers));
                    end
                    
                    % 获取逻辑核心数
                    poolSize = feature('numcores');
                    
                    % 确保请求的工作进程数不超过最大值
                    poolSize = min(poolSize, c.NumWorkers);
                    
                    % 创建并行池
                    pool = parpool(c, poolSize);
                    logger.log_message('info', sprintf('成功创建%d个工作进程的并行池', pool.NumWorkers));
                catch ME
                    logger.log_message('warning', sprintf('创建并行池失败: %s', ME.message));
                    logger.log_message('info', '继续使用串行计算...');
                    pool = [];
                end
            else
                pool = gcp('nocreate');
                logger.log_message('info', sprintf('使用现有的并行池（%d个工作进程）', pool.NumWorkers));
            end
        end
    end
end
