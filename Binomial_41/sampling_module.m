%% sampling_module.m - 采样模块
classdef sampling_module
    methods(Static)
        function [train_indices, test_indices] = bootstrap_sampling(y, train_ratio, n_samples)
            % 使用Bootstrap进行分层抽样
            % 输入:
            %   y - 因变量
            %   train_ratio - 训练集比例
            %   n_samples - 样本数量
            % 输出:
            %   train_indices - 训练集索引
            %   test_indices - 测试集索引
            
            t_start = toc;
            
            % 找出各类别的索引
            class_0_idx = find(y == 0);
            class_1_idx = find(y == 1);
            
            % 预计算常量避免在parfor中重复计算
            n0 = length(class_0_idx);
            n1 = length(class_1_idx);
            n0_train = round(train_ratio * n0);
            n1_train = round(train_ratio * n1);
            total_samples = length(y);
            
            % 预分配结果数组
            train_indices = cell(n_samples, 1);
            test_indices = cell(n_samples, 1);
            
            % 预分配随机种子数组确保并行迭代随机性
            rng_seeds = randi(1000000, n_samples, 1);
            
            % 创建逻辑索引数组提高效率
            total_mask = false(total_samples, n_samples);
            
            % 使用parfor并行处理
            parfor i = 1:n_samples
                % 设置当前迭代的随机种子
                rng(rng_seeds(i));
                
                % 对每个类别进行分层抽样
                train_idx_0 = class_0_idx(randsample(n0, n0_train));
                train_idx_1 = class_1_idx(randsample(n1, n1_train));
                
                % 合并训练集索引
                train_idx = [train_idx_0; train_idx_1];
                
                % 使用逻辑索引代替setdiff提高性能
                mask = false(total_samples, 1);
                mask(train_idx) = true;
                
                % 存储训练集和测试集
                train_indices{i} = train_idx;
                test_indices{i} = find(~mask);
                
                % 存储total_mask
                total_mask(:, i) = mask;
            end
            
            % 输出Bootstrap样本的统计信息
            train_sizes = cellfun(@length, train_indices);
            test_sizes = cellfun(@length, test_indices);
            logger.log_message('info', sprintf('Bootstrap样本统计: 平均训练集大小=%.1f, 平均测试集大小=%.1f', ...
                mean(train_sizes), mean(test_sizes)));
            
            % 计算样本覆盖率
            coverage = mean(sum(total_mask, 2) > 0);
            logger.log_message('info', sprintf('数据覆盖率: %.2f%%', coverage * 100));
            
            t_end = toc;
            logger.log_message('info', sprintf('Bootstrap抽样完成，生成了%d个训练/测试集，耗时：%.2f秒', ...
                length(train_indices), t_end - t_start));
        end
    end
end