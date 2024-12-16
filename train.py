import numpy as np
import time
from common import get_args, experiment_setup
import wandb
import torch 

if __name__ == '__main__':
    args = get_args()
    env, agent, buffer, learner, tester = experiment_setup(args)
    wandb.login(
    
    key='b98d2b806f364f5af900550ec98e26e2f418e8a7'
    )
    wandb.init(
        project='mujoco',
        config={
            'env':args.env,
            'alg':args.alg,
            'rrd_batch_size':args.rrd_batch_size,
            'rrd_sample_size':args.rrd_sample_size,
            'alpha':args.alpha,
            'train_batches':args.train_batches,
            'q_lr':args.q_lr,
            'Adam_eps':args.Adam_eps,
            'double':args.double,
            'dueling':args.dueling,
            'chi2_coeff':args.chi2_coeff,
            'rrd_bias_correction':args.rrd_bias_correction,
            'polyak':args.polyak,
                } )
    
    
    if args.code=='pytorch':
        args.logger.summary_init()
        
    if args.code=='tf':
        args.logger.summary_init(agent.graph, agent.sess)
        args.logger.summary_setup()

    # Progress info
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('Episodes@green')
    args.logger.add_item('Timesteps')
    args.logger.add_item('TimeCost(sec)/train')
    args.logger.add_item('TimeCost(sec)/test')

    # Algorithm info
    for key in agent.train_info.keys():
        args.logger.add_item(key, 'scalar')
    for key in learner.learner_info:
        args.logger.add_item(key, 'scalar')

    # Test info
    for key in agent.step_info.keys():
        args.logger.add_item(key, 'scalar')
    for key in env.env_info.keys():
        args.logger.add_item(key, 'scalar')
    for key in tester.info:
        args.logger.add_item(key, 'scalar')

    

    episodes_cnt = 0
    for epoch in range(args.epochs):
        for cycle in range(args.cycles):
            args.logger.tabular_clear()
            args.logger.summary_clear()

            start_time = time.time()
            learner.learn(args, env, agent, buffer)
            
            # Ghi lại thời gian huấn luyện
            args.logger.add_record('TimeCost(sec)/train', time.time() - start_time)

            start_time = time.time()
            tester.cycle_summary()
            # Ghi lại thời gian kiểm thử
            args.logger.add_record('TimeCost(sec)/test', time.time() - start_time)

            # Ghi lại thông tin Epoch và Cycle
            args.logger.add_record('Epoch', f'{epoch}/{args.epochs}')
            args.logger.add_record('Cycle', f'{cycle}/{args.cycles}')
            args.logger.add_record('Episodes', learner.ep_counter)
            args.logger.add_record('Timesteps', learner.step_counter)

            # Hiển thị bảng kết quả
            args.logger.tabular_show(args.tag)
            # Ghi log lên TensorBoard
            args.logger.summary_show(learner.step_counter)
      
        tester.epoch_summary()

    tester.final_summary()
    wandb.finish()
