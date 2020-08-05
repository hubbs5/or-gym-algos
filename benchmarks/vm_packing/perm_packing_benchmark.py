#!usr/bin/env python

from algos.vm_packing.math_prog import build_online_vm_opt
from algos.vm_packing.heuristics import *
from algos.math_prog_utils import solve_shrinking_horizon_mp, solve_math_program, extract_vm_packing_plan
import or_gym
import sys
from argparse import ArgumentParser
from str2bool import str2bool
import re
import pandas as pd
import numpy as np

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--n_tests', type=int, default=100,
        help='Set number of tests to get sample size.')
    parser.add_argument('--print', type=str2bool, default=True,
        help='Print output.')
    parser.add_argument('--solver', type=str, default='glpk')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_known_args()

def optimize_vmp_perm(env, solver='glpk', print_output=True):
    # Run iteratively to make model tractable
    env.reset()
    model, actions, rewards, results = solve_shrinking_horizon_mp(env, build_online_vm_opt,
        extract_vm_packing_plan, solver, print_output)
    return model, actions, rewards, results

if __name__ == '__main__':
    env_name = 'VMPacking-v0'
    args, unknown = parse_arguments()
    env_config = {re.sub('--', '', unknown[i]): unknown[i+1] 
        for i in range(len(unknown)) if i % 2 == 0}
    env_config.update(args.__dict__)

    env = or_gym.make(env_name, env_config=env_config)
    print('N Machines:\t{}'.format(env.n_pms))
    print('N Steps:\t{}'.format(env.step_limit))
    test_results = np.zeros((2, args.n_tests))

    print("\nRunning optimization:\n")
    count = 0
    avg_opt_rewards = 0
    num_vars = 0
    num_cons = 0
    time_to_solve = 0
    for i in range(args.n_tests):
        # try:
        opt_model, opt_actions, opt_rewards, results = optimize_vmp_perm(env, solver=args.solver, print_output=args.print)
        test_results[0, count] = sum(opt_rewards)
        avg_opt_rewards += (opt_model.obj.expr() - avg_opt_rewards) / (count + 1)
        n_vars = np.array([j['Problem']()['Number of variables'] for j in results])
        n_cons = np.array([j['Problem']()['Number of constraints'] for j in results])
        t = np.array([j['Solver']()['Time'] for j in results])
        # num_vars += (n_vars - num_vars) / (count + 1)
        # num_cons += (n_cons - num_cons) / (count + 1)
        # time_to_solve += (t - time_to_solve) / (count + 1)
        if (count+1) % 10 == 0:
            print("Episodes Complete: {}\tMean:\t{:.1f}".format(count+1, test_results[0, :count].mean()))
        count += 1
        # except Exception as e:
            # pass
    try:
        print("Avg N Vars\t\t\t=\t{}\t{}".format(n_vars.mean(), n_vars.max()))
        print("Avg N Constraints\t\t=\t{}\t{}".format(n_cons.mean(), n_cons.max()))
        print("Avg time to Solve\t\t=\t{:.5f}\t{}".format(t.mean(), t.max()))
    except Error as e:
        print('Num variables\n')
        print(n_vars)
        print('\nNum Contstraints\n')
        print(n_cons)
        print('\nSolve time\n')
        print(t)

    # print("Average Optimal Reward\t\t=\t{}".format(avg_opt_rewards))
    print(results)

    # print("\nRunning heuristic:\n")
    # for i in range(args.n_tests):
    #     heur_actions, heur_rewards = first_fit_heuristic(env)
    #     test_results[1, i] = sum(heur_rewards)
    #     if (i+1) % 10 == 0:
    #         print("Episodes Complete: {}\tMean:\t{:.1f}".format(i+1, test_results[1, :i].mean()))
    # print("")
    # print("Optimization Results\n\tMean Rewards\t=\t{:.1f}\n\tStd Rewards\t=\t{:.1f}".format(
    #     test_results[0].mean(), np.std(test_results[0])))
    # print("Heuristic Results\n\tMean Rewards\t=\t{:.1f}\n\tStd Rewards\t=\t{:.1f}".format(
    #     test_results[1].mean(), np.std(test_results[1])))
    # # Save results
    # df = pd.DataFrame(test_results.T, columns=['optimization', 'heuristic'])
    # df.to_csv('temp_packing_results.csv', index=False)
