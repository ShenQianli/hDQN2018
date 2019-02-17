import numpy as np
from dsdp_env import dsdp
from DQN_modified import DeepQNetwork
import tensorflow as tf
from matplotlib import pyplot as plt




def train_hdqnm(seed, file):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # dsdp game
    env = dsdp()
    n_goals = 6
    max_episode = 20000
    controller_start = 5000
    meta_controller_start = 10000
    controller = DeepQNetwork(env.n_actions, env.n_features + 1, 'controller',
                              optimizer='rmsprop',
                              momentum=0.9,
                              learning_rate=0.00025,
                              opt_decay=0.99,
                              reward_decay=0.99,
                              e_greedy=0,
                              e_greedy_max=0.99,
                              e_greedy_increment=1e-4,
                              e_greedy_iter=5e3,
                              replace_target_iter=200,
                              memory_size=10000,
                              output_graph=False,
                              prioritized_replay=True,
                              prioritized_replay_alpha=0.6,
                              prioritized_replay_beta0=0.4,
                              prioritized_replay_beta_iters=5e3,
                              prioritized_replay_eps=1e-6
                              )

    meta_controller = DeepQNetwork(n_goals, env.n_features, 'meta_controller',
                                   optimizer='rmsprop',
                                   momentum=0.9,
                                   learning_rate=1e-3,
                                   opt_decay=0.99,
                                   reward_decay=0.99,
                                   e_greedy=0,
                                   e_greedy_max=0.99,
                                   e_greedy_increment=1e-4,
                                   e_greedy_iter=5e3,
                                   replace_target_iter=200,
                                   memory_size=10000,
                                   output_graph=False,
                                   prioritized_replay=True,
                                   prioritized_replay_alpha=0.6,
                                   prioritized_replay_beta0=0.4,
                                   prioritized_replay_beta_iters=5e3,
                                   prioritized_replay_eps=1e-6,
                                   )

    def play_maze():
        # def goal_reached(g, s):
        #    if(g == 0):
        #        return (s[0]==0)
        #    else:
        #        return ((s[2*g-1]==0) and (s[2*g]==0))
        def goal_reached(g, s):
            return (s[1] == g)

        s = env.reset()
        g = meta_controller.choose_action(s, test=True)
        score = 0
        while True:
            env.render()
            a = controller.choose_action(np.hstack((s, g)), test=True)
            s_, r, done = env.step(a)
            score += r
            if done:
                break
            s = s_
            if goal_reached(g, s):
                g = meta_controller.choose_action(s, test=True)
        return score

    def run_maze(max_episode):
        def goal_reached(g, s):
            return (s[1] == g)

        def reward(g, s_, done):
            # return 1+1/episode_step if goal_reached(g, s_) else 0

            # if done and not goal_reached(g, s_): return -1
            return 1 if goal_reached(g, s_) else -1

            # return -goal_distance(g, s_)

            # if goal_reached(g, s_):
            #     return 1
            # else:
            #     return -1

            # for i in range(n_goals):
            #    if goal_reached(i, s_):
            #        if i == g: return 1
            #        else: return -1
            # return 0

        step = 0
        score_list = []
        avescore_list = []
        testscore_list = []
        flag = [False, False, False, False]
        for episode in range(max_episode):
            # initial observation
            s = env.reset()
            score = 0
            g = meta_controller.choose_action(s)
            Done = False
            while True:
                F = 0
                s0 = s
                episode_step = 1
                while True:
                    env.render()
                    a = controller.choose_action(np.hstack((s, g)))
                    s_, f, done = env.step(a)
                    score += f
                    r = reward(g, s_, done)
                    controller.memory.add(np.hstack((s, g)), a, r, np.hstack((s_, g)), done)
                    if (step > controller_start) and (step % 5 == 0):
                        controller.learn()
                    # if (step > meta_controller_start) and (step % 5 == 0):
                    #    meta_controller.learn()

                    if step == meta_controller_start:
                        print('\nmeta controller start learn~~~~~~~~~~~~~~~~~~~~')
                    if step == controller_start:
                        print('\ncontroller start learn~~~~~~~~~~~~~~~~~~~~~~~~~~')

                    F = F + f
                    s = s_
                    step = step + 1

                    if done:
                        Done = True
                        break
                    if goal_reached(g, s):
                        # reached_goal_list.append()
                        break

                    episode_step = episode_step + 1
                if goal_reached(g, s):
                    meta_controller.memory.add(s0, g, F, s, done)
                if step > meta_controller_start:
                    meta_controller.learn()
                if Done:
                    break
                g = meta_controller.choose_action(s)
            # print(step)
            score_list.append(score)
            if (episode > 0 and episode % 50 == 0):
                # average score
                avescore = np.average(np.array(score_list[-50:]))
                avescore_list.append(avescore)
                print("\nepisode %d : average score = %f" % (episode, avescore))
                # test score
                testscore = 0
                for i in range(5):
                    testscore += play_maze()
                testscore /= 5
                testscore_list.append(testscore)
                print("episode %d : test score = %f\n" % (episode, testscore))

                # logs
                if avescore > 0.1 and not flag[0]:
                    flag[0] = True
                    with open(file, 'a+') as f:
                        f.write('train average score achieves 0.1 (' + str(avescore) + ')\n')

                    #print('game over')
                    #env.destroy()
                    #return
                elif avescore > 0.02 and not flag[1]:
                    flag[1] = True
                    with open(file, 'a+') as f:
                        f.write('train average score achieves 0.02 (' + str(avescore) + ')\n')

                if testscore > 0.1 and not flag[2]:
                    flag[2] = True
                    with open(file, 'a+') as f:
                        f.write('test score achieves 0.1 (' + str(testscore) + ')\n')
                elif testscore > 0.02 and not flag[3]:
                    flag[3] = True
                    with open(file, 'a+') as f:
                        f.write('test score achieves 0.02 (' + str(testscore) + ')\n')

            # loss
            if (episode > 0 and episode % 10 == 0):
                if step > controller_start:
                    print('controller loss:', np.mean(controller.cost_his[np.max(
                        [0, controller.learn_step_counter - 100]):controller.learn_step_counter]))
                if step > meta_controller_start:
                    print('meta controller loss:', np.mean(meta_controller.cost_his[np.max(
                        [0, meta_controller.learn_step_counter - 100]):meta_controller.learn_step_counter]))
        # end of game
        print('game over')
        env.destroy()
        plt.plot(range(len(avescore_list)), avescore_list)
        plt.show()
        np.savetxt("E:/my_py_project/rl/log/dsdp/hdqnm/SEED_%d_avescore_log.txt" % (seed), avescore_list)
        np.savetxt("E:/my_py_project/rl/log/dsdp/hdqnm/SEED_%d_testscore_log.txt" % (seed), testscore_list)

    env.after(100, run_maze, max_episode)
    env.mainloop()
    controller.plot_cost()
    meta_controller.plot_cost()

if __name__ == "__main__":
    seed = 5839
    # seed = np.random.randint(10000, size=1)
    print('SEED=', seed)
    file = r'E:\my_py_project\rl\log\dsdp_hdqnm_score.txt'
    with open(file, 'a+') as f:
        f.write('\nSEED=' + str(seed) + '\n')
    train_hdqnm(seed=seed, file=file)

