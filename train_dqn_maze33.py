import numpy as np
from mymaze_env import Maze
from DQN_modified import DeepQNetwork
import tensorflow as tf
from matplotlib import pyplot as plt




def train_dqn(seed, file):
    # maze game
    MAZE_H = 3
    MAZE_W = 3
    hell_coord = [0, 2]
    door_coord = [2, 2]
    oval_coord = [1, 1]

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # maze game
    env = Maze(MAZE_H, MAZE_W, hell_coord, door_coord, oval_coord)
    max_episode = 10000
    dqn_start = 5000
    dqn = DeepQNetwork(env.n_actions, env.n_features, 'dqn',
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
                              prioritized_replay_beta_iters=1e5,
                              prioritized_replay_eps=1e-6
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
        score = 0
        while True:
            env.render()
            a = dqn.choose_action(s, test=True)
            s_, r, done = env.step(a)
            score += r
            if done:
                break
            s = s_

        return score

    def run_maze(max_episode):
        step = 0
        score_list = []
        avescore_list = []
        testscore_list = []
        flag = [False, False, False, False]
        for episode in range(max_episode):
            # initial observation
            s = env.reset()
            score = 0
            while True:
                env.render()
                a = dqn.choose_action(s)
                s_, r, done = env.step(a)
                score += r
                dqn.memory.add(s, a, r, s_, done)

                if (step > dqn_start) and (step % 5 == 0):
                    dqn.learn()
                if step == dqn_start:
                    print('\ndqn start learn~~~~~~~~~~~~~~~~~~~~')

                s = s_
                step = step + 1
                if done:
                    break

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
                if step > dqn_start:
                    print('dqn loss:', np.mean(dqn.cost_his[np.max(
                        [0, dqn.learn_step_counter - 100]):dqn.learn_step_counter]))

        # end of game
        print('game over')
        env.destroy()
        plt.plot(range(len(avescore_list)), avescore_list)
        plt.show()
        np.savetxt("E:/my_py_project/rl/log/maze33/dqn/SEED_%d_avescore_log.txt" % (seed), avescore_list)
        np.savetxt("E:/my_py_project/rl/log/maze33/dqn/SEED_%d_testscore_log.txt" % (seed), testscore_list)

    env.after(100, run_maze, max_episode)
    env.mainloop()
    dqn.plot_cost()

if __name__ == "__main__":
    # seed = 337
    seed = np.random.randint(10000, size=1)
    print('SEED=', seed)
    file = r'E:\my_py_project\rl\log\maze33_dqn_score.txt'
    with open(file, 'a+') as f:
        f.write('\nSEED=' + str(seed) + '\n')
    train_dqn(seed=seed, file=file)

