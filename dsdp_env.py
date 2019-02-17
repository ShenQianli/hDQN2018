"""
Reinforcement learning Discrete stochastic dicision process with delayed reward example.

Red rectangle:          explorer.
White rectangle:        stations.

In state s_i, the discrete action space is [0, 1].
When agent choose action 0, explorer go to state s_i+1.
Otherwise if action 1 chosen, explorer go to state s_i-1 or s_i+1 with equal possibility.
Initial state is s_2 and terminal state is station s_1.
The reward at the terminal state depends on whether s_n is visited(r = 1) or not(r = 0.01)


"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
n_states = 6


class dsdp(tk.Tk, object):
    def __init__(self):
        super(dsdp, self).__init__()
        self.action_space = [0, 1]
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('discrete stochastic dicision process')
        self.geometry('{0}x{1}'.format(n_states * UNIT, UNIT))
        self._build_dsdp()

    def _build_dsdp(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=UNIT,
                           width=n_states * UNIT)

        # create grids
        for c in range(0, n_states * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, n_states * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        self.origin = np.array([60, 20])
        origin = self.origin

        # create oval
        oval_center = origin + np.array([UNIT * (n_states - 2), 0])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        
        self.terminal_coords = [5, 5, 35, 35]
        
        self.flag = 1

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        #time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = self.origin
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        oval_center = origin + np.array([UNIT * (n_states - 2), 0])
        if(self.flag == 0):
            self.oval = self.canvas.create_oval(
                oval_center[0] - 15, oval_center[1] - 15,
                oval_center[0] + 15, oval_center[1] + 15,
                fill='yellow')
        self.flag = 1
        self.counter = 0
        # return observation
        return np.array([self.flag, 1 + (self.canvas.coords(self.rect)[1] - self.origin[1]) / UNIT])

    def step(self, action):
        self.counter += 1
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 1:   # 50% left, 50% right
            temp = np.random.uniform()
            if(temp < 0.5 and s[0] > UNIT):
                base_action[0] -= UNIT
            if(temp >= 0.5 and s[0] < (n_states - 1) * UNIT):
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.terminal_coords:
            done = True
            reward = 0.01 if self.flag else 1
        elif(self.flag == 1 and next_coords == self.canvas.coords(self.oval)):
            self.flag = 0
            self.canvas.delete(self.oval)
            reward = 0
            done = False
        else:
            reward = 0
            done = False
        if self.counter > 100:
            done = True
        s_ = np.array([self.flag, 1 + (self.canvas.coords(self.rect)[1] - self.terminal_coords[1]) / UNIT])
        self.render()
        return s_, reward, done

    def render(self):
        #time.sleep(0.01)
        self.update()


