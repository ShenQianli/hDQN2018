"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width


class Maze(tk.Tk, object):
    def __init__(self, MAZE_H, MAZE_W, hell_coord, door_coord, oval_coord):
        super(Maze, self).__init__()
        self.MAZE_W = MAZE_W
        self.MAZE_H = MAZE_H
        self.hell_coord = hell_coord
        self.door_coord = door_coord
        self.oval_coord = oval_coord
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 7
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.counter = 0


    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.MAZE_H * UNIT,
                           width=self.MAZE_W * UNIT)

        # create grids
        for c in range(0, self.MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([self.hell_coord[0]*UNIT, UNIT * self.hell_coord[1]])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        
        # hell2_center = origin + np.array([UNIT * 2, 0])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')
            
        # door
        door_center = origin + np.array([UNIT * self.door_coord[0], UNIT * self.door_coord[1]])
        self.door = self.canvas.create_rectangle(
            door_center[0] - 15, door_center[1] - 15,
            door_center[0] + 15, door_center[1] + 15,
            fill='green')

        # create oval
        oval_center = origin + np.array([UNIT*self.oval_coord[0], UNIT*self.oval_coord[1]])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        self.flag = 1

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        time.sleep(0.01)
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        self.flag = 1
        self.counter = 0
        # return observation
        return np.hstack([(np.array([self.flag])),
                          (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(self.MAZE_H*UNIT),
                          (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.hell1)[:2]))/(self.MAZE_H*UNIT),
                          (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.door)[:2]))/(self.MAZE_H*UNIT),
                          ])

    def step(self, action):
        self.counter += 1
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (self.MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (self.MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if(self.flag == 1 and next_coords == self.canvas.coords(self.oval)):
            reward = 1
            done = False
            self.flag = 0
        elif(self.flag == 0 and next_coords == self.canvas.coords(self.door)):
            reward = 2
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        
        if(self.counter > 50):
            done = True
        s_ = np.hstack([(np.array([self.flag])),
                        (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(self.MAZE_H*UNIT),
                        (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.hell1)[:2]))/(self.MAZE_H*UNIT),
                        (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.door)[:2]))/(self.MAZE_H*UNIT),
                        ])
        return s_, reward, done

    def render(self):
        time.sleep(10)
        self.update()


