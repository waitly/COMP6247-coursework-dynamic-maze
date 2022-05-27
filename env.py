from matplotlib import pyplot as plt

import read_maze
import numpy as np
import sys

class Maze_env:
    def __init__(self, maze):
        x = maze.shape[0]
        y = maze.shape[1]
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.init_position = [1,1]
        self.current_position = [1,1]
        self.goal = [199,199]
        self.maze = maze
        self.around = 0
        self.distance = (self.goal[0] - self.current_position[0]) + (self.goal[1] - self.current_position[1])
        self.trajectory = []
        self.mask = np.zeros((201, 201),dtype=int) # The set of path information (where is wall or empty) in a episode, to draw the maze image.
        self.color = [(255,255,255),(0,0,0)]
    # Get the surrounding information
    def getObservation(self,x,y):
        self.around = read_maze.get_local_maze_information(x,y)
        return self.around
    # This function will give a set of feedback for the action, the agent choose in current position
    def step(self,action):

        if action == 0:   # up
            move = [-1,0]
        elif action == 1:   # down
            move = [1,0]
        elif action == 2:   # left
            move = [0,-1]
        elif action == 3:   # right
            move = [0,1]
        # Get the new position
        new_position = [self.current_position[0] + move[0],self.current_position[1] + move[1]]

        # reward function

        if new_position == self.goal: # If the new position is the goal
            reward = 1
            done = True
            s_ = 'terminal'
        elif not self.isWall(new_position):
            # Manhattan Distance between goal and new position
            new_distance = (self.goal[0] - new_position[0]) + (self.goal[1] - new_position[1])
            if self.isFire(new_position):
                reward = 0
                s_ = 'fire'
            else:
                if self.distance > new_distance:
                    reward = 0.15 + (198*2-self.distance)*0.0006
                else:
                    reward = -0.05
                self.distance = new_distance
                # revisited
                if new_position in self.trajectory:
                    reward = -0.2
                # update position
                self.current_position = new_position
                s_ = new_position
            done = False
        else:
            # If the new position is wall,give -1 as penalty
            reward = -1
            done = False
            s_ = self.current_position

        return s_, reward, done
    # Add the new state to the trajectory.
    def update_trajectory(self,state):
        self.trajectory.append(state)
    # Judge whether the new position is wall or not
    def isWall(self,new_position):
        around = self.around
        x = new_position[0] - self.current_position[0] + 1
        y = new_position[1] - self.current_position[1] + 1
        if around[x][y][0]==1:
            return False
        else:
            return True
    # Judge whether the new position have fire or not.
    def isFire(self,new_position):
        around = self.around
        x = new_position[0] - self.current_position[0] + 1
        y = new_position[1] - self.current_position[1] + 1
        if around[x][y][1] == 0:
            return False
        elif around[x][y][1] > 0:
            return True
    # Reset the basic setting of the environment.
    def reset(self):
        self.current_position = [1, 1]
        self.mask = np.zeros((201, 201), dtype=int)
        self.trajectory = []
        read_maze.load_maze()
        self.maze = read_maze.maze_cells
        return self.current_position
    # Update the mask region
    def updateMask(self,current_position,around):
        x = current_position[0]
        y = current_position[1]
        for i in range(3):
            for j in range(3):
                self.mask[x+i-1][y+j-1] = around[i][j][0]

    # Draw the path taken by the agent in this episode
    def draw(self, filename):
        maze_img = np.empty([self.maze.shape[1], self.maze.shape[0], 3], dtype=np.uint8)
        for i in range(self.mask.shape[1]):
            for j in range(self.mask.shape[0]):
                if self.mask[i][j] == 1:
                    maze_img[i][j] = self.color[0]
                else:
                    maze_img[i][j] = self.color[1]

        plt.figure(figsize=(16,16))
        im = plt.imshow(maze_img, interpolation='none', aspect='equal', cmap='Greys')
        ax = plt.gca()

        plt.xticks([], [])
        plt.yticks([], [])

        ax.plot(self.goal[1], self.goal[0],
                'bs', markersize = 4)
        ax.plot(self.init_position[1], self.init_position[0],
                'rs', markersize = 4)
        for i in range(len(self.trajectory)):
            ax.plot(self.trajectory[i][1], self.trajectory[i][0],
                    'ys', markersize=3)

        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.show()
