import numpy as np
from matplotlib import pyplot as plt
from RL_brain import QLTable
from env import Maze_env
import read_maze

def update():
    time_list = []
    for episodes in range(50):
        # Reset the environment and q_table instance
        state = env.reset()
        RL.reset(episodes)
        time = 0
        path_actions = []
        path_observations = []
        while True:
            # print(state)
            # Add the current position to trajectory
            env.update_trajectory(state)
            # Observe surroundings
            around = env.getObservation(state[0],state[1])
            # current observation to list
            path_observations.append(around)
            # RL choose action based on current state
            action = RL.choose_action(str(state))
            # Update the mask
            env.updateMask(state,around)
            # Add the action to list
            path_actions.append(action)
            # print("path",action)
            time += 1
            # Rl take action and get new state and reward
            state_, reward, done = env.step(action)
            # if the new state is fire, then keep the state and wait
            if state_ == 'fire':
                state = state
            else:
                # RL learn from this transition
                RL.learning(str(state),action,reward,str(state_))
                # update state
                state = state_
            # print(RL.q_table)
            # print(time)
            
            # break while loop when end of this episode
            if done:
                break
        env.draw(f'm{episodes}')
        time_list.append(time)
        print("cost:",time)
    # end of this game
    print('game over')
    print(time_list)
    # Plot cost time for each episode.
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.title("The cost time in each episode")
    plt.xlabel("Episodes")
    plt.plot(list(range(50)), time_list, '-')
    plt.show()
    # output the trace
    resultOutput(path_observations,path_actions)

def resultOutput(observations, actions):

    for i in range(len(actions)):
        if actions[i] == 0:
            actions[i] = 'up'
        elif actions[i] == 1:
            actions[i] = 'down'
        elif actions[i] == 2:
            actions[i] = 'left'
        elif actions[i] == 3:
            actions[i] = 'right'
    # write observation, action  with timing information into txt document
    file = open('output.txt', 'w')
    for i in range(len(actions)):
        file.write(("Time:" + str(i)) + '\n')
        file.write(("Position:" + '('+ str(env.trajectory[i][0])+','+ str(env.trajectory[i][1]) + ')') + '\n')
        file.write(("Action:" + actions[i]) + '\n')
        file.write(("Observation:") + '\n')
        for j in range(3):
            for k in range(3):
                if observations[i][j][k][0]==1:
                    if observations[i][j][k][1] == 0:
                        file.write(("empty") + ' ')
                    else:
                        file.write(("fire") + ' ')
                else:
                    file.write(("wall") + ' ')
            file.write('\n')
    file.close()
    # print(env.trajectory)

if __name__ == '__main__':
    # Load the maze
    read_maze.load_maze()
    # Create the environment instance
    env = Maze_env(read_maze.maze_cells)
    # Create the Q-table for the algorithm
    RL = QLTable(list(range(env.n_actions)))
    update()
