import numpy as np
import pandas as pd

class QLTable:
    def __init__(self, actions, learning_rate=0.2, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)
    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state]
            # state_action = state_action.reindex(np.random.permutation(state_action))
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
    def reset(self,e):
        self.lr = self.lr * (1-e/50)
        self.epsilon += 0.1 * e / 45
    def learning(self, state, action, reward, state_):
        print(self.lr)
        self.check_state_exist(state_)
        q_predict = self.q_table.loc[state,action]
        if state_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_,:].max()
        else:
            q_target = reward
        # update q_table
        self.q_table.loc[state,action] += self.lr*(q_target-q_predict)
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat([self.q_table, pd.DataFrame([[0]*len(self.actions)],columns = self.q_table.columns,index = [state])])
