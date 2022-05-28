# COMP6247-coursework-dynamic-maze
This project comes from one part of coursework in COMP6247 - Reinforcement and Online Learning.
The purpose of this part is to adopt a reinforcement learning (RL) technique to solving a dynamic maze problem to compute the minimum time path from the top left corner to the bottom right corner of the maze.

The maze's size is 200x200, and the only way the agent is able to get the maze information is surrounding observation. Every time the agent observe the maze will generate fire in surround randomly.


# Compile and run
Just download all .py files, then run the test.py.

# Results
The train time depends on the agent, sometimes it need long time because the agent may go in the wrong way deeply and in maze. But after around 15-20 episode, the model would be trained very well and alway go along the optimal route.

# Traversal cost time in each episode

![WechatIMG351](https://user-images.githubusercontent.com/48167274/170840057-37056cee-00a9-4831-950f-457377e55a99.jpeg)

Sometimes it will be like:
![WechatIMG352](https://user-images.githubusercontent.com/48167274/170840141-7a107be4-379d-4831-bc8c-1872530cb18a.jpeg)

# The optimal path:
![m49](https://user-images.githubusercontent.com/48167274/170840013-bcae8282-6570-468e-b276-09424a60a04a.png)
