# RL QWOP

Forked from [here](https://github.com/juanto121/qwop-ai).

A friend of mine is really good at the video game QWOP. 
It's a really hard game where you control a sprinters legs 
to run a 100 meter dash: you can [try it out yourself](http://www.foddy.net/Athletics.html). 
I'm not very good at the game, but I wanted to beat my friend, so I trained a computer to beat the game for me.


## QWOP Environment

I modified the internals of QWOP so that it can be run as a gym environment. Agents can send key commands to the game and 
observe the state of the game, how far the runner has run, and if the game has been completed.
The game is hosted in a local node server. To get it running, run the following:

```
cd game
npm i
node server
```

The `./gym-qwop` folder has python classes modeling QWOP as an [open-ai gym](https://gym.openai.com/) environment.
There are three versions of the environment: `qwop-v0`, `frame-qwop-v0`, and `multi-frame-qwop-v0`.
Run `pip install ./gym-qwop/` to install the gym environment.

The default environment (`qwop-v0`) returns observations representing the position and angle of each of the runners limbs.
`frame-qwop-v0` returns observations as the pixel data of the current frame of the game. `multi-frame-qwop-v0` also the previous three frames as observations.

Since we have modeled the environment as a gym environment, we can use openai's implementation of many popular RL algorithms.

Running `python run-gym.py` will train a model using one of openai's algorithms.


## PPO

The `./RLQWOP` directory contains a customized implementation of the proximal policy optimization algorithm. My implementation borrows much from [openai's spinning up implementation of PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html#) with a modified strategy to paralellize training.

In my implementation, several actor processes maintain their own copy of the model and gather experiences from the environment. Each actor adds their experiences to a shared replay buffer. A single learner process does gradient updates from the experiences stored in the replay buffer. After updating the model, the learner distributes the model's parameters back to the actor processes.