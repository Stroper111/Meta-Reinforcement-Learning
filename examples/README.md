# Examples

In this folder there are several examples on how to use gym and procgen in
general and how you can use this Framework to run both environments
directly without having to change your main loop.

The following examples are provided for the gym and procgen environment:

- [Basic Gym commands](./base_gym.py)
- [Basic Gym run loop](./base_gym_loop.py)
- [Basic Procgen loop](./base_procgen.py)

Now Procgen already allows for multiple games, while Gym doesn't by default. 
The MultiEnv provides a wrapper that enables multiple gym environments at the
same time, similar to Procgen. A demonstration can be found in the following file:

- [Basic MultiEnv](./multi_env_base.py)

Now that the basic options are known, there are also some more advanced features 
to help training for Meta Reinforcement Learning. This means that it is possible
to play multiple different games at the same time, using either the same process
as in Procgen (Multiprocessing) or on a single core. These examples are shown in

- [Extra MultiEnv](./multi_env_extra.py) 
