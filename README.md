# Meta Reinforcement Learning

This is a framework for building  a meta reinforcement learning agent on the 
[procgen](https://github.com/openai/procgen) environment from OpenAI.


## Framework layout

```
├───core
│   ├───agents
│   ├───memory
│   │   ├───sampling
│   ├───models
│   ├───preprocessing
│   │   ├───wrappers
│   ├───tools
├───examples
├───references
└───tests
```

## Framework usage

Currently there are three agents implemented, 'default', 'gym', 'hvasslab'. 
The 'default' agent can handle multiple games from the procgen environment, the 
'gym' and 'hvasslab' are build for interfacing with the gym environment. 

The games and instances can be entered as a dictionary directly in the program 
or as a list in the terminal, when no number of instances is given this is 
interpreted as 1.

```cmd
python main.py --agent default --setup coinrun 1 bigfish 1
```
# References

## OpenAI procgen
```
@article{cobbe2019procgen,
  title={Leveraging Procedural Generation to Benchmark Reinforcement Learning},
  author={Cobbe, Karl and Hesse, Christopher and Hilton, Jacob and Schulman, John},
  year={2019}
}
```
