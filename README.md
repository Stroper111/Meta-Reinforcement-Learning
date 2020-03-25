
# AI Framework

Our (horrible) attempt of creating AI, it supports the following packages:
- OpenAI: gym
- OpenAI: procgen


## Install gym on windows

To test if you have atari gym installed follow the following procedure in Pycharms 'Python console', which is located at the bottom of Pycharm.

If you do not have Pycharm go to your venv and run ```python```, followed by these commands:

```` 
import gym
env = gym.make("Pong-v0')
````

If this gave no errors you are done and can skip the remaining part.

If you are using Ubuntu/Linux you can use the following command and skip the remaining of this explanation (do this in the terminal). 

```text
$ pip install gym[atari]
```

For Windows users, this won't work most of the time due to the _cmake_ package not being available for windows user. So for this we need to install [Visual Studio](https://visualstudio.microsoft.com/downloads/).

You can simply install the community version for free. When continuing the install make sure to select C++ Build tools for Windows development.  In case you missed it you can install [vs buildtools.exe](https://aka.ms/vs/16/release/vs_buildtools.exe) and use that to install it. Note: this will take about 5GB, being a data scientist this is still a small investment.

Now use the following commands to install all environments of gym: (don't forget to check that you are on your venv.)
```text
pip install gym
pip install git+https://github.com/Kojoley/atari-py.git
pip install gym[atari]
```


## PyTorch
- [Download link](https://pytorch.org/get-started/locally/)

## Tensorflow
- Upgrade setuptools (`pip install setuptools --upgrade`)
- [Download link](https://www.tensorflow.org/install/gpu)
