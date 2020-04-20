


# 2020-02-22

Path to windows install for the Box2D env and MuJuCo.
- [Windows install](https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5)



# 2020-02-22

Changed min epsilon to 0.001 for more consistent performance.



# 2020-02-21

## NN Architecture
two FC layers = [150, 120]
activations = LeakyReLU(alpha=0.1)

## Hyperparameters
learning rate = 0.001\
optimizer = Adam\
gamma = 0.99\
alpha = 0.1\
batch_size = 64\
kernel_init = He Normal\
replay memory size = 1e6

### Performance
Model converges to an average score of approx. 150 after 200 episodes. 
Once in 50 episode it still gets a negative score so the performance is
still a bit unstable.
