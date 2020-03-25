

# Pong Policy Gradient


##  Possible improvements

- We are now batching based on episodes, but using mp is not training equally
a better approach would be to update every x-frames. (~10 x 1000 frames = 10.000 frames)

- The image preprocessing is not really the best, maybe gray conversion with a 
cutoff would be better and converge sooner.

- It does seem that using multiple environments is reducing the variability
in the training process. This needs to be verified!