
# Change log

2020-02-26:
Added easier loading and saving of older models


2020-02-11:
    - Fixed memory bug, where the next state was not correctly used (took next, next state)
    - Fixed issue where there was no learning result. By increasing the learning rate.

| Method | LR   |
|--------|------|
| Batch  | 1e-2 |
| Single | 1e-3 |

2020-02-10:
    - CartPole was showing no learning with update formula:
```python
target = rewards[k] + self.alpha * (self.gamma * np.amax(q_values_next[k]))
``` 

