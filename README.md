# Ranger-Deep-Learning-Optimizer
Ranger - a synergistic optimizer combining RAdam (Rectified Adam) and LookAhead in one codebase.
Latest version 9.3.19 - full refactoring for slow weights and one pass handling (vs two before).  Refactor should eliminate any random save/load issues regarding memory.   

Medium article with more info:  
https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

Multiple updates:
1 - Ranger is the optimizer we used to beat the high scores for 12 different categories on the FastAI leaderboards!  (Previous records all held with AdamW optimizer).

2 - Highly recommend combining Ranger with: Mish activation function, and flat+ cosine anneal training curve.

3 - Based on that, also found .95 is better than .90 for beta1 (momentum) param (ala betas=(0.95, 0.999)).

Fixes:
1 - Differential Group learning rates now supported.  This was fix in RAdam and ported here thanks to @sholderbach.
2 - save and then load may leave first run weights stranded in memory, slowing down future runs = fixed.


Usage and notebook to test are available here:
https://github.com/lessw2020/Ranger-Mish-ImageWoof-5



