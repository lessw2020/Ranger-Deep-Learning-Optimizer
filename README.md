# Ranger-Deep-Learning-Optimizer
Ranger - a synergistic optimizer combining RAdam (Rectified Adam) and LookAhead in one codebase.
Medium article with more info:  https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

Multiple updates:
1 - We used Ranger to beat the FastAI leaderboard score by nearly 20% (19.77%).  The trick was to combine Ranger with: Mish activation function, and flat+ cosine anneal training curve.
2 - Based on that, also found .95 is better than .90 for beta1 (momentum) param (ala betas=(0.95, 0.999)).
3 - Verified no load/save issues in our codebase here.  It was an issue for people that were using LookAhead/RAdam as seperate components.


Usage and notebook to test are available here:
https://github.com/lessw2020/Ranger-Mish-ImageWoof-5



