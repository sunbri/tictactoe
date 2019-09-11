# tictactoe
Contains code for a Tabular Q-Learning agent (method of reinforcement learning) for Tic Tac Toe. In progress are agents that use deep Q-learning, a solution with convolution, and policy gradient descent.

### Random vs. Random
In the most basic case of Tic Tac Toe, we have random agents playing against each other. 
![Image of Random vs. Random](https://github.com/sunbri/tictactoe/blob/master/rvr.png)

We can see that  in the case where random agents are playing against each other, the first player wins approximately 65% of the time. This is expected as playing first would definitely yield an advantage for Tic Tac Toe with random players.

### Q-Learning vs. Random
We hope that the Tabular Q-Learning agent can systematically outperform the random agent for Tic Tac Toe, even if going second. Here are the results: 
![Image of Q vs. Random](https://github.com/sunbri/tictactoe/blob/master/qvr.png)

When the Q-agent goes first, it can usually beat or tie the random agent all of the time after some training. However, we can see the random agent still winning some games. The reason for this anomaly could arise in a simple scenario. Suppose that the Q-agent made a move that is usually very bad (say it gives the random agent two possibilities to win the next move). However, since the random is indeed random, it could miss the guaranteed win, and even worse, the Q-Agent could win the game. In that case, the Q-Agent adjusts its table in such a way that the bad move it played earlier actually is seen in a better light. With this type of learning, we can see it is feasible for the random agent to win.

Here are the results when the Q-Agent goes second: 
![Image of Random vs. Q](https://github.com/sunbri/tictactoe/blob/master/rvq.png)

In this scenario, we can see a similar pattern as above (although it takes more training). The Q-Agent wins or ties most of the games near the end, with the random agent still winning a decent number.

### Q-Learning vs. Q-Learning
The result speaks for itself here:
![Image of Q vs. Q](https://github.com/sunbri/tictactoe/blob/master/qvr.png)

In this case, both of the Q agents learned to play optimally (against each other). It ends with a 100% tie rate, as both players figure out how to avoid a loss. 

### Future Work
In the future, there will be agents with deep Q-learning, a convolutional neural network, and policy gradient descent. Learning to play these types of games is a great exercise in reinforcement learning.
