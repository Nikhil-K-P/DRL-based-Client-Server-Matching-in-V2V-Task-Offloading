import numpy as np
from ddpg_torch import Agent
from environment import Environment
from utils import plot_learning_curve

if __name__ == '__main__':
    env = Environment(20)
    agent = Agent(alpha=0.0001, beta=0.001, 
                  input_dims=env.state_space_shape, tau=0.001,
                  batch_size=64, fc1_dims=200, fc2_dims=100, 
                  n_actions=1)
    n_games = 250
    filename = 'V2V_Client_Server_Matching' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + 'episodes'
    figure_file = 'plots/' + filename + '.png'

    best_score = 1
    score_history = []
    avg_scores = []  # To store average scores from the beginning

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action[0])
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)

        # Calculate the average score from the beginning
        avg_score = np.mean(score_history)
        avg_scores.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, avg_scores, figure_file)
