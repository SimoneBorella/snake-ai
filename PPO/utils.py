import matplotlib.pyplot as plt
from IPython import display

plt.ion()
plt.draw()
plt.pause(1)

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title("Snake - PPO training")
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean score")
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.pause(0.1)