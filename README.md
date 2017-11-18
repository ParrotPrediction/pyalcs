# Parrot Prediction OpenAI environments

## Maze

Initializing

    maze = gym.make('MazeF1-v0')

Getting all possible transitions

    transitions = maze.env.get_all_possible_transitions()

## Boolean Multiplexer
Read blog [post](https://medium.com/parrot-prediction/boolean-multiplexer-in-practice-94e3236821b5) describing the usage.