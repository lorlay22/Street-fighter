
# AI Learns to Play Street Fighter II with Reinforcement Learning

This project documents the process of training a Reinforcement Learning (RL) agent to play the classic SEGA Genesis game, **Street Fighter II: Special Champion Edition**. The agent learns from raw pixel data, using the Proximal Policy Optimization (PPO) algorithm to maximize its in-game score.

This work is based on the tutorial by [Nicholas Renotte](https://www.youtube.com/c/NicholasRenotte).
**▶️ [Watch the original tutorial on YouTube](https://www.youtube.com/watch?v=rzbFhu6So5U)**

## Table of Contents

1.  [Project Pipeline](https://www.google.com/search?q=%23project-pipeline)
2.  [Technologies Used](https://www.google.com/search?q=%23technologies-used)
3.  [Setup & Installation](https://www.google.com/search?q=%23setup--installation)
4.  [Preprocessing Logic](https://www.google.com/search?q=%23preprocessing-logic)
5.  [Hyperparameter Tuning with Optuna](https://www.google.com/search?q=%23hyperparameter-tuning-with-optuna)
6.  [Training and Testing the Model](https://www.google.com/search?q=%23training-and-testing-the-model)

## Project Pipeline

The project follows these key steps to train a competent AI agent:

1.  **Environment Setup**: Configure a Python environment with OpenAI Gym and Gym Retro to run the Street Fighter II ROM.
2.  **Preprocessing**: Create a custom Gym environment to process game frames (grayscale, resize, frame differencing) and shape the reward signal.
3.  **Hyperparameter Tuning**: Use the Optuna framework to automatically find the best hyperparameters for the PPO model.
4.  **Model Training**: Train the PPO agent using the optimized hyperparameters with Stable Baselines3.
5.  **Evaluation & Testing**: Run the trained model on the game to evaluate its performance.

## Technologies Used

  - **Python 3.7.3**
  - **Jupyter Notebook**
  - **OpenAI Gym**: The core toolkit for developing and comparing reinforcement learning algorithms.
  - **Gym Retro**: An extension of Gym for running classic video game ROMs.
  - **Stable Baselines3**: A set of reliable implementations of reinforcement learning algorithms in PyTorch.
  - **Optuna**: A hyperparameter optimization framework to automate the search for the best model settings.
  - **OpenCV-Python**: For real-time image processing.
  - **NumPy**: For numerical operations and array manipulation.

## Setup & Installation

It is **highly recommended** to use a dedicated virtual environment to avoid dependency conflicts.

### 1\. Create a Conda Environment

Open an Anaconda Prompt and create a new environment with a specific Python version:

```bash
conda create -n gym_env python=3.7.3 jupyter notebook
```

Activate the environment:

```bash
conda activate gym_env
```

### 2\. Install Dependencies

Create a file named `requirements.txt` in your project folder and add the following lines:

```
gym==0.21.0
gym-retro
opencv-python
stable-baselines3[extra]
optuna
pyglet==1.5.27
importlib-metadata==4.13.0
```

Then, install all libraries with a single command from your activated `gym_env` terminal:

```bash
pip install -r requirements.txt
```

### 3\. Download the Game ROM

You will need the ROM for *Street Fighter II: Special Champion Edition*.

  - **Download Link**: [WowRoms - Street Fighter II](https://wowroms.com/en/roms/sega-genesis-megadrive/street-fighter-ii-special-champion-edition-europe/26496.html)
  - After downloading, unzip the file and place the ROM file (e.g., `Street Fighter II' - Special Champion Edition (E).md`) in a dedicated `roms` folder within your project.

### 4\. Import the ROM into Gym Retro

This step makes the game available to your Python environment.

  - Open the Anaconda Prompt and activate your environment (`conda activate gym_env`).
  - Navigate to the folder where you placed the ROM file.

<!-- end list -->

```bash
# Example:
cd path/to/your/project/roms
```

  - Run the import command:

<!-- end list -->

```bash
python -m retro.import .
```

You should see a confirmation message:

```
Importing StreetFighterIISpecialChampionEdition-Genesis
Imported 1 games
```

***Troubleshooting Note:*** *If the import fails with one version of the ROM (e.g., the Europe version), try a different one (e.g., the USA version). Sometimes the importer can be sensitive.*

## Preprocessing Logic

To make learning more efficient, a custom Gym environment wrapper (`StreetFighter(Env)`) was created. It handles all preprocessing tasks automatically.

#### `__init__(self)` (Constructor)

  - **Observation Space**: Defines what the AI "sees." This is set to a `Box` of `(84, 84, 1)`, meaning a single-channel (grayscale) 84x84 pixel image. Pixel values range from 0 to 255.
  - **Action Space**: Defines what the AI "can do." This is a `MultiBinary(12)` space, representing 12 buttons that can be either pressed (1) or not pressed (0) at each step. `use_restricted_actions=retro.Actions.FILTERED` simplifies the possible button combinations.

#### `preprocess(self, observation)`

This function takes a raw game frame and applies several transformations:

1.  **Grayscaling**: Converts the color image to grayscale using `cv2.cvtColor`. This reduces the complexity from 3 color channels to 1.
2.  **Resizing**: Downscales the image to 84x84 pixels using `cv2.resize`. This significantly reduces the number of pixels the model needs to process, speeding up training.
3.  **Reshaping**: Adds a channel dimension to the 2D image array, making its shape `(84, 84, 1)`, which is the standard input format for Convolutional Neural Networks (CNNs).

#### `step(self, action)`

This is the core of the agent-environment interaction.

1.  The agent's chosen `action` is passed to the game.
2.  The game returns the `next observation`, a `reward`, a `done` signal (if the game is over), and an `info` dictionary.
3.  **Frame Differencing**: The new observation is preprocessed, and the difference between it and the previous frame is calculated (`frame_delta = obs - self.previous_frame`). This helps the AI perceive motion.
4.  **Reward Shaping**: The reward is not taken directly from the game. Instead, it's calculated as the **change in score** (`reward = info['score'] - self.score`). This encourages the agent to perform actions that increase its score.

## Hyperparameter Tuning with Optuna

To find the best settings for the PPO model, we use **Optuna** to automate hyperparameter optimization.

  - **`optimize_ppo(trial)`**: This function defines the search space for key hyperparameters like `n_steps`, `gamma`, `learning_rate`, `clip_range`, and `gae_lambda`. For each trial, Optuna suggests a new combination of values from these ranges.
  - **`optimize_agent(trial)`**: This is the objective function for Optuna. For each set of hyperparameters suggested by `trial`:
    1.  It creates a fresh, wrapped Street Fighter environment.
    2.  It initializes a PPO model with the suggested parameters.
    3.  It trains the model for a fixed number of timesteps (e.g., 75,000).
    4.  It evaluates the trained model over 10 episodes and returns the `mean_reward`.
    5.  Optuna uses this returned score to guide its search for the best-performing combination.

## Training and Testing the Model

  - **Training**: The model is trained using the best hyperparameters found during the Optuna study. A custom callback is used to save the model periodically.
  - **Testing**: After training, the final model can be loaded and run in render mode to visually confirm its performance and watch it play the game.
