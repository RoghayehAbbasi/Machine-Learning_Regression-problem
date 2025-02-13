# Machine-Learning_Regression-problem
the task is to predict a momentary self-reported well being score that was measured while people were playing a video game designed to lower stress and improve mental health.
Data files

# The dataset contains the following files:
train_data.csv
test_data.csv
supplimental_data.csv

# There are a number of predictor variables that appear in these datasets:
UserID: a unique ID for each user. Some IDs only occur in the training dataset, some occur only in the test dataset, and most appear in both datasets. For the IDs that that appear in both datasets, the entries in the test dataset all come after (in TimeUtc) the entries in the training dataset.
QuestionTiming: binary variable indicating if the question was prompted by the system or the user initiated the question-answering.
TimeUtc: time of the response
CurrentGameMode: the current mode the user selected to play
CurrentTask: the current task/goal in the game the user was working to complete
CurrentSessionLength: the amount of time the user has been logged into the game during this session
LastTaskCompleted: the previous task/goal in the game that the user completed
LevelProgressionAmount: how far (measured as a proportion) the user has progressed to the next level
QuestionType: what aspect of mental health is being asked about. In the training and test datasets, these are all questions about "Wellbeing". The supplemental dataset contains answers to other question types from the same set of users.
ResponseValue: the score the user gave to the current question. This column is missing in the test dataset.

# Evaluation metric
The evaluation metric for this task is Mean Absolute Error. More details about how to submit your answers to the competition server will be posted later.
