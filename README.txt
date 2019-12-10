# 01.112-ML-Project

01.112 Machine Learning Project: Sentiment labelling using Hidden Markov Model

To run evaluation:
$python3 ./EvalScript/evalResult.py ./EN/dev.out ./EN/dev.prediction.out
- change the '/EN/' folder to any of the other 3 given
- change the 'prediction' to either 'p2', 'p3', 'p4' or 'p5'


# Part 2

To run prediction:
$python3 ./part2.py ./EN
- change the './EN' to any of the other 3 given if required

# Part 3

To run Viterbi algorithm:
$python3 ./part3.py ./EN
- change the './EN' to any of the other 3 given if required

# Part 4

To run modified Viterbi algorithm:
$python3 ./part4.py ./EN k

- change the './EN' to any of the other 3 given if required
- change k to wanted kth best sequence

e.g. $python3 ./part4.py ./EN 7


