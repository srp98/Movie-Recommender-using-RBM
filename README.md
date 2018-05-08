# Movie-Recommender-using-RBM
A Movie Recommender System using Restricted Boltzmann Machine (RBM) approach used is collaborative filtering. This system is an algorithm that recommends items by trying to find users that are similar to each other based on their item ratings.

RBM is a Generative model with two layers(Visible and Hidden) that assigns a probability to each possible *binary state vectors* over its visible units. Visible layer nodes have *visible bias(vb)* and Hideen layer nodes have *hidden bias(hb)*. A weight matrix of row length equal to input nodes and column length equal to output nodes.

## Requirements
- Python 3.6 and above
- Tensorflow 1.6.0 and above
- NumPy
- Pandas
- Matplotlib

## Dataset
The dataset used is MovieLens 1M Dataset acquired by [Grouplens](https://grouplens.org/datasets/movielens/) contains movies, users and movie ratings by these users.

## Model Description
Our model works in the following manner :- 
- The hidden layer is used to learn features from the information fed through the input layer.
- The input is going to contain X neurons, where X is the amount of movies in our dataset.
- Each of these neurons will possess a normalized rating value varying from 0 to 1: 0 meaning that a user has not watched that movie and the closer the value is to 1, the more the user likes the movie that neuron's representing.
- These normalized values will be extracted and normalized from the ratings dataset.
- After passing in the input, we train the RBM on it and have the hidden layer learn its features.
- These features are used to reconstruct the input, which will predict the ratings for movies that the input hasn't watched, which is what we can use to recommend movies!

## References
- Inspired from the idea presented in paper [Salakhutdinov, R., Mnih, A., & Hinton, G. (2007, June). Restricted Boltzmann machines for collaborative filtering](http://www.cs.utoronto.ca/~hinton/absps/netflixICML.pdf)
- Additional Notes used for understanding
  - [Extra Notes - 1](http://swoh.web.engr.illinois.edu/courses/IE598/handout/fall2016_slide2.pdf)
  - [Extra Notes - 2](https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf)
