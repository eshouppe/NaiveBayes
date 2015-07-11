# NaiveBayes
Naive Bayes classification of undergraduate courses as liberal arts or STEM.

Naive Bayes is a supervised learning algorithm. The algorithm is a trained on one set of data that already has class labels.  It is then tested on unlabeled data.

The Naive Bayes algorithm relies on conditional probability and Bayes' Rule.  Bayesian probability allows the inclusion of prior knowledge when making statements.  For example, given that I know a blackjack player was dealt a ten, what is the probability that the player will get blackjack?  A pseudo representation of the Bayes' classifer is:
posterior probability = conditional probability * prior probability / evidence

As with all probabilistic models, a thorough understanding of the model's assumptions and data requirements is vital.  The Naive Bayes algorithms works with features selected from the data.  It is very important that the features are meaningful in the domain, invariant, and discriminatory between patterns.  Naive Bayes assumes the data features are statistically independent and identically distributed.  A simple example of indpendence is the role of a pair of fair dice.  If one die lands one a five, that outcome has no bearing on the number shown on the second die.  This differs from the earlier card  example, where being dealt a ten of hearts removes that card from the remaining options.

In this project I used a bag of words model to classify text.  My data set is University of Maryland undergraduate course descriptions.  I only include courses that could be considered explicitly STEM or liberal arts.  STEM courses include the expected engineering array and the sciences, and the liberal arts courses include history, english, sociology, etc.

My full data set:
330 Liberal Arts courses with 2090 unique words, which is about 6.3 unique words per course
214 STEM courses with 1982 unique words, which is about 9.3 unique words per course

The program follows these steps:
  1) Load the data and data labels.  Format the remove punctuation and to make all letters lowercase.
  2) Remove stop words, which are words like course or student that violate the model requirements by not discriminating between classes
  3) Create the vocabularly, which is a list of each unique word remaining in the data set
  4) Create a feature vector for each class.  Each vector is of size (1 * number of unique words).  The vector holds the count of words used by training datum in each class.
  5) Calculate relative probabilties using Bayes' rule for each feature vector. 
  6) Classify unlabeled test data using total and individual class probabilities from training data.
  7) Measure the accuracy.

Popular liberal arts words              Popular STEM words
social                                  analysis
political                               systems
american                                design
development                             engineering
study                                   materials
research                                circuits
history                                 basic
cultural                                data
issues                                  techniques
world                                   equations
