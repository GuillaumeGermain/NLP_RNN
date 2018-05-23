# NLP RNN
This project is finding out how well RNN networks perform on predicting restaurant reviews.

## Project origin
It's actually the spin-off of an initial [Bag of Words Project](https://github.com/GuillaumeGermain/NLP_bag_of_words)
There, restaurant reviews are pre-preprocessed and grouped into "bag of words" and classified using a battery of standard machine learning algorithms (Logistic Regression, Random Forest, etc.).

The big winner happened to be Logistic regression with a short processing time and a 76.6% accuracy.

Talking of this to a friend, Alain, came quickly the question, how better would RNN methods work in this context?
So here we go. We took a decently big [dataset from Kaggle](https://www.kaggle.com/c/restaurant-reviews/data)
of 82000 restaurant reviews, rated between 1 and 5.
Alain started with a Notebook using Tensorflow, which I integrated and cleaned up. Then we continued with other changes.
After some model changes and model fitting epochs, we have progressively increased the accuracy, or better said, we get closer predictions over the test set.

It's still a work in progress, stay tuned!
