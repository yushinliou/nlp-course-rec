# Recommendation system for Hahow

+ It is a implement of course recommendation system. Our model is base on behaviour sequence transformers model.
    + our target is to find a function S to predict the last position of a user's behaviour sequence, which is the possibility of u's next behaviour is buy the traget item v.
    + $S(u)=\{v_1,v_2,v_3,...,v_n\}$
    + reference:Chen, Qiwei, et al. "Behavior sequence transformer for e-commerce recommendation in alibaba." Proceedings of the 1st international workshop on deep learning practice for high-dimensional sparse data. 2019.
    + [reference code](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/movielens_recommendations_transformers.ipynb)
+ Training data is from a Taiwan online course website "hahow" consist of three part:
    + user context: user_id, interest, gender, occupation
    + course context: course_id, course name, genre
    + rating data: user_id, course_id, rating, the time user buy the course

### Behavior Sequence Transformer (BST)
+ **Data Preprocessing**
    + **Raw Data**
        + **User Profile:** User's age, interests, gender.
        + **Item:** Product information, e.g., name.
        + **Rating:** Ratings and purchase times for specific products. Absence of course purchase times leads to using the original data order, assuming equidistant intervals between course purchases. Courses purchased by a user are rated 1.
    + Majority of users have only one course purchase. The model requires a sequence of at least two: the first for past behavior, the second for future behavior.
    + A fictional course named "hahow Online Course" in "Education" category is assumed purchased by all, given user interest in hahow content.
    + **Training Data Format:** Fields connected by `|`:
        1. **User ID:** Original string IDs re-encoded to integer.
        2. **User Course Purchase Record:** Length of 2.
        3. **Rating:** Set to 1 for both courses in the sequence.
        4. **Occupation:** Missing data coded as "other".
        5. **Gender:** Missing data coded as "X".
        6. **Interests:** Separated by `_`.

+ **Model Architecture**
    + Includes embedding layer, transformer, MLP layers. Loss function is mean-square error: \(\frac{1}{n}\sum_{i=1}^{N}{(y-\hat{y})^2}\), where \(y=1\), \(\hat{y}\) is predicted probability of 1, \(n\) is number of training data.
    + **Output:** Probability of user purchasing a target item.
        + **Seen Course:** Predicts probability of purchasing each of 728 courses, ranking them by probability.
        + **Seen Subgroup:** Predicted courses converted to corresponding subgroup.

+ **Experiment**
    + Initial use of sample code's predicted parameters, with subsequent adjustments:
        + **Batch Size:** 256
        + **Learning Rate:** 0.01
        + **Dropout Rate:** 0.1
        + **Number of Epochs:** 10
    + Eval
      + ![Orignial parameter](https://i.imgur.com/DzOD6X2.png)
      + ![learning rate = 3e-5](https://i.imgur.com/mV7Ij6N.png)
      + ![batch size = 128](https://i.imgur.com/l1rhc1j.png)
      + ![dropout rate = 3e-5](https://i.imgur.com/xcjHsYn.png)


+ **Performance**
    + Best performance with original sample code parameters.
        + **Seen Course:** 0.08492
        + **Seen Subgroup:** 0.1984

+ **Discussion**
    + **Limitations**
        + **Loss/Evaluation Design:** Training data includes only ratings of 1, biasing the model towards predicting target course purchases. Not purchasing does not necessarily indicate dislike.
        + **Missing Data:** User purchase behavior could be cross-platform, but only hahow information is available, potentially leading to incomplete behavioral sequences.
        + **Short Sequences:** Most users' purchase history is too brief, necessitating a sequence length of only 2.
        + **Applicability:** Limited to users with usage records, not suitable for new user course recommendations based only on user profile, applicable only to seen courses.

# reproduce
```shell
# download raw data (hahow user data)
bash download.sh
# train model and do the predict, it takes about 5 min in colab environment(with gpu)
bash run.sh path/to/pred.csv
```

