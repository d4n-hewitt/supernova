# Framework agnostic ensembling for deep learning models

## Design plan

### Ensemble models trained with different batches of one data source (Bagging) :white_check_mark:

- Create a sampler class to allow sampling with and without replacement [Complete]
- Create a framework to take in a base learner and train it with a sampled data set [Complete]
- Create a user-specificed number of trained base-learners with unique data sets [Complete]
- Aggregate the outputs of the trained base-learners together (mean only) [Complete]
- Add an Evaluation function which allows simpler evaluation on the test set [Complete]
- Expand aggregation options to allow for median, min, max [Complete]
- Expand aggregation options to allow for weighted means [Complete]

### Iterative multistage combination methods (Boosting) :white_check_mark:

- Create a framework to take in a base learner and train it with a full data set [Complete]

#### DanBoost method :construction:

- Create a method for calculating the error (abs diff between pred and label) [Complete]
- Allow extra parameterisation over error calculation [Complete]
- Implement a method for weighting samples in next iteration using error of previous iteration (feature level weights) [Complete]
- Create a method for training the second base learner with all samples weighted by these errors [Complete]
- Add functionality to allow errors from previous iterations to be included [Complete]
- Add functionality to weight previous models errors in combination for current iteration training, so sample weights are calculated by weighted errors of all previous model iterations [Complete]
- Allow different basic averaging schemes for model ensemble other than mean
- Add a weighted averaging scheme for model ensembling to give more priority to the first model that was trained
- Add early stopping
- Residual based regression


### AdaBoost method :x:

- Similar to DanBoost but use AdaBoost weighting methods

#### XGBoost method :x:

- Identify residual errors
- Train next model on residual errors

### Ensemble different model architectures (Voting) :x:

- Integrate combining PyTorch models with the standard Keras MLP
- Integrate classical ML models such as XGB