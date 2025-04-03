# Framework agnostic ensembling for deep learning models

## Design plan

### Ensemble models trained with different batches of one data source (Bagging)

- Create a sampler class to allow sampling with and without replacement [Complete]
- Create a framework to take in a base learner and train it with a sampled data set [Complete]
- Create a user-specificed number of trained base-learners with unique data sets [Complete]
- Aggregate the outputs of the trained base-learners together (mean only) [Complete]
- Add an Evaluation function which allows simpler evaluation on the test set [Complete]
- Expand aggregation options to allow for median, min, max [Complete]

- Expand aggregation options to allow for weighted means

### Iterative multistage combination methods (Boosting)

- Create a framework to take in a base learner and train it with a full data set

#### AdaBoost method

- Create a method for identifying the misclassified examples
- Allow extra parameterisation over misclassified examples e.g. thresholding
- Implement a method for weighting misclassified samples higher in next iteration
- Create a method for training the second base learner with these weighted samples

#### XGBoost method

- Identify residual errors
- Train next model on residual errors

### Ensemble different model architectures (Voting)

- Integrate combining PyTorch models with the standard Keras MLP
- Integrate classical ML models such as XGB