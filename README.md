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

### Ensemble different model architectures (Voting)

- Integrate combining PyTorch models with the standard Keras MLP
- Integrate classical ML models such as XGB