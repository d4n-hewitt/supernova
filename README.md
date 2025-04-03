# Framework agnostic ensembling for deep learning models

## Design plan

### Ensemble models trained with different batches of one data source

- Create a sampler class to allow sampling with and without replacement [Complete]
- Create a framework to take in a base learner and train it with a sampled data set
- Create a user-specificed number of trained base-learners with unique data sets
- Aggregate the outputs of the trained base-learners together (mean only)

- Add an Evaluation function which allows simpler evaluation on the test set

- Expand aggregation options to allow for median, min, max
- Expand aggregation options to allow for weighted means