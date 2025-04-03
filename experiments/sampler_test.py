from src.data_utils.sampling import Sampler

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
sampler = Sampler(X, y)

num_samples = 3
sampled_with_replacement = sampler.sample_with_replacement(num_samples)
sampled_without_replacement = sampler.sample_without_replacement(num_samples)
print("Sampled with replacement:", sampled_with_replacement)
print("Sampled without replacement:", sampled_without_replacement)


num_samples = sampler.calculate_sample_size(0.3)
sampled_with_replacement = sampler.sample_with_replacement(num_samples)
sampled_without_replacement = sampler.sample_without_replacement(num_samples)
print("Sampled with replacement:", sampled_with_replacement)
print("Sampled without replacement:", sampled_without_replacement)
