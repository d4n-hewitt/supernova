from src.data_utils.sampling import Sampler

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sampler = Sampler(data)

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
