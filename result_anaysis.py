def get_method(url):
	return url.split('/')[-1].split('.')[0].split('_')[1]

option2int = {'optionA':0, 'optionB':1, 'optionC':2}

results = {}
with open('Label.csv') as f:
	for line in f:
		_, a, b, c, choice = line.strip().split(',')

		idx = int(a.split('/')[-1].split('.')[0].split('_')[0])
		methods = [get_method(a), get_method(b), get_method(c)]
		if idx not in results:
			results[idx] = []
		results[idx].append((set(methods), methods[option2int[choice]]))

from itertools import combinations
methods = ['poisson-gan-encoder', 'poisson-gan-wgan', 'multi-splines', 'poisson', 'modified-poisson']
method_combination = list(combinations(methods, 2))
method_combination = [set(m) for m in method_combination]

cleaned_results = {}
for idx, record in results.items():
	scores = {key:0 for key in methods}

	for cb in method_combination:
		method_a, method_b = cb
		a, b = 0, 0
		for hit in record:
			if cb.issubset(hit[0]):
				if hit[1] == method_a:
					a += 1
				if hit[1] == method_b:
					b += 1
		
		if a > b:
			scores[method_a] += 1
		if b > a:
			scores[method_b] += 1

	cleaned_results[idx] = scores

import numpy as np
score_each_method = {key:[] for key in methods}
for scores in cleaned_results.values():
	for key in methods:
		score_each_method[key].append(scores[key])
for key in methods:
	print(key, np.mean(score_each_method[key]), np.std(score_each_method[key]))

final_scores = {key:0 for key in methods}
for scores in cleaned_results.values():
	for key in methods:
		final_scores[key] += scores[key]
print(final_scores)