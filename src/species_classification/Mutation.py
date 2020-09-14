import numpy as np


class Mutation(object):
    def __init__(self, nmute):
        self.nmute=nmute

    def __call__(self, sample):
        data, labels = sample['data'], sample['labels']

        to_mutate=np.random.choice(len(data),size=(self.nmute),replace=False)
        mutation=np.random.randint(4,size=(self.nmute))
        sample['data'][to_mutate]=mutation
        return sample
