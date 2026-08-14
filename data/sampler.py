import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

class TrainingSampler:
    def __init__(
        self,
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2022, 12, 31, 23, 0, 0),
        max_crop_x_start=500,
        max_crop_y_start=350,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)
        self.time_steps = self.get_time_steps(start_time, end_time)
        self.max_crop_x_start = max_crop_x_start
        self.max_crop_y_start = max_crop_y_start
  
    def get_time_steps(self, start_time, end_time):
        time_steps = []
        curr_time = start_time
        while curr_time <= end_time:
            time_steps.append(curr_time)
            curr_time += relativedelta(hours=1)

        return time_steps

    def __iter__(self):
        return self

    def __next__(self):
        ts_idx = self.rng.choice(len(self.time_steps), size=None, replace=False)
        ts = self.time_steps[ts_idx]

        crop_x_start = self.rng.choice(self.max_crop_x_start, size=None, replace=False)
        crop_y_start = self.rng.choice(self.max_crop_y_start, size=None, replace=False)

        return ts, crop_x_start, crop_y_start
    
if __name__ == '__main__':
    sampler = TrainingSampler()
    for i, sample in enumerate(sampler):
        print(i, sample)

        if i >= 20:
            break
