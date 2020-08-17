import torch
from torch.utils.data import sampler

class RandomEpochSampler(sampler.RandomSampler):
  def __init__(self, data_source, replacement=False, num_samples=None, epochs=1):
    self.epochs = epochs
    super(RandomEpochSampler, self).__init__(data_source, replacement, num_samples)

  @property
  def num_samples(self):
    # dataset size might change at runtime
    if self._num_samples is None:
      return len(self.data_source) * self.epochs
    return self._num_samples * self.epochs

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n = len(self.data_source)
    for e in range(self.epochs):
      x = torch.randperm(n).tolist()
      for v in x:
        yield v


class CycleBatchSampler(sampler.BatchSampler):

  def __init__(self, sampler, batch_size, drop_last, schedule, long_cycle_bs_scale):
    super(CycleBatchSampler, self).__init__(sampler, batch_size, drop_last)

    # long cycles -> step-wise.
    # input of [last_step_1, last_step_2, ...]
    self.schedule = schedule
    # e.g., [8,4,2,1] for the 4 shape
    self.long_cycle_bs_scale = long_cycle_bs_scale

    # evenly divide each interval into 4 chunks
    # each of these 4 chunks gets assigned to one lone cycle
    # compute BS for each cycle


    # For each short cycle, do updates as well

  def __iter__(self):
    iteration_counter = 0
    phase = 1
    phase_steps = self.schedule[phase] / len(self.long_cycle_bs_scale)
    long_cycle_index = 0
    iter_offset = 0

    batch_size = self.batch_size * self.long_cycle_bs_scale[long_cycle_index]
    short_cycle_batch = batch_size
    batch = []
    for idx in self.sampler:
      batch.append((idx, long_cycle_index))
      if len(batch) == short_cycle_batch:
        yield batch
        #print(iteration_counter, len(batch), long_cycle_index)
        batch = []
        iteration_counter += 1
        if iteration_counter > self.schedule[phase]:
          phase += 1
          phase_steps = ((self.schedule[phase] - self.schedule[phase-1]) /
                         len(self.long_cycle_bs_scale))
          long_cycle_index = 0
          if phase == len(self.schedule)-1: # make sure last phase is run without long-cycle changes
            long_cycle_index = -1
          iter_offset = iteration_counter
          batch_size = (self.batch_size *
                        self.long_cycle_bs_scale[long_cycle_index])
        if iteration_counter >= phase_steps + iter_offset:
          long_cycle_index += 1
          if phase == len(self.schedule)-1:
            long_cycle_index = -1

          long_cycle_index = min(long_cycle_index,
                                 len(self.long_cycle_bs_scale)-1)
          iter_offset = iteration_counter
          batch_size = (self.batch_size *
                        self.long_cycle_bs_scale[long_cycle_index])
        if iteration_counter % 3 == 0:
          short_cycle_batch = batch_size
        if iteration_counter % 3 == 1:
          short_cycle_batch = batch_size * 2
        if iteration_counter % 3 == 2:
          short_cycle_batch = batch_size * 4


    if len(batch) > 0 and not self.drop_last:
      yield batch
