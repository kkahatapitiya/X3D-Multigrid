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
    #for e in range(self.epochs):
    while True:
      x = torch.randperm(n).tolist()
      for v in x:
        yield v


class CycleBatchSampler(sampler.BatchSampler):

  def __init__(self, sampler, batch_size, drop_last, schedule, cur_iterations, long_cycle_bs_scale):
    super(CycleBatchSampler, self).__init__(sampler, batch_size, drop_last)

    # long cycles -> step-wise.
    # input of [last_step_1, last_step_2, ...]
    self.schedule = schedule
    # e.g., [8,4,2,1] for the 4 shape
    self.long_cycle_bs_scale = long_cycle_bs_scale

    # evenly divide each interval into 4 chunks
    # each of these 4 chunks gets assigned to one lone cycle
    # compute BS for each cycle

    self.iteration_counter = cur_iterations # 0
    self.short_iteration_counter = 0
    self.phase = 1
    self.phase_steps = ((self.schedule[self.phase] - self.schedule[self.phase-1]) / len(self.long_cycle_bs_scale))
    self.long_cycle_index = 0
    self.iter_offset = 0

    # For each short cycle, do updates as well

  def __iter__(self):
    batch_size = self.batch_size * self.long_cycle_bs_scale[self.long_cycle_index]
    self.short_iteration_counter = 0
    batch = []
    for _ in range(5):
      batch_size = self.adjust_long_cycle(batch_size)
    short_cycle_batch = self.adjust_short_cycle(batch_size)

    for idx in self.sampler:
      batch.append((idx, self.long_cycle_index))
      if len(batch) == short_cycle_batch:
        yield batch

        batch = []

        self.iteration_counter += 1
        self.short_iteration_counter += 1
        batch_size = self.adjust_long_cycle(batch_size)
        short_cycle_batch = self.adjust_short_cycle(batch_size)

    if len(batch) > 0 and not self.drop_last:
      yield batch


  def adjust_long_cycle(self, batch_size):
    if self.iteration_counter > self.schedule[self.phase]: # NUMBER OF LONG CYCLES
      self.iter_offset = self.schedule[self.phase]
      self.phase += 1
      self.phase_steps = ((self.schedule[self.phase] - self.schedule[self.phase-1]) / len(self.long_cycle_bs_scale))
      self.long_cycle_index = 0
      if self.phase == len(self.schedule)-1: # make sure last phase is run without long-cycle changes
        self.long_cycle_index = -1
      batch_size = (self.batch_size * self.long_cycle_bs_scale[self.long_cycle_index])

    elif self.iteration_counter >= self.phase_steps + self.iter_offset: # INSIDE LONG CYCLES
      self.iter_offset += self.phase_steps
      self.long_cycle_index += 1
      if self.phase == len(self.schedule)-1:
        self.long_cycle_index = -1

      self.long_cycle_index = min(self.long_cycle_index, len(self.long_cycle_bs_scale)-1)
      batch_size = (self.batch_size * self.long_cycle_bs_scale[self.long_cycle_index])

    return batch_size


  def adjust_short_cycle(self, batch_size):
    # IN MULTI-GRID PAPER FIG.2(d) FOR LARGER BATCH SIZES (8x,16x) ONLY 2 SHORT CYCLES USED (INSTEAD OF 4)
    if self.long_cycle_index in [0,1]:
        if self.short_iteration_counter % 2 == 0:
          short_cycle_batch = batch_size * 2
        if self.short_iteration_counter % 2 == 1:
          short_cycle_batch = batch_size
    else:
        if self.short_iteration_counter % 3 == 0:
          short_cycle_batch = batch_size * 4
        if self.short_iteration_counter % 3 == 1:
          short_cycle_batch = batch_size * 2
        if self.short_iteration_counter % 3 == 2:
          short_cycle_batch = batch_size

    return short_cycle_batch
