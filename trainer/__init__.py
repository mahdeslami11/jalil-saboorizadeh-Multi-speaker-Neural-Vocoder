import torch
from torch.autograd import Variable

import heapq


# Based on torch.utils.trainer.Trainer code.
# Allows multiple inputs to the model, not all need to be Tensors.
class Trainer(object):
    def __init__(self, model, criterion_rnn, criterion_discriminant, optimizer_samplernn, optimizer_discriminant,
                 dataset, cuda, scheduler_samplernn, scheduler_discriminant, lambda_weight):
        self.model = model
        self.criterion_rnn = criterion_rnn
        self.criterion_discriminant = criterion_discriminant
        self.optimizer_samplernn = optimizer_samplernn
        self.optimizer_discriminant = optimizer_discriminant
        self.scheduler_samplernn = scheduler_samplernn
        self.scheduler_discriminant = scheduler_discriminant
        self.dataset = dataset
        self.cuda = cuda
        self.iterations = 0
        self.epochs = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        self.lambda_weight = lambda_weight
        self.loss2 = None

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for self.epochs in range(self.epochs + 1, self.epochs + int(epochs) + 1):
            self.train()
            if self.scheduler_samplernn is not None:
                self.scheduler_samplernn.step()
            if self.scheduler_discriminant is not None:
                self.scheduler_discriminant.step()
            self.call_plugins('epoch', self.epochs)

    def train(self):
        for (self.iterations, data) in enumerate(self.dataset, self.iterations + 1):
            inputs = data[0]
            reset = data[1]
            batch_target = data[2]
            if reset[0] == 1:
                reset = True
            else:
                reset = False
            batch_inputs = (inputs, reset)
            batch_cond = data[3]
            batch_spk = data[4]

            self.call_plugins(
                'batch', self.iterations, batch_inputs, batch_target, batch_cond, batch_spk
            )

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input)
                    if self.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target)
            batch_cond = Variable(batch_cond)
            batch_spk = Variable(batch_spk)

            if self.cuda:
                batch_target = batch_target.cuda()
                batch_cond = batch_cond.cuda()
                batch_spk = batch_spk.cuda()
                
            plugin_data = [None, None]

            def closure_samplernn():
                batch_output, spk_prediction = self.model(*batch_inputs, batch_cond, batch_spk)

                loss1 = self.criterion_rnn(self.batch_output, batch_target)

                self.loss2 = self.criterion_discriminant(self.spk_prediction, batch_spk)

                current_lambda_weight = (self.iterations/self.lambda_weight[2]) * \
                                        (self.lambda_weight[1]-self.lambda_weight[0]) + self.lambda_weight[0]

                loss = loss1-current_lambda_weight*loss2

                loss.backward()

                if plugin_data[0] is None:
                    plugin_data[0] = self.batch_output.data
                    plugin_data[1] = loss.data

                return loss

            def closure_discriminant():
                if plugin_data[0] is None:
                    plugin_data[0] = self.batch_output.data
                    plugin_data[1] = self.loss2.data

                return self.loss2

            self.optimizer_samplernn.zero_grad()
            self.optimizer_samplernn.step(closure_samplernn)

            self.optimizer_discriminant.zero_grad()
            self.optimizer_discriminant.step(closure_discriminant)

            self.call_plugins(
                'iteration', self.iterations, batch_inputs, batch_target,
                *plugin_data
            )
            self.call_plugins('update', self.iterations, self.model)
