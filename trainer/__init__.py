import torch
from torch.autograd import Variable

import heapq


# Based on torch.utils.trainer.Trainer code.
# Allows multiple inputs to the model, not all need to be Tensors.
class Trainer(object):
    def __init__(self, model, criterion_rnn, criterion_discriminant, optimizer, dataset, cuda, scheduler,
                 lambda_weight, frames_spk):
        self.model = model
        self.criterion_rnn = criterion_rnn
        self.criterion_discriminant = criterion_discriminant
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        self.frames_spk = frames_spk
        self.cond_cnn = None

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
            if self.scheduler is not None:
                self.scheduler.step()
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

            def closure():
                batch_output, spk_prediction = self.model(*batch_inputs, batch_cond, batch_spk,
                                                          self.cond_cnn, self.frames_spk)

                loss1 = self.criterion_rnn(batch_output, batch_target)
                loss1.backward()

                if spk_prediction is not None:
                    loss2 = self.criterion_discriminant(spk_prediction, batch_spk)
                    self.spk_loss = loss2
                    loss2.backward()
                    self.cond_cnn = None
                else:
                    loss2 = self.spk_loss

                current_lambda_weight = (self.iterations/self.lambda_weight[2]) * \
                                        (self.lambda_weight[1]-self.lambda_weight[0]) + self.lambda_weight[0]

                loss = loss1-current_lambda_weight*loss2

                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins(
                'iteration', self.iterations, batch_inputs, batch_target,
                *plugin_data
            )
            self.call_plugins('update', self.iterations, self.model)
