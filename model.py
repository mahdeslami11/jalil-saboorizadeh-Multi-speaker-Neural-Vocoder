import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
# from torchqrnn import QRNN

import numpy as np

verbose = False


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels, ulaw, weight_norm, cond_dim, spk_dim, qrnn=False):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        self.ulaw = ulaw
        self.cond_dim = cond_dim
        self.spk_dim = spk_dim

        if ulaw:
            self.dequantize = utils.udequantize
        else:
            self.dequantize = utils.linear_dequantize

        ns_frame_samples = map(int, np.cumprod(frame_sizes))

        # print('frame sizes:', frame_sizes)
        # print('ns frame samples:', list(ns_frame_samples), '\n')

        # frame sizes: [16, 4]
        # ns frame samples: [16, 64]

        # I think frame sizes are in fact the upsampling ratio
        # Lower interp tier: 16 ==> 16 samples (1 ms)
        # Higher inter tier:  4 ==> 4*16 = 64 samples (4 ms)

        is_cond = [False]*len(frame_sizes)
        is_cond[-1] = True
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, IsCond, cond_dim, spk_dim, weight_norm, qrnn
            )
            for (frame_size, n_frame_samples, IsCond) in zip(
                frame_sizes, ns_frame_samples, is_cond
            )
        ])

        # frame sizes[0]: 16
        self.sample_level_mlp = SampleLevelMLP(frame_sizes[0], dim, q_levels, weight_norm)

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, is_cond, cond_dim, spk_dim, w_norm, qrnn):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim
        self.cond_dim = cond_dim
        self.spk_dim = spk_dim
        self.weight_norm = w_norm
        self.qrnn = qrnn

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))

        self.input_expand = torch.nn.Conv1d(
            in_channels=n_frame_samples,
            out_channels=dim,
            kernel_size=1
        )
        if is_cond:
            # Acoustic conditioners expansion
            self.cond_expand = torch.nn.Conv1d(
                in_channels=cond_dim,
                out_channels=dim,
                kernel_size=1
            )

            # Initialize 1D-Convolution (Fully-connected Layer) for acoustic conditioners
            init.kaiming_uniform(self.cond_expand.weight)
            init.constant(self.cond_expand.bias, 0)

            # Speaker embedding
            self.spk_embedding = torch.nn.Embedding(
                self.spk_dim,
                self.spk_dim
            )
            # Speaker embedding expansion
            self.spk_expand = torch.nn.Conv1d(
                in_channels=spk_dim,
                out_channels=dim,
                kernel_size=1
            )

            # Initialize 1D-Convolution (Fully-connected Layer) for acoustic conditioners
            init.kaiming_uniform(self.spk_expand.weight)
            init.constant(self.spk_expand.bias, 0)

            # Apply weight normalization if chosen
            if self.weight_norm:
                self.cond_expand = weight_norm(self.cond_expand, name='weight')
                self.spk_expand = weight_norm(self.spk_expand, name='weight')

        else:
            self.cond_expand = None
        init.kaiming_uniform(self.input_expand.weight)
        init.constant(self.input_expand.bias, 0)

        if self.weight_norm:
            self.input_expand = weight_norm(self.input_expand, name='weight')

        if self.qrnn:
            self.rnn = torch.nn.GRU(
                input_size=dim,
                hidden_size=dim,
                num_layers=n_rnn,
                batch_first=True
            )

            # self.rnn = QRNN(
            # input_size=dim,
            # hidden_size=dim,
            # num_layers=n_rnn,
            # )
   
        else:    
            self.rnn = torch.nn.GRU(
                input_size=dim,
                hidden_size=dim,
                num_layers=n_rnn,
                batch_first=True
            )
        for i in range(n_rnn):
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal]
            )
            init.constant(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.upsampling = nn.LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )
        init.uniform(
            self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        )
        init.constant(self.upsampling.bias, 0)

        if weight_norm:
            self.upsampling.conv_t = weight_norm(self.upsampling.conv_t, name='weight')

    def forward(self, prev_samples, upper_tier_conditioning, hidden, cond, spk):
        (batch_size, _, _) = prev_samples.size()
        # The first called
        # forward rnn     frame_size:  4  n_frame_samples:  64    prev_samples: torch.Size([128, 16, 64])
        # input: torch.Size([128, 16, 512])
        # output before upsampling torch.Size([128, 16, 512]) 	hidden torch.Size([2, 128, 512])
        # output torch.Size([128, 64, 512])
        # (=> 16 frames, 64 input samples/frame)

        # The second called (64 x 16 = 1024)
        # forward rnn 	frame_size:  16 	n_frame_samples:  16 	prev_samples: torch.Size([128, 64, 16])
        # input: torch.Size([128, 64, 512])
        # output before upsampling torch.Size([128, 64, 512]) 	hidden torch.Size([2, 128, 512])
        # output torch.Size([128, 1024, 512])
        # (=> 64 frames, 16 input samples/frame)

        input_rnn = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)
        if upper_tier_conditioning is not None:
            input_rnn += upper_tier_conditioning
        else:
            cond = self.cond_expand(cond.permute(0, 2, 1).float()).permute(0, 2, 1)
            input_rnn += cond
            if verbose:
                print('Input rnn has size:', input_rnn.size())
                print('After expansion, conditioner has size: ', cond.size())
                print('Compute speaker embedding for spk of size: ', spk.size())
            spk_embed = self.spk_embedding(spk.long())
            if verbose:
                print('Embedding has size: ', spk_embed.size())
            spk = self.spk_expand(spk_embed.permute(0, 2, 1).float()).permute(0, 2, 1)
            if verbose:
                print('After expansion, speaker has size: ', spk.size())
            input_rnn += spk
            if verbose:
                print('After adding speaker, input rnn has size:', input_rnn.size())

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()

        # The first called
        # forward rnn     frame_size:  4  n_frame_samples:  64    prev_samples: torch.Size([128, 16, 64])
        # input: torch.Size([128, 16, 512])
        # output before upsampling torch.Size([128, 16, 512]) 	hidden torch.Size([2, 128, 512])
        # output torch.Size([128, 64, 512])
        # (=> 16 frames, 64 input samples/frame)

        # The second called (64 x 16 = 1024)
        # forward rnn 	frame_size:  16 	n_frame_samples:  16 	prev_samples: torch.Size([128, 64, 16])
        # input: torch.Size([128, 64, 512])
        # output before upsampling torch.Size([128, 64, 512]) 	hidden torch.Size([2, 128, 512])
        # output torch.Size([128, 1024, 512])
        # (=> 64 frames, 16 input samples/frame)

        (output, hidden) = self.rnn(input_rnn, hidden)
        if self.qrnn:
            output = output.permute(1, 0, 2)
            hidden = hidden.permute(1, 0, 2)
        output1 = output
        output = self.upsampling(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)

        if verbose:
            print('forward rnn',
                  '\tframe_size: ', self.frame_size,
                  '\tn_frame_samples: ', self.n_frame_samples,
                  '\tprev_samples:', prev_samples.size(),
                  '\tinput:', input_rnn.size(),
                  '\toutput1', output1.size(),
                  '\thidden', hidden.size(),
                  '\toutput', output.size())

        return output, hidden


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, wnorm):
        super().__init__()

        self.q_levels = q_levels
        self.weight_norm = wnorm

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.input = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=frame_size,
            bias=False
        )
        init.kaiming_uniform(self.input.weight)

        self.hidden = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform(self.hidden.weight)
        init.constant(self.hidden.bias, 0)

        self.output = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        nn.lecun_uniform(self.output.weight)
        init.constant(self.output.bias, 0)

        if self.weight_norm:
            self.input = weight_norm(self.input,  'weight')
            self.hidden = weight_norm(self.hidden, 'weight')
            self.output = weight_norm(self.output, 'weight')

    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()

        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        return F.log_softmax(x.view(-1, self.q_levels)) \
                .view(batch_size, -1, self.q_levels)


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning, cond, spk):
        if cond is None:
            (output, new_hidden) = rnn(
                prev_samples, upper_tier_conditioning, self.hidden_states[rnn], cond, spk
            )
        else:
            (output, new_hidden) = rnn(
                prev_samples, upper_tier_conditioning, self.hidden_states[rnn], cond, spk
            )

        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset, cond, spk):
        if reset:
            self.reset_hidden_states()

        # input_seq: 128 x 1087; reset: boolean

        (batch_size, _) = input_sequences.size()
        # print('model input', imput.size())
        (batch_size, numcond, cond_dim) = cond.size()
        # print('model cond', cond.size())

        # predictor rnn 0 -79
        # predictor rnn prev_samples torch.Size([128, 1040])
        # predictor rnn prev_samples view torch.Size([128, 13, 80])

        # predictor rnn 60 -19
        # predictor rnn prev_samples torch.Size([128, 1024])
        # predictor rnn uppertier_cond torch.Size([128, 52, 1024])
        # predictor rnn prev_samples view torch.Size([128, 52, 20])

        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            if verbose:
                print('predictor rnn ', from_index, to_index)

            # prev_samples = 2 * utils.linear_dequantize(
            prev_samples = 2 * self.model.dequantize(
                input_sequences[:, from_index: to_index],
                self.model.q_levels
            )
            if upper_tier_conditioning is None:
                cond = cond[:, :, :]
                cond = cond.contiguous().view(
                    batch_size, -1, cond_dim
                )

                spk = spk.contiguous().view(
                    batch_size, -1
                )

            if verbose:
                print('conditioner size =', cond.size())
                print('spk size =', spk.size())
                print('predictor rnn prev_samples', prev_samples.size())
                if upper_tier_conditioning is not None:
                    print('predictor rnn upper tier cond', upper_tier_conditioning.size())

            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )
            if verbose:
                print('predictor rnn prev_samples view', prev_samples.size())
            if upper_tier_conditioning is None:
                upper_tier_conditioning = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning, cond, spk
                )
            else:
                cond = None
                spk = None
                upper_tier_conditioning = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning, cond, spk
                )

        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences[:, self.model.lookback - bottom_frame_size:]

        if verbose:
            print('predictor mlp', self.model.lookback-bottom_frame_size)
            print('predictor mlp inputseq', mlp_input_sequences.size())
            print('predictor mlp uppertier_cond', upper_tier_conditioning.size(), '\n')

        # predictor mlp 48
        # predictor mlp inputseq torch.Size([128, 1039])
        # predictor mlp uppertier_cond torch.Size([128, 1024, 512])

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len, cond, spk):
        # generation doesn't work with CUDNN for some reason

        cuda_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        self.reset_hidden_states()
        (num_cond, n_dim) = cond.shape
        condtot = cond
        seq_len = num_cond*self.model.lookback
        print('seq len', seq_len)
        print('model look-back', self.model.lookback)
        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = torch.LongTensor(n_seqs, self.model.lookback + seq_len).fill_(utils.q_zero(self.model.q_levels))
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                # 2 * utils.linear_dequantize(
                print('Predicting sample ', i)
                prev_samples = torch.autograd.Variable(
                    2 * self.model.dequantize(
                        sequences[:, i - rnn.n_frame_samples: i],
                        self.model.q_levels
                    ).unsqueeze(1),
                    volatile=True
                )
                # print('prev samples', prev_samples)
                if self.cuda:
                    prev_samples = prev_samples.cuda()

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                    j = i//self.model.lookback - 1
                    print('Using conditioning ', j)
                    cond = condtot[j, :]
                    cond = torch.from_numpy(cond.reshape(1, 1, n_dim))

                else:
                    cond = None
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                        .unsqueeze(1)

                if self.cuda:
                    cond = Variable(cond).cuda()
                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning, cond, spk
                )
            # print('frame out', frame_level_outputs)
            prev_samples = torch.autograd.Variable(
                sequences[:, i - bottom_frame_size: i],
                volatile=True
            )
            # print('prev samples', prev_samples)
            if self.cuda:
                prev_samples = prev_samples.cuda()
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = cuda_enabled

        return self.model.dequantize(sequences[:, self.model.lookback:], self.model.q_levels)
