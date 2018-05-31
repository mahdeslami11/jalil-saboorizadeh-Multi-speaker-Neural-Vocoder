import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils import weight_norm

import numpy as np

verbose = False


class SampleRNNGAN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels, ulaw, weight_norm, ind_cond_dim, spk_dim, qrnn):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        self.ulaw = ulaw
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
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, IsCond, ind_cond_dim, spk_dim, weight_norm, qrnn
            )
            for (frame_size, n_frame_samples, IsCond) in zip(
                frame_sizes, ns_frame_samples, is_cond
            )
        ])

        # frame sizes[0]: 16
        self.sample_level_mlp = SampleLevelMLP(frame_sizes[0], dim, q_levels, weight_norm)

    @property
    def look_back(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim, learn_h0, is_cond, ind_cond_dim, spk_dim, w_norm, qrnn):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim
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
            self.cond_expand = torch.nn.Conv1d(
                in_channels=ind_cond_dim,
                out_channels=dim,
                kernel_size=1
            )

            # Initialize 1D-Convolution (Fully-connected Layer) for acoustic conditioners
            init.kaiming_uniform(self.cond_expand.weight)
            init.constant(self.cond_expand.bias, 0)
            if self.weight_norm:
                self.cond_expand = weight_norm(self.cond_expand, name='weight')

            # Speaker embedding
            self.spk_embedding = torch.nn.Embedding(
                self.spk_dim,
                dim
            )

        else:
            self.spk_embedding = None
        init.kaiming_uniform(self.input_expand.weight)
        init.constant(self.input_expand.bias, 0)

        if self.weight_norm:
            self.input_expand = weight_norm(self.input_expand, name='weight')

        if self.qrnn:
            from torchqrnn import QRNN

            self.rnn = QRNN(
                input_size=dim,
                hidden_size=dim,
                num_layers=n_rnn,
            )
   
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

    def forward(self, prev_samples, upper_tier_conditioning, hidden, cond_cnn, spk):
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
            if verbose:
                print('Input rnn has size:', input_rnn.size())
                print('Before expansion, conditioner has size: ', cond_cnn.size())
            cond_rnn = self.cond_expand(cond_cnn.permute(0, 2, 1)).permute(0, 2, 1)
            input_rnn += cond_rnn
            if verbose:
                print('After expansion, conditioner has size: ', cond_rnn.size())
                print('Compute speaker embedding for spk of size: ', spk.size())
            spk_embed = self.spk_embedding(spk.long())
            if verbose:
                print('Embedding has size: ', spk_embed.size())
            input_rnn += spk_embed
            if verbose:
                print('After adding speaker, input rnn has size:', input_rnn.size())

        reset = hidden is None

        if reset:
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

    def __init__(self, frame_size, dim, q_levels, w_norm):
        super().__init__()

        self.q_levels = q_levels
        self.weight_norm = w_norm

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


class ConditionerCNN(torch.nn.Module):
    def __init__(self, cond_dim, ind_cond_dim, w_norm):
        super().__init__()
        self.weight_norm = w_norm
        # Acoustic conditioners expansion
        self.cond_bottle_neck = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=cond_dim,
                out_channels=40,
                kernel_size=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=40,
                out_channels=30,
                kernel_size=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=30,
                out_channels=20,
                kernel_size=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=20,
                out_channels=ind_cond_dim,
                kernel_size=1
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        if verbose:
            print('Conditioner CNN input shape:', x.size())
        x = self.cond_bottle_neck(x.permute(0, 2, 1).float()).permute(0, 2, 1)
        if verbose:
            print('Conditioner CNN output shape:', x.size())
        return x


class Discriminant(torch.nn.Module):
    def __init__(self, spk_dim, ind_cond_dim, cond_frames):
        super().__init__()
        self.ind_cond_dim = ind_cond_dim
        self.cond_frames = cond_frames

        # Architecture inspired by paper published as arXiv:1804.02812v1
        self.block1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.InstanceNorm2d(512)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.InstanceNorm2d(512)
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.InstanceNorm2d(512)
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.LeakyReLU(),
            torch.nn.ReflectionPad2d(2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=5
            ),
            torch.nn.InstanceNorm2d(512)
        )
        self.fc = torch.nn.Linear(512 * self.ind_cond_dim * self.cond_frames, spk_dim)

    def forward(self, x):
        if verbose:
            print('Discriminant input shape:', x.size())
        x = x.contiguous().view(-1, 1, self.cond_frames, self.ind_cond_dim)
        x = self.block1(x)
        x = self.block2(x) + x
        x = self.block3(x) + x
        x = self.block4(x) + x
        if verbose:
            print('Discriminant shape before FC layer:', x.size())
        x = x.view(-1, 512 * self.ind_cond_dim * self.cond_frames)
        x = self.fc(x)
        if verbose:
            print('Discriminant output shape:', x.size())
        return F.log_softmax(x)


class Runner:

    def __init__(self, samplernn_model, conditioner_model, discriminant_model):
        super().__init__()
        self.samplernn_model = samplernn_model
        self.conditioner_model = conditioner_model
        self.discriminant_model = discriminant_model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.samplernn_model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning, cond_cnn, spk):
        (output, new_hidden) = rnn(
            prev_samples, upper_tier_conditioning, self.hidden_states[rnn], cond_cnn, spk
        )

        self.hidden_states[rnn] = new_hidden.detach()
        return output

    def run_cond(self, cond):
        cond_cnn = self.conditioner_model(cond)
        return cond_cnn

    def run_discriminant(self, cond_cnn):
        output = self.discriminant_model(cond_cnn)
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, samplernn_model, conditioner_model, discriminant_model):
        super().__init__(samplernn_model, conditioner_model, discriminant_model)

    def forward(self, input_sequences, reset, cond, spk):
        if reset:
            self.reset_hidden_states()

        spk_prediction = None

        # input_seq: 128 x 1087; reset: boolean

        (batch_size, _) = input_sequences.size()
        # print('model input', input.size())
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
        for rnn in reversed(self.samplernn_model.frame_level_rnns):
            from_index = self.samplernn_model.look_back - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            if verbose:
                print('predictor rnn ', from_index, to_index)

            # prev_samples = 2 * utils.linear_dequantize(
            prev_samples = 2 * self.samplernn_model.dequantize(
                input_sequences[:, from_index: to_index],
                self.samplernn_model.q_levels
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

            if upper_tier_conditioning is None:
                cond_cnn = self.run_cond(cond)
                if verbose:
                    print('predictor rnn prev_samples view', prev_samples.size())
                    print('Cond_cnn has size:', cond_cnn.size())
                upper_tier_conditioning = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning, cond_cnn, spk
                )

                spk_prediction = self.run_discriminant(cond_cnn)
            else:
                cond_cnn = None
                spk = None
                upper_tier_conditioning = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning, cond_cnn, spk
                )

        bottom_frame_size = self.samplernn_model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences[:, self.samplernn_model.look_back - bottom_frame_size:]

        if verbose:
            print('predictor mlp', self.samplernn_model.look_back - bottom_frame_size)
            print('predictor mlp input seq', mlp_input_sequences.size())
            print('predictor mlp upper tier_cond', upper_tier_conditioning.size(), '\n')

        # predictor mlp 48
        # predictor mlp input seq torch.Size([128, 1039])
        # predictor mlp upper tier_cond torch.Size([128, 1024, 512])

        return self.samplernn_model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        ), spk_prediction


class Generator(Runner):

    def __init__(self, samplernn_model, conditioner_model, discriminant_model, cuda=False):
        super().__init__(samplernn_model, conditioner_model, discriminant_model)
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len, cond, spk):
        # generation doesn't work with CUDNN for some reason

        cuda_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        self.reset_hidden_states()
        (num_cond, n_dim) = cond.shape
        condtot = cond
        global_spk = spk
        seq_len = num_cond*self.samplernn_model.look_back
        print('seq len', seq_len)
        print('model look-back', self.samplernn_model.look_back)
        bottom_frame_size = self.samplernn_model.frame_level_rnns[0].n_frame_samples
        sequences = torch.LongTensor(n_seqs, self.samplernn_model.look_back + seq_len).\
            fill_(utils.q_zero(self.samplernn_model.q_levels))
        frame_level_outputs = [None for _ in self.samplernn_model.frame_level_rnns]

        for i in range(self.samplernn_model.look_back, self.samplernn_model.look_back + seq_len):
            for (tier_index, rnn) in reversed(list(enumerate(self.samplernn_model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                print('Predicting sample ', i)
                prev_samples = torch.autograd.Variable(
                    2 * self.samplernn_model.dequantize(
                        sequences[:, i - rnn.n_frame_samples: i],
                        self.samplernn_model.q_levels
                    ).unsqueeze(1),
                    volatile=True
                )
                # print('prev samples', prev_samples)
                if self.cuda:
                    prev_samples = prev_samples.cuda()

                if tier_index == len(self.samplernn_model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                    j = i // self.samplernn_model.look_back - 1
                    cond = condtot[j, :]
                    cond = torch.from_numpy(cond.reshape(1, 1, n_dim))
                    spk = global_spk
                    spk = torch.from_numpy(np.array(spk).reshape(1, 1))
                else:
                    cond = None
                    spk = None
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.samplernn_model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                        .unsqueeze(1)

                if cond is not None:
                    if self.cuda:
                        cond = Variable(cond).cuda()
                    cond_cnn = self.run_cond(cond)
                else:
                    cond_cnn = None

                if self.cuda:
                    spk = Variable(spk).cuda()

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning, cond_cnn, spk
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
            sample_dist = self.samplernn_model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = cuda_enabled

        return self.samplernn_model.dequantize(sequences[:, self.samplernn_model.look_back:],
                                               self.samplernn_model.q_levels)
