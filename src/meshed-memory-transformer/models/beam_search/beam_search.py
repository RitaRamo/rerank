import torch
import utils
import numpy as np


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        print("\n\n _expand_state")

        def fn(s):
            shape = [int(sh) for sh in s.shape]
            print("shape", shape)
            beam = selected_beam
            print("beam", beam)

            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            print("beam again", beam)
            print("self.beam_size",self.beam_size)
            print("self.b_s",self.b_s)
            print("cur_beam_size",cur_beam_size)
            print("shape[1:])", shape[1:])
            print("index 0", [self.b_s, cur_beam_size] + shape[1:])
            print("index0", ([self.b_s, self.beam_size] + shape[1:]))
            print("s.view(*([self.b_s, cur_beam_size] + shape[1:]))", s.view(*([self.b_s, cur_beam_size] + shape[1:])))
            print("other", beam.expand(*([self.b_s, self.beam_size] + shape[1:])))

            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])).long())
            print("this is s", s)
            print("this is s", s.size())

            s = s.view(*([-1, ] + shape[1:]))
            print("this is s.size(", s.size())

            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, visual: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
        print("\n\n function apply")
        self.b_s = utils.get_batch_size(visual)
        print("self.b_s batch size", self.b_s)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        print("self.seq_mask",  self.seq_mask)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        print("self.seq_logprob", self.seq_logprob)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []
        with self.model.statefulness(self.b_s):
            print("entrei no statefulness")
            for t in range(self.max_len):
                visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)
                print("visual", visual)
                print("outputs", outputs)
        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        print("seq_logprob", seq_logprob)
        print("sort_idxs", sort_idxs)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        print("\n\nfunction select")
        
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        print("selected_logprob", selected_logprob)
        print("selected_idx", selected_idx)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        print("selected_idx", selected_idx)
        print("selected_logprob", selected_logprob)
        return selected_idx, selected_logprob

    def iter(self, t: int, visual: utils.TensorOrSequence, outputs, return_probs, **kwargs):
        print("\n\nfunction iter")
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(t, self.selected_words, visual, None, mode='feedback', **kwargs)
        print("\nword_logprob", word_logprob)
        print("word_logprob extender", word_logprob[0,:100])
        print("\nword_logprob size(", word_logprob.size())

        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        print("word_logprob after view", word_logprob.size())
        print("word_logprob after view", word_logprob)

        candidate_logprob = self.seq_logprob + word_logprob
        print("candidate_logprob", candidate_logprob)
        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            print("mask", mask)
            self.seq_mask = self.seq_mask * mask
            print("self.seq_mask", self.seq_mask)
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            print("word_logprob", word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            print("old_seq_logprob", old_seq_logprob)
            old_seq_logprob[:, :, 1:] = -999
            print("old_seq_logprob", old_seq_logprob)
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)
            print("candidate_logprob", candidate_logprob)
        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        print("selected_idx", selected_idx)
        print("selected_logprob", selected_logprob)
        selected_beam = selected_idx / candidate_logprob.shape[-1]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
        print("selected_beam", selected_beam)
        print("selected_words", selected_words)

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        visual = self._expand_visual(visual, cur_beam_size, selected_beam)
        print("this is visual", visual)
        print("this is visual size", visual.size())
        self.seq_logprob = selected_logprob.unsqueeze(-1)
        print("seq log prog", self.seq_logprob)
        print("self.seq_mask before", self.seq_mask)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        print("self seq mask after gather", self.seq_mask)
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        print("outputs lists", outputs)
        
        outputs.append(selected_words.unsqueeze(-1))
        print("outputs", outputs)

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        print("this is word log prob", this_word_logprob)
        print("this is word log prob size", this_word_logprob.size())

        
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        
        print("this is word log prob aftter", this_word_logprob)
        print("this is word log prob aftter size", this_word_logprob.size())

        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        
        print("self.log_probs", self.log_prob)
        print("self.log_probs", self.log_prob.size())

        self.log_probs.append(this_word_logprob)
        
        print("self.log_probs with word log prob", self.log_prob)
        print("self.log_probs with word log prob", self.log_prob.size())

        self.selected_words = selected_words.view(-1, 1)
        print("selected words",self.selected_words)
        return visual, outputs
