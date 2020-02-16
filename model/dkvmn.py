# Reference: https://github.com/tianlinyang/DKVMN
import torch
import torch.nn as nn
import numpy as np
import json
import torch.nn.init


# Utils
def varible(tensor, device):
    return torch.autograd.Variable(tensor).to(device)


# def to_scalar(var):
#     return var.view(-1).data.tolist()[0]


# def save_checkpoint(state, track_list, filename):
#     with open(filename + '.json', 'w') as f:
#         json.dump(track_list, f)
#     torch.save(state, filename + '.model')


# def adjust_learning_rate(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# Memory
class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = torch.nn.Linear(
                self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(
                self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    def addressing(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = torch.nn.functional.softmax(
            similarity_score, dim=1)  # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(
                control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(
                control_input=control_input, memory=memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory


class DKVMN(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)

        self.memory_key = init_memory_key

        # self.memory_value = self.init_memory_value
        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(
            control_input=control_input, memory=self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(
            memory=self.memory_value, read_weight=read_weight)

        return read_content

    def write(self, write_weight, control_input, if_write_memory):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)
        # if_write_memory = torch.cat([if_write_memory.unsqueeze(1) for _ in range(self.memory_value_state_dim)], 1)

        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value


# Model
class MODEL(nn.Module):

    def __init__(self, config, device):
        super(MODEL, self).__init__()
        self.config = config
        self.device = device
        self.n_question = config.n_skills
        self.batch_size = config.batch_size
        self.q_embed_dim = 50
        self.qa_embed_dim = 200
        self.memory_size = 20
        self.memory_key_state_dim = self.q_embed_dim
        self.memory_value_state_dim = self.qa_embed_dim
        self.final_fc_dim = 50
        self.student_num = None

        self.input_embed_linear = nn.Linear(
            self.q_embed_dim, self.final_fc_dim, bias=True)
        self.read_embed_linear = nn.Linear(
            self.memory_value_state_dim + self.final_fc_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(
            self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(
            self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                         memory_key_state_dim=self.memory_key_state_dim,
                         memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat(
            [self.init_memory_value.unsqueeze(0) for _ in range(self.batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        self.q_embed = nn.Embedding(
            self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(
            2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

        self.init_embeddings()
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
        # nn.init.constant(self.input_embed_linear.bias, 0)
        # nn.init.normal(self.input_embed_linear.weight, std=0.02)

    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target, student_id=None):

        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat(
            [self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        predict_logs = []
        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)
            if_memory_write = slice_q_data[i].squeeze(1).ge(1)
            if_memory_write = varible(torch.FloatTensor(
                if_memory_write.data.tolist()), self.device)

            # Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(
                correlation_weight, qa, if_memory_write)

            # read_content_embed = torch.tanh(self.read_embed_linear(torch.cat([read_content, q], 1)))
            # pred = self.predict_linear(read_content_embed)
            # predict_logs.append(pred)

        all_read_value_content = torch.cat(
            [value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat(
            [input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        # input_embed_content = input_embed_content.view(batch_size * seqlen, -1)
        # input_embed_content = torch.tanh(self.input_embed_linear(input_embed_content))
        # input_embed_content = input_embed_content.view(batch_size, seqlen, -1)

        predict_input = torch.cat(
            [all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(
            predict_input.view(batch_size*seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        # predicts = torch.cat([predict_logs[i] for i in range(seqlen)], 1)
        target_1d = target                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            filtered_pred, filtered_target)

        #print(filtered_pred, filtered_pred.shape) #-> torch.Size([6399])
        out = {
            'loss': loss,
            'filtered_pred': torch.sigmoid(filtered_pred),
            'filtered_target': filtered_target,
            # 'pred_vect': pred_vect,  # (20, 100, 124)
            # 'pred_prob': pred_prob,  # (20, 100)
        }
        return out

    def loss_batch(self, xseq, yseq, opt=None):
        i_skill = self.config.n_skills
        device = self.device
        # q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        q_one_seq = torch.matmul(yseq.float().to(device), torch.Tensor(
            [[1], [0]]).to(device)).long().to(device)
        # qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = torch.matmul(yseq.float().to(device), torch.Tensor(
            [[1], [i_skill]]).to(device)).long().to(device)
        # target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = torch.matmul(yseq.float().to(device), torch.Tensor(
            [[1], [i_skill]]).to(device)).long().to(device)
        # [[24. 24. 24. ...  0.  0.  0.]
        # ...
        # [29.  3. 29. ... 59. 41. 41.]] (32, 200)
        # [[ 24. 134.  24. ...   0.   0.   0.]
        # ...
        # [ 29.   3.  29. ... 169.  41. 151.]] (32, 200)
        # [[ 24. 134.  24. ...   0.   0.   0.]
        # ...
        # [ 29.   3.  29. ... 169.  41. 151.]] (32, 200)
        q_one_seq = q_one_seq.squeeze(2)
        qa_batch_seq = qa_batch_seq.squeeze(2)
        target = target.squeeze(2)
        # print(q_one_seq, q_one_seq.shape)
        # print(qa_batch_seq, qa_batch_seq.shape)
        # print(target, target.shape)

        target = (target.float().cpu().numpy() - 1) / self.n_question
        target = np.floor(target)
        input_q = varible(torch.LongTensor(q_one_seq), self.device)
        input_qa = varible(torch.LongTensor(qa_batch_seq), self.device)
        target = varible(torch.FloatTensor(target), self.device)
        target_to_1d = torch.chunk(target, self.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i]
                               for i in range(self.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        out = self.forward(input_q, input_qa, target_1d)
        loss = out['loss']

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return out
