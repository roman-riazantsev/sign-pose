import os

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class ModelProjective:
    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.lr = config['lr']
        self.log_rate = config['log_rate']
        self.save_rate = config['save_rate']

        build_id = config['build_id']
        self.save_path = f'results/{build_id}/{build_id}.pt'
        self.writer = SummaryWriter(f'results/{build_id}')

        print(self.save_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        save_exists = os.path.exists(self.save_path)
        if save_exists:
            checkpoint = torch.load(self.save_path)

        self.parameter_names = ['p00', 'p01', 'p02', 'p10', 'p11', 'p12']
        self.parameters = []

        for p_name in self.parameter_names:
            if save_exists:
                setattr(self, p_name, checkpoint[p_name])
            else:
                var = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
                setattr(self, p_name, var)
            self.parameters.append(getattr(self, p_name))

        self.optimizer = optim.SGD(self.parameters, lr=self.lr)
        #     self.optimizer = optim.Adam(self.parameters, lr=self.lr)

        if save_exists:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.g_step = checkpoint['g_step'] + 1
            self.loss = checkpoint['loss']
        else:
            self.g_step = 1
            self.loss = 0

    def fit(self, A, b_true):
        for i in range(self.n_steps):
            def closure():
                projective_tensor = self.init_projective_tensor()

                b_pred = torch.transpose(self.matrix_mul([projective_tensor, torch.transpose(A, 0, 1)]), 0, 1)
                b_pred = b_pred[:, :2] / b_pred[:, -1:]
                d_1 = b_pred - b_true
                self.loss = torch.norm(d_1, p=2)

                self.loss.backward()

                if self.g_step % self.log_rate == 0:
                    print(f"step: {self.g_step}, loss: {self.loss}")
                    self.writer.add_scalar('training loss',
                                           self.loss,
                                           self.g_step)
                    print(f"m: {projective_tensor}")

                if self.g_step % self.save_rate == 0:
                    self.save_state()

                self.g_step += 1
                return self.loss

            self.optimizer.step(closure)

    def save_state(self):
        state_dict = {
            'g_step': self.g_step,
            'loss': self.loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        for p_name in self.parameter_names:
            state_dict[p_name] = getattr(self, p_name)

        torch.save(state_dict, self.save_path)
        print(f'Parameters saved at step {self.g_step}; loss {self.loss}.')

    def init_projective_tensor(self):
        projective_tensor = torch.autograd.Variable(torch.zeros(3, 3), requires_grad=False)

        projective_tensor[0, 0] = self.p00
        projective_tensor[0, 1] = self.p01
        projective_tensor[0, 2] = self.p02

        projective_tensor[1, 0] = self.p10
        projective_tensor[1, 1] = self.p11
        projective_tensor[1, 2] = self.p12

        projective_tensor[2, 0] = 0
        projective_tensor[2, 1] = 0
        projective_tensor[2, 2] = 1

        return projective_tensor.to(self.device)

    @staticmethod
    def matrix_mul(values):
        result = values[0]
        for value in values[1:]:
            result = torch.matmul(result, value)
        return result