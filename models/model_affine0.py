import os

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class ModelAffine0:
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

        self.parameter_names = ['sx', 'tx_1', 'ty_1', 'tz_1']
        self.parameters = []

        for p_name in self.parameter_names:
            if save_exists:
                setattr(self, p_name, checkpoint[p_name])
            else:
                var = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
                setattr(self, p_name, var)
            self.parameters.append(getattr(self, p_name))

        self.optimizer = optim.SGD(self.parameters, lr=self.lr)

        if save_exists:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.g_step = checkpoint['g_step'] + 1
            self.loss = checkpoint['loss']
        else:
            self.g_step = 1
            self.loss = 0

    def fit(self, A, b_true, K):
        for i in range(self.n_steps):
            def closure():
                # rx_t, ry_t, rz_t = self.init_rotation_tensors()
                s_t = self.init_scale_tensor()
                _, t_t_1 = self.init_translation_tensors()

                tensors = [s_t, t_t_1]

                b_pred = self.matrix_mul([A, *tensors])

                d_1 = b_pred - b_true
                loss_1 = torch.norm(d_1, p=2)

                # d_2 = torch.matmul(b_pred[..., :-1], K) - torch.matmul(b_true[..., :-1], K)
                # loss_2 = torch.norm(d_2, p=2)

                self.loss = loss_1
                self.loss.backward()

                if self.g_step % self.log_rate == 0:
                    print(f"step: {self.g_step}, loss: {self.loss}")
                    self.writer.add_scalar('training loss',
                                           self.loss,
                                           self.g_step)
                    # print(f"m: {self.matrix_mul(tensors)}")

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

    def init_scale_tensor(self):
        scale_t = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

        scale_t[0, 0] = self.scale
        scale_t[0, 1] = 0
        scale_t[0, 2] = 0
        scale_t[0, 3] = 0

        scale_t[1, 0] = 0
        scale_t[1, 1] = self.scale
        scale_t[1, 2] = 0
        scale_t[1, 3] = 0

        scale_t[2, 0] = 0
        scale_t[2, 1] = 0
        scale_t[2, 2] = self.scale
        scale_t[2, 3] = 0

        scale_t[3, 0] = 0
        scale_t[3, 1] = 0
        scale_t[3, 2] = 0
        scale_t[3, 3] = 1

        return scale_t.to(self.device)

    def init_rotation_tensors(self):
        rot_x = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)
        rot_y = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)
        rot_z = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

        rot_x[0, 0] = 1
        rot_x[0, 1] = 0
        rot_x[0, 2] = 0
        rot_x[0, 3] = 0

        rot_x[1, 0] = 0
        rot_x[1, 1] = self.rx.cos()
        rot_x[1, 2] = self.rx.sin()
        rot_x[1, 3] = 0

        rot_x[2, 0] = 0
        rot_x[2, 1] = -self.rx.sin()
        rot_x[2, 2] = self.rx.cos()
        rot_x[2, 3] = 0

        rot_x[3, 0] = 0
        rot_x[3, 1] = 0
        rot_x[3, 2] = 0
        rot_x[3, 3] = 1

        rot_y[0, 0] = self.ry.cos()
        rot_y[0, 1] = 0
        rot_y[0, 2] = -self.ry.sin()
        rot_y[0, 3] = 0

        rot_y[1, 0] = 0
        rot_y[1, 1] = 1
        rot_y[1, 2] = 0
        rot_y[1, 3] = 0

        rot_y[2, 0] = self.ry.sin()
        rot_y[2, 1] = 0
        rot_y[2, 2] = self.ry.cos()
        rot_y[2, 3] = 0

        rot_y[3, 0] = 0
        rot_y[3, 1] = 0
        rot_y[3, 2] = 0
        rot_y[3, 3] = 1

        rot_z[0, 0] = self.rz.cos()
        rot_z[0, 1] = -self.rz.sin()
        rot_z[0, 2] = 0
        rot_z[0, 3] = 0

        rot_z[1, 0] = self.rz.sin()
        rot_z[1, 1] = self.rz.cos()
        rot_z[1, 2] = 0
        rot_z[1, 3] = 0

        rot_z[2, 0] = 0
        rot_z[2, 1] = 0
        rot_z[2, 2] = 1
        rot_z[2, 3] = 0

        rot_z[3, 0] = 0
        rot_z[3, 1] = 0
        rot_z[3, 2] = 0
        rot_z[3, 3] = 1

        return rot_x.to(self.device), rot_y.to(self.device), rot_z.to(self.device)

    def init_translation_tensors(self):
        def generate(t_x, t_y, t_z):
            trans_t = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

            trans_t[0, 0] = 1
            trans_t[0, 1] = 0
            trans_t[0, 2] = 0
            trans_t[0, 3] = t_x

            trans_t[1, 0] = 0
            trans_t[1, 1] = 1
            trans_t[1, 2] = 0
            trans_t[1, 3] = t_y

            trans_t[2, 0] = 0
            trans_t[2, 1] = 0
            trans_t[2, 2] = 1
            trans_t[2, 3] = t_z

            trans_t[3, 0] = 0
            trans_t[3, 1] = 0
            trans_t[3, 2] = 0
            trans_t[3, 3] = 1

            return trans_t

        # t_t_0 = generate(self.tx_0, self.ty_0, self.tz_0)
        t_t_1 = generate(self.tx_1, self.ty_1, self.tz_1)

        return t_t_1.to(self.device), t_t_1.to(self.device)

    @staticmethod
    def matrix_mul(values):
        result = values[0]
        for value in values[1:]:
            result = torch.matmul(result, value)
        return result
