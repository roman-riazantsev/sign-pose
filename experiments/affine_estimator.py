import os

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class AffineEstimator:
    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.lr = config['lr']
        self.log_rate = config['log_rate']
        self.save_path = config['save_path']
        self.save_rate = config['save_rate']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.r_x = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.r_y = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.r_z = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.s = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.t_x = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.t_y = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.t_z = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)

        self.t_x2 = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.t_y2 = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)
        self.t_z2 = Variable(torch.rand(1, 1).to(self.device), requires_grad=True).to(self.device)

        self.g_step = 1
        self.loss = 0

        self.load_state()

        self.writer = SummaryWriter('experiments/affine_log_2   cd')

    def estimate_transformation(self, A, b_true, K):
        for i in range(self.n_steps):
            def closure():
                rx_t, ry_t, rz_t = self.get_rotation_tensors()
                s_t = self.get_scale_tensor()
                t_t = self.get_translation_tensor()
                t_t2 = self.get_translation_tensor2()

                first_part = torch.matmul(-t_t2, rx_t)
                second_part = torch.matmul(rz_t, t_t2)

                b_pred = self.mul_6(A, first_part, second_part, rz_t, s_t, t_t)

                d_1 = b_pred - b_true
                loss_1 = torch.norm(d_1, p=2)

                d_2 = torch.matmul(b_pred[..., :-1], K) - torch.matmul(b_true[..., :-1], K)
                loss_2 = torch.norm(d_2, p=2)

                self.loss = loss_1 + loss_2
                self.loss.backward()

                if self.g_step % self.log_rate == 0:
                    print(f"step: {self.g_step}, loss: {self.loss}")

                # if self.g_step % self.save_rate == 0:
                #     self.save_state()
                #     # print("matrix:", self.mul_5(rx_t, ry_t, rz_t, s_t, t_t))

                if self.g_step % 10000 == 0:
                    first_part = torch.matmul(-t_t2, rx_t)
                    second_part = torch.matmul(rz_t, t_t2)
                    print("matrix:", self.mul_5(first_part, ry_t, second_part, s_t, t_t))
                    self.writer.add_scalar('training loss',
                                           self.loss,
                                           self.g_step)
                    self.writer.add_histogram('transform_matrix:',
                                              self.mul_5(first_part, ry_t, second_part, s_t, t_t),
                                              self.g_step)

                self.g_step += 1
                return self.loss

            self.optimizer.step(closure)

    def load_state(self):
        print(self.save_path)
        # if os.path.exists(self.save_path):
        #     checkpoint = torch.load(self.save_path)
        #
        #     self.r_x = checkpoint['r_x']
        #     self.r_y = checkpoint['r_y']
        #     self.r_z = checkpoint['r_z']
        #     self.s = checkpoint['s']
        #     self.t_x = checkpoint['t_x']
        #     self.t_y = checkpoint['t_y']
        #     self.t_z = checkpoint['t_z']
        #
        #     self.g_step = checkpoint['g_step']
        #     self.loss = checkpoint['loss']
        #
        #     self.trainable_parameters = [self.r_x, self.r_y, self.r_z, self.s, self.t_x, self.t_y, self.t_z]
        #
        #     self.optimizer = optim.SGD(self.trainable_parameters, lr=self.lr)
        #
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #
        # else:
        self.trainable_parameters = [self.r_x, self.r_y, self.r_z, self.s, self.t_x, self.t_y, self.t_z,
                                     self.t_x2, self.t_y2, self.t_z2]

        self.optimizer = optim.SGD(self.trainable_parameters, lr=self.lr)


    def save_state(self):
        print(90 < self.loss < 100)
        if 90 < self.loss < 100:
            torch.save({
                'g_step': self.g_step + 1,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'r_x': self.r_x,
                'r_y': self.r_y,
                'r_z': self.r_z,
                's': self.s,
                't_x': self.t_x,
                't_y': self.t_y,
                't_z': self.t_z,
                'loss': self.loss
            }, self.save_path)
            print(self.save_path)

            print(f'Parameters saved at step {self.g_step}; loss {self.loss}.')
        elif 50 < self.loss < 90:
            torch.save({
                'g_step': self.g_step + 1,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'r_x': self.r_x,
                'r_y': self.r_y,
                'r_z': self.r_z,
                's': self.s,
                't_x': self.t_x,
                't_y': self.t_y,
                't_z': self.t_z,
                'loss': self.loss
            }, 'experiments/l90/af_est_state_old.pt.pt')
            print('experiments/l90/af_est_state_old.pt.pt')

            print(f'Parameters saved at step {self.g_step}; loss {self.loss}.')
        elif 10 < self.loss < 50:
            torch.save({
                'g_step': self.g_step + 1,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'r_x': self.r_x,
                'r_y': self.r_y,
                'r_z': self.r_z,
                's': self.s,
                't_x': self.t_x,
                't_y': self.t_y,
                't_z': self.t_z,
                'loss': self.loss
            }, 'experiments/l50/af_est_state_old.pt.pt')
            print('experiments/l50/af_est_state_old.pt.pt')

            print(f'Parameters saved at step {self.g_step}; loss {self.loss}.')
        elif self.loss < 10:
            torch.save({
                'g_step': self.g_step + 1,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'r_x': self.r_x,
                'r_y': self.r_y,
                'r_z': self.r_z,
                's': self.s,
                't_x': self.t_x,
                't_y': self.t_y,
                't_z': self.t_z,
                'loss': self.loss
            }, 'experiments/l10/af_est_state_old.pt.pt')
            print('experiments/l10/af_est_state_old.pt.pt')

            print(f'Parameters saved at step {self.g_step}; loss {self.loss}.')


    def get_scale_tensor(self):
        scale_m = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

        scale_m[0, 0] = self.s
        scale_m[0, 1] = 0
        scale_m[0, 2] = 0
        scale_m[0, 3] = 0

        scale_m[1, 0] = 0
        scale_m[1, 1] = self.s
        scale_m[1, 2] = 0
        scale_m[1, 3] = 0

        scale_m[2, 0] = 0
        scale_m[2, 1] = 0
        scale_m[2, 2] = self.s
        scale_m[2, 3] = 0

        scale_m[3, 0] = 0
        scale_m[3, 1] = 0
        scale_m[3, 2] = 0
        scale_m[3, 3] = 1

        return scale_m.to(self.device)


    def get_translation_tensor(self):
        translation_m = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

        translation_m[0, 0] = 1
        translation_m[0, 1] = 0
        translation_m[0, 2] = 0
        translation_m[0, 3] = self.t_x

        translation_m[1, 0] = 0
        translation_m[1, 1] = 1
        translation_m[1, 2] = 0
        translation_m[1, 3] = self.t_y

        translation_m[2, 0] = 0
        translation_m[2, 1] = 0
        translation_m[2, 2] = 1
        translation_m[2, 3] = self.t_z

        translation_m[3, 0] = 0
        translation_m[3, 1] = 0
        translation_m[3, 2] = 0
        translation_m[3, 3] = 1

        return translation_m.to(self.device)


    def get_rotation_tensors(self):
        rot_x = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)
        rot_y = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)
        rot_z = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

        rot_x[0, 0] = 1
        rot_x[0, 1] = 0
        rot_x[0, 2] = 0
        rot_x[0, 3] = 0

        rot_x[1, 0] = 0
        rot_x[1, 1] = self.r_x.cos()
        rot_x[1, 2] = self.r_x.sin()
        rot_x[1, 3] = 0

        rot_x[2, 0] = 0
        rot_x[2, 1] = -self.r_x.sin()
        rot_x[2, 2] = self.r_x.cos()
        rot_x[2, 3] = 0

        rot_x[3, 0] = 0
        rot_x[3, 1] = 0
        rot_x[3, 2] = 0
        rot_x[3, 3] = 1

        ####

        rot_y[0, 0] = self.r_y.cos()
        rot_y[0, 1] = 0
        rot_y[0, 2] = -self.r_y.sin()
        rot_y[0, 3] = 0

        rot_y[1, 0] = 0
        rot_y[1, 1] = 1
        rot_y[1, 2] = 0
        rot_y[1, 3] = 0

        rot_y[2, 0] = self.r_y.sin()
        rot_y[2, 1] = 0
        rot_y[2, 2] = self.r_y.cos()
        rot_y[2, 3] = 0

        rot_y[3, 0] = 0
        rot_y[3, 1] = 0
        rot_y[3, 2] = 0
        rot_y[3, 3] = 1

        ####

        rot_z[0, 0] = self.r_z.cos()
        rot_z[0, 1] = -self.r_z.sin()
        rot_z[0, 2] = 0
        rot_z[0, 3] = 0

        rot_z[1, 0] = self.r_z.sin()
        rot_z[1, 1] = self.r_z.cos()
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


    def mul_5(self, x_1, x_2, x_3, s_m, t_m):
        return torch.matmul(torch.matmul(torch.matmul(torch.matmul(x_1, x_2), x_3), s_m), t_m)


    def mul_6(self, A, x_1, x_2, x_3, s_m, t_m):
        return torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.matmul(A, x_1), x_2), x_3), s_m), t_m)


    def get_translation_tensor2(self):
        translation_m = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

        translation_m[0, 0] = 1
        translation_m[0, 1] = 0
        translation_m[0, 2] = 0
        translation_m[0, 3] = self.t_x2

        translation_m[1, 0] = 0
        translation_m[1, 1] = 1
        translation_m[1, 2] = 0
        translation_m[1, 3] = self.t_y2

        translation_m[2, 0] = 0
        translation_m[2, 1] = 0
        translation_m[2, 2] = 1
        translation_m[2, 3] = self.t_z2

        translation_m[3, 0] = 0
        translation_m[3, 1] = 0
        translation_m[3, 2] = 0
        translation_m[3, 3] = 1

        return translation_m.to(self.device)
