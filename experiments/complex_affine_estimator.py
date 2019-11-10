import os

import torch
from torch import optim
from torch.autograd import Variable

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
device = "cpu"

A = torch.tensor([[96.3155, 6.4408, 6.2320, 1],
                  [80.5500, -18.2667, 31.7445, 1],
                  [59.6476, -21.7967, 54.3762, 1],
                  [49.9231, -11.3441, 77.6003, 1],
                  [23.8699, 2.8700, 95.4979, 1],
                  [59.0175, 3.9633, 89.4892, 1],
                  [43.0577, 8.9765, 118.1685, 1],
                  [29.8227, 11.5733, 135.9823, 1],
                  [23.2641, 13.6527, 158.6822, 1],
                  [51.6065, 28.1710, 87.6219, 1],
                  [34.8829, 37.7173, 113.0928, 1],
                  [20.6123, 46.2407, 129.6086, 1],
                  [10.1921, 58.5846, 150.8793, 1],
                  [47.9414, 45.0360, 66.8808, 1],
                  [32.8974, 50.6703, 91.1090, 1],
                  [17.8783, 60.6983, 108.3596, 1],
                  [11.4558, 70.0814, 129.8244, 1],
                  [43.4404, 54.3447, 47.4658, 1],
                  [31.4775, 63.6320, 62.4144, 1],
                  [19.0282, 73.5074, 73.0234, 1],
                  [15.5087, 86.7201, 89.3633, 1]]).to(device)

b = torch.tensor([[8.2529e-02, -1.4945e-01, 5.6315e-01, 1],
                  [6.6335e-02, -1.7623e-01, 5.8878e-01, 1],
                  [4.4146e-02, -1.8020e-01, 6.1253e-01, 1],
                  [3.3096e-02, -1.6868e-01, 6.3744e-01, 1],
                  [2.7584e-03, -1.5482e-01, 6.5776e-01, 1],
                  [4.3642e-02, -1.5129e-01, 6.5138e-01, 1],
                  [2.6643e-02, -1.4592e-01, 6.8187e-01, 1],
                  [1.2250e-02, -1.4310e-01, 7.0081e-01, 1],
                  [3.6873e-03, -1.3518e-01, 7.2599e-01, 1],
                  [3.5133e-02, -1.2599e-01, 6.4969e-01, 1],
                  [1.7186e-02, -1.1599e-01, 6.7677e-01, 1],
                  [1.9427e-03, -1.0687e-01, 6.9442e-01, 1],
                  [-5.4359e-03, -9.3021e-02, 7.1830e-01, 1],
                  [3.0754e-02, -1.0840e-01, 6.2777e-01, 1],
                  [1.4456e-02, -1.0251e-01, 6.5401e-01, 1],
                  [-1.7585e-03, -9.1628e-02, 6.7300e-01, 1],
                  [-1.3311e-02, -8.3420e-02, 6.9725e-01, 1],
                  [2.5966e-02, -9.8290e-02, 6.0717e-01, 1],
                  [1.3304e-02, -8.8364e-02, 6.2325e-01, 1],
                  [-7.6417e-07, -7.7943e-02, 6.3471e-01, 1],
                  [-7.6329e-03, -6.4999e-02, 6.5365e-01, 1]]).to(device)

K = torch.tensor([[388.9018, 0.0000, 76.3069],
                  [0.0000, 388.7123, 195.5198],
                  [0.0000, 0.0000, 1.0000]]).to(device)


def rotation_tensor(t_x, t_y, t_z):
    rot_x = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)
    rot_y = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)
    rot_z = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

    rot_x[0, 0] = 1
    rot_x[0, 1] = 0
    rot_x[0, 2] = 0
    rot_x[0, 3] = 0

    rot_x[1, 0] = 0
    rot_x[1, 1] = t_x.cos()
    rot_x[1, 2] = t_x.sin()
    rot_x[1, 3] = 0

    rot_x[2, 0] = 0
    rot_x[2, 1] = -t_x.sin()
    rot_x[2, 2] = t_x.cos()
    rot_x[2, 3] = 0

    rot_x[3, 0] = 0
    rot_x[3, 1] = 0
    rot_x[3, 2] = 0
    rot_x[3, 3] = 1

    ####

    rot_y[0, 0] = t_y.cos()
    rot_y[0, 1] = 0
    rot_y[0, 2] = -t_y.sin()
    rot_y[0, 3] = 0

    rot_y[1, 0] = 0
    rot_y[1, 1] = 1
    rot_y[1, 2] = 0
    rot_y[1, 3] = 0

    rot_y[2, 0] = t_y.sin()
    rot_y[2, 1] = 0
    rot_y[2, 2] = t_y.cos()
    rot_y[2, 3] = 0

    rot_y[3, 0] = 0
    rot_y[3, 1] = 0
    rot_y[3, 2] = 0
    rot_y[3, 3] = 1

    ####

    rot_z[0, 0] = t_z.cos()
    rot_z[0, 1] = -t_z.sin()
    rot_z[0, 2] = 0
    rot_z[0, 3] = 0

    rot_z[1, 0] = t_z.sin()
    rot_z[1, 1] = t_z.cos()
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

    return rot_x.to(device), rot_y.to(device), rot_z.to(device)


def scale_tensor(s):
    scale_m = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

    scale_m[0, 0] = s
    scale_m[0, 1] = 0
    scale_m[0, 2] = 0
    scale_m[0, 3] = 0

    scale_m[1, 0] = 0
    scale_m[1, 1] = s
    scale_m[1, 2] = 0
    scale_m[1, 3] = 0

    scale_m[2, 0] = 0
    scale_m[2, 1] = 0
    scale_m[2, 2] = s
    scale_m[2, 3] = 0

    scale_m[3, 0] = 0
    scale_m[3, 1] = 0
    scale_m[3, 2] = 0
    scale_m[3, 3] = 1

    return scale_m.to(device)


def translation_tensor(t_x, t_y, t_z):
    translation_m = torch.autograd.Variable(torch.zeros(4, 4), requires_grad=False)

    translation_m[0, 0] = 1
    translation_m[0, 1] = 0
    translation_m[0, 2] = 0
    translation_m[0, 3] = t_x

    translation_m[1, 0] = 0
    translation_m[1, 1] = 1
    translation_m[1, 2] = 0
    translation_m[1, 3] = t_y

    translation_m[2, 0] = 0
    translation_m[2, 1] = 0
    translation_m[2, 2] = 1
    translation_m[2, 3] = t_z

    translation_m[3, 0] = 0
    translation_m[3, 1] = 0
    translation_m[3, 2] = 0
    translation_m[3, 3] = 1

    return translation_m.to(device)


def mul(A, x_1, x_2, x_3, s_m, t_m):
    # print(x_3.type())
    # res = torch.matmul(torch.matmul(torch.matmul(A, x_1), x_2), x_3)
    return torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.matmul(A, x_1), x_2), x_3), s_m), t_m)


def mul_2(x_1, x_2, x_3, s_m, t_m):
    # print(x_3.type())
    # res = torch.matmul(torch.matmul(torch.matmul(A, x_1), x_2), x_3)
    return torch.matmul(torch.matmul(torch.matmul(torch.matmul(x_1, x_2), x_3), s_m), t_m)


x_1 = Variable(torch.rand(1, 1).to(device), requires_grad=True)
x_2 = Variable(torch.rand(1, 1).to(device), requires_grad=True)
x_3 = Variable(torch.rand(1, 1).to(device), requires_grad=True)
s = Variable(torch.rand(1, 1).to(device), requires_grad=True)
t_x = Variable(torch.rand(1, 1).to(device), requires_grad=True)
t_y = Variable(torch.rand(1, 1).to(device), requires_grad=True)
t_z = Variable(torch.rand(1, 1).to(device), requires_grad=True)

m_1, m_2, m_3 = rotation_tensor(x_1, x_2, x_3)
scale_t = scale_tensor(s)
trans_t = translation_tensor(t_x, t_y, t_z)


# print(res)
# Δ = mul(A, m_1, m_2, m_3, scale_t, trans_t) - b
# loss = torch.norm(Δ, p=2)
#
# g_step = 0
#


def traing_approach_1():
    g_step = 0
    print('test_save_all_3.pt')
    x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step = torch.load('test_save_all_3.pt')

    optimizer = optim.SGD([x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z], lr=0.00000002)
    for i in range(100000):
        def closure():
            nonlocal g_step

            m_1, m_2, m_3 = rotation_tensor(x_1, x_2, x_3)
            scale_t = scale_tensor(s_x, s_y, s_z)
            trans_t = translation_tensor(t_x, t_y, t_z)
            Δ = mul(A, m_1, m_2, m_3, scale_t, trans_t) - b
            loss = torch.norm(Δ, p=1)
            loss.backward()

            if g_step % 100 == 0:
                print(g_step, loss)

            if loss < 0.44:
                print(loss)
                print("Whoa")
                torch.save([x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step], 'test_save_all_4.pt')

            # if g_step % 1000 == 0:
            #     torch.save([x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step], 'test_save_all_3.pt')
            # print(m_1)
            # print(m_2)
            # print(m_3)
            # print(scale_t)
            # print(trans_t)
            # f_r = mul_2(m_1, m_2, m_3, scale_t, trans_t)
            # print("F")
            # print(loss, f_r)

            g_step += 1

            return loss

        optimizer.step(closure)


def traing_approach_2():
    x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step = torch.load('test_save_all_3.pt')

    stop_loss = 1e-3
    step_size = stop_loss / 150.0

    for i in range(300001):
        m_1, m_2, m_3 = rotation_tensor(x_1, x_2, x_3)
        scale_t = scale_tensor(s_x, s_y, s_z)
        trans_t = translation_tensor(t_x, t_y, t_z)
        Δ = mul(A, m_1, m_2, m_3, scale_t, trans_t) - b
        loss = torch.norm(Δ, p=2)
        loss.backward()
        if g_step % 100 == 0:
            print(g_step, loss)

        if g_step % 10000 == 0:
            print(m_1)
            print(m_2)
            print(m_3)
            print(scale_t)
            print(trans_t)
            f_r = mul_2(m_1, m_2, m_3, scale_t, trans_t)
            print("F")
            print(loss, f_r)
            # torch.save([x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step], 'test_save_all_2.pt')

        for par in [x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z]:
            par.data -= step_size * par.grad.data  # step
            par.grad.data.zero_()

        g_step += 1


def traing_approach_3():
    g_step = 0
    print('test_save_all_5.pt')

    if os.path.exists('af_est_state_old.pt.pt'):
        checkpoint = torch.load('af_est_state_old.pt.pt')

        x_1 = checkpoint['r_x']
        x_2 = checkpoint['r_y']
        x_3 = checkpoint['r_z']
        s = checkpoint['s']
        t_x = checkpoint['t_x']
        t_y = checkpoint['t_y']
        t_z = checkpoint['t_z']

        g_step = checkpoint['g_step']
        loss = checkpoint['loss']

    optimizer = optim.SGD([x_1, x_2, x_3, s, t_x, t_y, t_z], lr=0.0000000000001)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for i in range(10000000):
        def closure():
            nonlocal g_step

            m_1, m_2, m_3 = rotation_tensor(x_1, x_2, x_3)
            scale_t = scale_tensor(s)
            trans_t = translation_tensor(t_x, t_y, t_z)
            y_ = mul(A, m_1, m_2, m_3, scale_t, trans_t)
            Δ_1 = y_ - b
            loss_1 = torch.norm(Δ_1, p=2)
            Δ_2 = torch.matmul(y_[..., :-1], K) - torch.matmul(b[..., :-1], K)
            loss_2 = torch.norm(Δ_2, p=2)
            loss = loss_1 + loss_2
            loss.backward()

            if g_step % 100 == 0:
                print(g_step, loss)

            # if loss < 0.44:
            #     print(loss)
            #     print("Whoa")
            #     torch.save([x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step], 'test_save_all_4.pt')

            # if g_step % 10000 == 0:
            #     torch.save([x_1, x_2, x_3, s, t_x, t_y, t_z, loss, g_step], 'test_save_all_7.pt')
            #     print(m_1)
            #     print(m_2)
            #     print(m_3)
            #     print(scale_t)
            #     print(trans_t)
            #     f_r = mul_2(m_1, m_2, m_3, scale_t, trans_t)
            #     print("Final_loss_and_matrix")
            #     print(loss, f_r)
            if g_step % 10000 == 0:
                torch.save({
                    'g_step': g_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'r_x': x_1,
                    'r_y': x_2,
                    'r_z': x_3,
                    's': s,
                    't_x': t_x,
                    't_y': t_y,
                    't_z': t_z,
                    'loss': loss
                }, 'af_est_state_old.pt.pt')
                print("saved")

            g_step += 1

            return loss

        optimizer.step(closure)


def traing_approach_4():
    g_step = 0
    # print('test_save_all_5.pt')
    # x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step = torch.load('test_save_all_5.pt')

    stop_loss = 1e-3
    step_size = stop_loss / 4100.0

    for i in range(100000):
        m_1, m_2, m_3 = rotation_tensor(x_1, x_2, x_3)
        scale_t = scale_tensor(s)
        trans_t = translation_tensor(t_x, t_y, t_z)
        y_ = mul(A, m_1, m_2, m_3, scale_t, trans_t)
        Δ_1 = y_ - b
        loss_1 = torch.norm(Δ_1, p=2)
        Δ_2 = torch.matmul(y_[..., :-1], K) - torch.matmul(b[..., :-1], K)
        loss_2 = torch.norm(Δ_2, p=2)
        loss = loss_1 + loss_2
        loss.backward()
        if g_step % 100 == 0:
            print(g_step, loss)
        # if loss < 0.44:
        #     print(loss)
        #     print("Whoa")
        #     torch.save([x_1, x_2, x_3, s_x, s_y, s_z, t_x, t_y, t_z, loss, g_step], 'test_save_all_4.pt')
        if g_step % 1000 == 0:
            torch.save([x_1, x_2, x_3, s, t_x, t_y, t_z, loss, g_step], 'test_save_all_5.pt')
            print(m_1)
            print(m_2)
            print(m_3)
            print(scale_t)
            print(trans_t)
            f_r = mul_2(m_1, m_2, m_3, scale_t, trans_t)
            print("Final_loss_and_matrix")
            print(loss, f_r)

        for par in [x_1, x_2, x_3, s, t_x, t_y, t_z]:
            par.data -= step_size * par.grad.data  # step
            par.grad.data.zero_()

        g_step += 1


traing_approach_3()
