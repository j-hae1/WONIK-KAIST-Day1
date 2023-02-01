import torch
import torch.nn as nn


class MultiLinearResSSM1(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 state_order: int,
                 action_order: int):
        super(MultiLinearResSSM1, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_order = state_order
        self.action_order = action_order
        # In Multi-step Linear model, we initialize as zeros
        self.A = nn.Parameter(torch.zeros(state_order, state_dim, state_dim))
        self.B = nn.Parameter(torch.zeros(action_order, action_dim, state_dim))

    def forward(self, x, u):
        """
        :param x: B x state_order x state_dim
        :param u: B x action_order x action_dim
        :return:
        """
        ar_terms = torch.sum(torch.einsum('box, oxy -> boy', x, self.A), dim=1)
        exo_terms = torch.sum(torch.einsum('box, oxy -> boy', u, self.B), dim=1)
        return ar_terms + exo_terms + x[:, -1, :]

    def rollout(self, x0, u0, us):
        """
        :param x0: B x state_order x state_dim
        :param u0: B x (action_order-1) x action_dim
        :param us: B x H x action_dim
        :return:
        """
        xs = []
        if u0 is not None:
            u_cat = torch.cat([u0, us], dim=1)
        else:
            u_cat = us
        for i in range(us.shape[1]):
            x = self.forward(x0, u_cat[:, i:i + self.action_order]).unsqueeze(dim=1)
            xs.append(x)
            x0 = torch.cat([x0[:, 1:, :], x], dim=1)
        return torch.cat(xs, dim=1)

    def multi_step_prediction(self, x0, u0, us):
        """
        :param x0: state_order x state_dim
        :param u0: (action_order-1) x action_dim
        :param us: H x action_dim
        :return:
        """
        x0 = x0.unsqueeze(dim=0)
        u0 = u0.unsqueeze(dim=0)
        us = us.unsqueeze(dim=0)
        xs = self.rollout(x0, u0, us)
        return xs.squeeze(dim=0)


class MultiLinearResSSM2(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 state_order: int,
                 action_order: int):
        super(MultiLinearResSSM2, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_order = state_order
        self.action_order = action_order
        # In Multi-step Linear model, we initialize as zeros
        self.A = nn.Parameter(torch.zeros(state_order, state_dim, state_dim))
        self.B = nn.Parameter(torch.zeros(action_order, action_dim, state_dim))
        self.C = nn.Parameter(torch.zeros(1, state_dim))

    def forward(self, x, u):
        """
        :param x: B x state_order x state_dim
        :param u: B x action_order x action_dim
        :return:
        """
        ar_terms = torch.einsum('box, oxy -> by', x, self.A)
        exo_terms = torch.einsum('box, oxy -> by', u, self.B)
        return ar_terms + exo_terms + x[:, -1, :] + self.C

    def rollout(self, x0, u0, us):
        """
        :param x0: B x state_order x state_dim
        :param u0: B x (action_order-1) x action_dim
        :param us: B x H x action_dim
        :return:
        """
        xs = []
        if u0 is not None:
            u_cat = torch.cat([u0, us], dim=1)
        else:
            u_cat = us
        for i in range(us.shape[1]):
            x = self.forward(x0, u_cat[:, i:i + self.action_order]).unsqueeze(dim=1)
            xs.append(x)
            x0 = torch.cat([x0[:, 1:, :], x], dim=1)
        return torch.cat(xs, dim=1)

    def multi_step_prediction(self, x0, u0, us):
        """
        :param x0: state_order x state_dim
        :param u0: (action_order-1) x action_dim
        :param us: H x action_dim
        :return:
        """
        x0 = x0.unsqueeze(dim=0)
        u0 = u0.unsqueeze(dim=0)
        us = us.unsqueeze(dim=0)
        xs = self.rollout(x0, u0, us)
        return xs.squeeze(dim=0)
