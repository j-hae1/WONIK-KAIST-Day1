import torch

from src.model.LinearStateSpaceModels import MultiLinearResSSM1, MultiLinearResSSM2
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_multi_linear_residual_model1(state_dim, action_dim, state_order, action_order):
    m = MultiLinearResSSM1(state_dim, action_dim, state_order, action_order)
    return m


def get_multi_linear_residual_model2(state_dim, action_dim, state_order, action_order):
    m = MultiLinearResSSM2(state_dim, action_dim, state_order, action_order)
    return m


name_model_dict = {
    'multistep_linear_res1': get_multi_linear_residual_model1,
    'multistep_linear_res2': get_multi_linear_residual_model2,
}


def get_multistep_linear_model(model_name, state_dim, action_dim, state_order, action_order, saved_model_path=None):
    m = name_model_dict[model_name](state_dim, action_dim, state_order, action_order)
    if saved_model_path is not None:
        m.load_state_dict(torch.load(saved_model_path, map_location=device))
    return m
