import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model.get_model import get_multistep_linear_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def main(evaluate_config):
    model_name = evaluate_config['model_name']
    state_dim = evaluate_config['state_dim']
    action_dim = evaluate_config['action_dim']
    state_order = evaluate_config['state_order']
    action_order = evaluate_config['action_order']
    state_scaler = evaluate_config['state_scaler']
    action_scaler = evaluate_config['action_scaler']
    test_data_paths = evaluate_config['test_data_paths']
    rollout_window = evaluate_config['rollout_window']
    saved_model_path = 'saved_model/{}/best_model_{}_{}.pt'.format(model_name, state_order, action_order)
    m = get_multistep_linear_model(model_name, state_dim, action_dim, state_order, action_order, saved_model_path)
    m.eval().to(device)
    test_states, test_actions = load_data(paths=test_data_paths,
                                          scaling=True,
                                          state_scaler=state_scaler,
                                          action_scaler=action_scaler,
                                          preprocess=True,
                                          history_x=state_order,
                                          history_u=action_order)

    history_xs, history_us, us, ys = get_data(states=test_states,
                                              actions=test_actions,
                                              rollout_window=rollout_window,
                                              history_x_window=state_order,
                                              history_u_window=action_order,
                                              num_glass_tc=140,
                                              num_control_tc=40,
                                              device=device)
    history_xs = history_xs[0].transpose(1, 2)
    ys = ys[0].transpose(1, 2)
    history_us = history_us.transpose(1, 2)
    us = us.transpose(1, 2)

    def loss_fn(y_predicted, y):
        idx_list = []
        for i in range(7):
            for j in range(5):
                idx_list.append(20 * i + j)
        amplitude = 10.0
        criteria_ = torch.nn.SmoothL1Loss(reduction='none')
        loss = criteria_(y_predicted, y)
        loss[..., idx_list] = amplitude * loss[..., idx_list]
        return loss.mean()

    with torch.no_grad():
        predicted_ys = m.rollout(history_xs, history_us, us)
        ys = ys * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
        predicted_ys = predicted_ys * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
        # loss_fn = torch.nn.MSELoss()
        loss = loss_fn(ys, predicted_ys)
        print('({}, {}), Loss: {}'.format(state_order, action_order, loss))
        ys = ys.cpu().detach().numpy()
        predicted_ys = predicted_ys.cpu().detach().numpy()
    return ys, predicted_ys, loss.item()


if __name__ == '__main__':
    # Specify which model you want to evaluate
    model_name = 'multistep_linear_res2'
    state_orders = [1, 5, 10]
    action_orders = [5, 10, 20, 50]
    results = []
    loss_lists = []
    for state_order in state_orders:
        for action_order in action_orders:
            evaluate_config = {
                'model_name': model_name,
                'state_dim': 140,
                'action_dim': 40,
                'state_order': state_order,
                'action_order': action_order,
                'state_scaler': (20.0, 420.0),
                'action_scaler': (20.0, 420.0),
                'test_data_paths': ['data/test_data/data_01.csv'],
                'rollout_window': 800
            }
            res = main(evaluate_config)
            fig, axes = plt.subplots(1, 2, figsize=(10, 7))
            axes_flatten = axes.flatten()
            axes_flatten[0].plot(res[0][0][:, :140])
            axes_flatten[0].set_title('True')
            axes_flatten[0].set_ylim([140, 400])
            axes_flatten[1].plot(res[1][0][:, :140])
            axes_flatten[1].set_title('Predicted')
            axes_flatten[1].set_ylim([140, 400])
            fig.suptitle('Model: ({}, {}), Loss: {}'.format(state_order, action_order, res[2]))
            fig.show()
            overshoot = np.max(res[0][0][:, :140]) - 375
            anneal_dev = np.max(res[0][0][:, :140], axis=1)[-180:] - np.min(res[0][0][:, :140], axis=1)[-180:]
            print('Overshoot: {}'.format(overshoot))
            print('Maximum Anneal Deviation: {}'.format(np.max(anneal_dev)))
            overshoot = np.max(res[1][0][:, :140]) - 375
            anneal_dev = np.max(res[1][0][:, :140], axis=1)[-180:] - np.min(res[1][0][:, :140], axis=1)[-180:]
            print('Overshoot: {}'.format(overshoot))
            print('Maximum Anneal Deviation: {}'.format(np.max(anneal_dev)))
            # results.append(res[1])
            loss_lists.append(res[2])
    ys = res[0]
    nrows = 3
    ncols = 3
    loss_lists = np.array(loss_lists)
    arg_loss_lists = np.argsort(loss_lists)
    num_plots = 5
    batch_idx = np.random.randint(ys.shape[0], size=nrows * ncols)
    tc_idx = np.random.randint(ys.shape[2], size=nrows * ncols)
    for j in range(num_plots):
        print('{}-th Model: ({}, {})'.format(j + 1, state_orders[arg_loss_lists[j] // len(action_orders)],
                                          action_orders[arg_loss_lists[j] % len(action_orders)]))
    loss_lists = np.reshape(loss_lists, (len(state_orders), len(action_orders)))
    plt.imshow(loss_lists)
    plt.colorbar()
    plt.xlabel('Action Order')
    plt.ylabel('State Order')
    plt.show()
