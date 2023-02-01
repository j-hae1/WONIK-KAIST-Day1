import os
import pickle

import torch

from torch.utils.data import DataLoader, TensorDataset
from src.utils.load_data import load_data
from src.model.get_model import get_multistep_linear_model
from src.utils.data_preprocess import get_data

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main(train_config):
    model_name = train_config['model_name']
    state_dim = train_config['state_dim']
    action_dim = train_config['action_dim']
    state_order = train_config['state_order']
    action_order = train_config['action_order']
    H = train_config['H']
    alpha = train_config['alpha']
    state_scaler = train_config['state_scaler']
    action_scaler = train_config['action_scaler']
    EPOCHS = train_config['EPOCHS']
    BS = train_config['BS']
    lr = train_config['lr']
    train_data_path = train_config['train_data_path']
    test_data_path = train_config['test_data_path']
    TEST_EVERY = 25
    SAVE_EVERY = 25

    # Prepare Model and Dataset
    m = get_multistep_linear_model(model_name, state_dim, action_dim, state_order, action_order).to(DEVICE)

    train_states, train_actions = load_data(paths=train_data_path,
                                            scaling=True,
                                            state_scaler=state_scaler,
                                            action_scaler=action_scaler,
                                            preprocess=True,
                                            history_x=state_order,
                                            history_u=action_order)

    # Set te minimum and maximum temperature as 20 and 420

    test_states, test_actions = load_data(paths=test_data_path,
                                          scaling=True,
                                          state_scaler=state_scaler,
                                          action_scaler=action_scaler,
                                          preprocess=True,
                                          history_x=state_order,
                                          history_u=action_order)

    train_history_xs, train_history_us, train_us, train_ys = get_data(states=train_states,
                                                                      actions=train_actions,
                                                                      rollout_window=H,
                                                                      history_x_window=state_order,
                                                                      history_u_window=action_order,
                                                                      num_glass_tc=state_dim,
                                                                      num_control_tc=action_dim,
                                                                      device=DEVICE)
    train_history_xs = train_history_xs[0].transpose(1, 2)
    train_history_us = train_history_us.transpose(1, 2)
    train_us = train_us.transpose(1, 2)
    train_ys = train_ys[0].transpose(1, 2)

    test_history_xs, test_history_us, test_us, test_ys = get_data(states=test_states,
                                                                  actions=test_actions,
                                                                  rollout_window=H,
                                                                  history_x_window=state_order,
                                                                  history_u_window=action_order,
                                                                  num_glass_tc=state_dim,
                                                                  num_control_tc=action_dim,
                                                                  device=DEVICE)
    test_history_xs = test_history_xs[0].transpose(1, 2)
    test_history_us = test_history_us.transpose(1, 2)
    test_us = test_us.transpose(1, 2)
    test_ys = test_ys[0].transpose(1, 2)

    # Training Route Setting
    criteria = torch.nn.SmoothL1Loss()
    train_ds = TensorDataset(train_history_xs, train_history_us, train_us, train_ys)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    test_criteria = torch.nn.SmoothL1Loss()

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    iters = len(train_loader)
    num_params = sum(p.numel() for p in m.parameters())

    num_updates = 0
    best_test_loss = float('inf')

    model_save_path = 'saved_model/{}'.format(model_name)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    with open('{}/train_config_{}_{}.txt'.format(model_save_path, state_order, action_order), 'wb') as f:
        pickle.dump(train_config, f)
    # Start Training
    print('Model Training Start, State Order: {}, Action Order: {}'.format(model_name, state_order, action_order))
    for ep in range(EPOCHS):
        if ep % 100 == 0:
            print("Epoch [{}] / [{}]".format(ep+1, EPOCHS))
        for i, (x0, u0, u, y) in enumerate(train_loader):
            opt.zero_grad()
            y_predicted = m.rollout(x0, u0, u)
            loss_prediction = criteria(y_predicted, y)
            loss_regularizer = alpha * (sum(torch.norm(param, p=1) for param in m.parameters())) / num_params
            # loss_regularizer = alpha * (torch.mean(torch.abs(m.A.data)) + torch.mean(torch.abs(m.B.data)))
            loss = loss_prediction + loss_regularizer
            loss.backward()
            opt.step()
            # scheduler.step(ep + i / iters)
            num_updates += 1
            log_dict = {}
            log_dict['train_loss_prediction'] = loss_prediction.item()
            log_dict['train_loss_regularizer'] = loss_regularizer.item()
            log_dict['train_loss'] = loss.item()
            log_dict['lr'] = opt.param_groups[0]['lr']
            if num_updates % TEST_EVERY == 0:
                with torch.no_grad():
                    test_predicted_y = m.rollout(test_history_xs, test_history_us, test_us)
                    test_loss_prediction = test_criteria(test_predicted_y, test_ys)
                    test_loss_regularizer = alpha * (sum(torch.norm(param, p=1) for param in m.parameters())) / num_params
                    test_loss = test_loss_prediction + test_loss_regularizer
                    log_dict['test_loss_prediction'] = test_loss_prediction.item()
                    log_dict['test_loss_regularizer'] = test_loss_regularizer.item()
                    log_dict['test_loss'] = test_loss.item()
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(m.state_dict(), '{}/best_model_{}_{}.pt'.format(model_save_path, state_order, action_order))
                    print('Best model saved at iteration {}, loss: {}'.format(num_updates, best_test_loss))
                log_dict['best_test_loss'] = best_test_loss
            if num_updates % SAVE_EVERY == 0:
                torch.save(m.state_dict(), '{}/curr_model_{}_{}.pt'.format(model_save_path, state_order, action_order))


if __name__ == '__main__':
    state_orders = [1, 5, 10]
    action_orders = [5, 10, 20, 50]
    for state_order in state_orders:
        for action_order in action_orders:
            train_config = {
                'model_name': 'multistep_linear_res2',
                'state_dim': 140,
                'action_dim': 40,
                'state_order': state_order,
                'action_order': action_order,
                'H': 100,
                'alpha': 0.0,
                'state_scaler': (20.0, 420.0),
                'action_scaler': (20.0, 420.0),
                'EPOCHS': 100,
                'BS': 128,
                'lr': 1e-6,
                'train_data_path': ['data/train_data/data_01.csv',
                                    'data/train_data/data_02.csv',
                                    'data/train_data/data_03.csv',
                                    'data/train_data/data_04.csv',
                                    'data/train_data/data_05.csv'],
                'test_data_path': ['data/val_data/data_01.csv',
                                   'data/val_data/data_02.csv',
                                   'data/val_data/data_03.csv',
                                   'data/val_data/data_04.csv']
            }
            main(train_config)
