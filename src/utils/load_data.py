import re
from typing import Union, List

import pandas as pd
import torch


def load_data(paths: Union[str, List[str]],
              scaling: bool = True,
              state_scaler: tuple = None,
              action_scaler: tuple = None,
              action_ws: bool = True,
              preprocess: bool = True,
              history_x: int = 1,
              history_u: int = 1,
              target_idx: List = [],
              device: str = 'cpu'):
    if isinstance(paths, str):
        paths = [paths]
    states = []
    actions = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        l = df.columns.to_list()
        action_rex = 'Z.+_WS$' if action_ws else 'Z.+_TC$'
        action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
        glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]
        if len(target_idx) > 0:
            glass_cols = [glass_cols[idx] for idx in target_idx]
        control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
        state_cols = glass_cols + control_cols
        state_df = df[state_cols]
        state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
        action_df = df[action_cols]
        action_nan_cols = action_df.columns[action_df.isna().any()].tolist()

        assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."
        assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

        state = torch.tensor(state_df.to_numpy()).float().to(device)
        action = torch.tensor(action_df.to_numpy()).float().to(device)

        step_col = ['Step_Name']
        step_df = df[step_col].values.tolist()
        step_df_length = len(step_df)

        if preprocess:
            for i in range(step_df_length):
                if step_df[i][0] == '375H' or step_df[i][0] == '200H' or step_df[i][0] == '400H':
                    first = i
                    break
            for i in range(step_df_length):
                if step_df[i][0] == 'SLOW_COOL':
                    state = state[first - history_x + 1:i + 1, :]
                    action = action[first - history_u + 1:i, :]
                    break
        states.append(state)
        actions.append(action)
    if scaling:
        for idx in range(len(states)):
            states[idx] = (states[idx] - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
            actions[idx] = (actions[idx] - action_scaler[0]) / (action_scaler[1] - action_scaler[0])

    return states, actions
