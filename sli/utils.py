import os
import pandas as pd


def check_path(*elements):
    """checks and creates path"""
    res = elements[0]
    for i in range(1, len(elements)):
        res = os.path.join(res, elements[i])
    if not os.path.exists(res):
        os.makedirs(res)
    return res


def files_langs_to_csv(files_list, path, csv_name):
    if len(files_list) > 0:
        df = pd.DataFrame.from_dict(files_list)
        df.to_csv(os.path.join(path, csv_name), index=False)
        return os.path.join(path, csv_name)
    else:
        return None
