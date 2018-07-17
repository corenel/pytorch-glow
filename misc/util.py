import os
import json


def load_profile(setting):
    profile_filepath = os.path.join(setting.profile_dir, '{}.json'.format(setting.profile))
    if os.path.exists(profile_filepath):
        with open(profile_filepath) as f:
            conf = json.load(f)
            for key in list(conf.keys()):
                setattr(setting, key, conf[key])
    return setting
