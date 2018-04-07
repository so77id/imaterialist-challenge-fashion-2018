from __future__ import absolute_import, division, print_function

import json
from collections import namedtuple


def get_config_from_json(config_file):
    """
    Get the config from a json file
    :param config_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(config_file, 'r') as content_file:
        content = content_file.read()
    return json.loads(
        content,
        object_hook=lambda d: namedtuple(
            'Configuration', d.keys())(* d.values()),
    )


def process_config(config_file):
    config = get_config_from_json(config_file)
    # config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
    # config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
    return config
