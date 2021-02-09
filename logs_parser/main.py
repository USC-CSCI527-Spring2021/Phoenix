"""Define `view` command"""
from __future__ import absolute_import

import logging
import sys
# from io import load_mjlog
from parser import parse_mjlog
from viewer import print_node
import pandas as pd
import xml.etree.ElementTree as ET

_LG = logging.getLogger(__name__)


def _print_meta(meta_data):
    for tag in ['SHUFFLE', 'GO', 'UN', 'TAIKYOKU']:
        if tag in meta_data:
            print_node(tag, meta_data[tag])


def _print_round(round_data):
    _LG.info('=' * 40)
    for node in round_data:
        print_node(node['tag'], node['data'])


def print_game(xml_str):
    node = ET.fromstring(xml_str)
    data = parse_mjlog(node)
    _print_meta(data['meta'])
    rounds = data['rounds']
    for round_data in rounds:
        _print_round(round_data)


def _init_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    format_ = (
        '%(message)s' if not debug else
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
    )
    logging.basicConfig(level=level, format=format_, stream=sys.stdout)


if __name__ == "__main__":
    _init_logging()
    df = pd.read_csv("2021.csv")
    print_game(df["log_content"][0])
