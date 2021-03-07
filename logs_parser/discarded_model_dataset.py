#!/usr/bin/env python
# coding: utf-8

# In[1]:


# script for json file


# In[2]:


import copy
import xml.etree.ElementTree as ET

import apache_beam as beam

from .parser import parse_mjlog


def get_round_info(dataset,
                   draw_tile_list,
                   hands_list,
                   discarded_tiles_pool_list,
                   four_players_open_hands_list,
                   discarded_tile
                   ):
    discarded_tiles_pool = []

    # acquire init scores and states
    init_data = dataset[0]["data"]
    four_player_hands = init_data["hands"]

    dealer = init_data["oya"]

    four_players_open_hands = [[], [], [], []]

    last_act = 'init_act'

    # acquire actions and states after each action
    for k in range(len(dataset)):
        action = dataset[k]
        act = action["tag"]

        if act == 'REACH' or act == 'DORA':
            continue

        elif act == "CALL":
            caller = action["data"]["caller"]
            callee = action["data"]["callee"]
            tiles = action["data"]["mentsu"]

            if (action["data"]["call_type"] == 'Pon' or action["data"]["call_type"] == 'Chi'):

                for tile in tiles:
                    if tile not in four_player_hands[caller]:
                        four_player_hands[caller].append(tile)
                        draw_tile_list.append(tile)
                        discarded_tiles_pool.remove(tile)

                    if tile not in four_players_open_hands[caller]:
                        four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])
                        four_players_open_hands[caller].append(tile)

                hands_list.append(copy.copy(four_player_hands[caller]))
                four_players_open_hands_list.append(copy.copy(four_players_open_hands))

            elif action["data"]["call_type"] == 'MinKan':

                for tile in tiles:
                    if tile not in four_player_hands[caller]:
                        four_player_hands[caller].append(tile)

                        discarded_tiles_pool.remove(tile)

                    if tile not in four_players_open_hands[caller]:
                        four_players_open_hands[caller].append(tile)

                four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])


            elif (action["data"]["call_type"] == 'AnKan' or action["data"]["call_type"] == 'KaKan'):

                # if ankan or kakan, then the last act must be draw. pop the last four_players_open_hands
                # from list to avoid duplicate
                hands_list.pop()
                draw_tile_list.pop()
                four_players_open_hands_list.pop()

                player_open_tile = four_players_open_hands[caller]

                for tile in tiles:
                    if tile not in player_open_tile:
                        player_open_tile.append(tile)

                four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])

        # deal with draw and discard operations
        elif act in ["DRAW", "DISCARD"]:
            pl_id = action["data"]["player"]
            tile = action["data"]["tile"]

            if act == "DRAW":

                four_player_hands[pl_id].append(tile)
                hands_list.append(copy.copy(four_player_hands[pl_id]))
                draw_tile_list.append(tile)
                four_players_open_hands = copy.copy(four_players_open_hands)
                four_players_open_hands_list.append(four_players_open_hands)


            elif act == "DISCARD":

                discarded_tiles_pool_list.append(copy.copy(discarded_tiles_pool))
                discarded_tile.append(tile)
                discarded_tiles_pool.append(tile)
                four_player_hands[pl_id].remove(tile)


        elif act == 'AGARI' or act == 'RYUUKYOKU':
            # if the tag is AGARI, then the last draw is meaningless for the tile_discarded_model
            if last_act == 'DRAW':
                hands_list.pop()
                draw_tile_list.pop()
                four_players_open_hands_list.pop()

        last_act = act


# In[ ]:


# In[3]:

# def main():
#     CLOUD = environ.get("TF_KERAS_RUNNING_REMOTELY")
#     # environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/junlin/key.json"
#
#     start_year = 2021
#     # path for dir
#     dir_path = create_or_join('dataset')
#     processed_path = create_or_join("processed_data")
#     # path for output hdf5 file
#     output_json_path = path.join(processed_path, 'discarded_model_summary.json')
#     if not tf.io.gfile.exists(processed_path):
#         tf.io.gfile.makedirs(processed_path)
#
#     if CLOUD:
#         start_year = 2009
#         client = bigquery.Client(project="mahjong-305819")
#     else:
#         file = open(output_json_path, 'a+', encoding='utf-8')
#     count = 1
#
#     for year in range(start_year, 2022):
#         if year == 2020:
#             continue
#
#         print("processing " + str(year))
#         input_csv_path = path.join(dir_path, str(year) + '.csv')
#         # read data from csv file
#         df = pd.read_csv(input_csv_path)
#
#         for i in range(len(df["log_content"])):
#             xml_str = df["log_content"][i]
#
#             if type(xml_str) != str:
#                 continue
#             else:
#                 # transform data from xml format to dict
#                 node = ET.fromstring(xml_str)
#                 data = parse_mjlog(node)
#
#                 # remove three-players games
#                 if len(data["meta"]['UN'][3]["name"]) == 0:
#                     continue
#                 else:
#
#                     for j in range(len(data["rounds"])):
#
#                         # features
#                         draw_tile_list = []
#                         hands_list = []
#                         discarded_tiles_pool_list = []
#                         four_players_open_hands_list = []
#
#                         # label
#                         discarded_tile = []
#
#                         round_data = data["rounds"][j]
#
#                         get_round_info(round_data,
#                                        draw_tile_list,
#                                        hands_list,
#                                        discarded_tiles_pool_list,
#                                        four_players_open_hands_list,
#                                        discarded_tile)
#
#                         for k in range(len(draw_tile_list)):
#                             res = {
#                                 u'id': count,
#                                 u'draw_tile': str(draw_tile_list[k]),
#                                 u'hands': str(hands_list[k]),
#                                 u'discarded_tiles_pool': str(discarded_tiles_pool_list[k]),
#                                 u'four_players_open_hands': str(four_players_open_hands_list[k]),
#                                 u'discarded_tile': discarded_tile[k]
#                             }
#
#                             res_str = json.dumps(res)
#                             yield res
#                             # if CLOUD:
#                             #     error = client.insert_rows_json('mahjong.discarded', [res])  # Make an API request.
#                             #     if error:
#                             #         print("Encountered errors while inserting rows: {}".format(error))
#                             # else:
#                             #     file.write(res_str + '\n')
#                             count += 1

class DiscardedFeatureExtractor(beam.DoFn):
    def process(self, log_data, **kwargs):
        xml_str = log_data[4]
        output = []
        if type(xml_str) == str:
            # transform data from xml format to dict
            try:
                node = ET.fromstring(xml_str)
                data = parse_mjlog(node)
                # remove three-players games
                if len(data["meta"]['UN'][3]["name"]) != 0:
                    for j in range(len(data["rounds"])):
                        # features
                        draw_tile_list = []
                        hands_list = []
                        discarded_tiles_pool_list = []
                        four_players_open_hands_list = []

                        # label
                        discarded_tile = []

                        round_data = data["rounds"][j]

                        get_round_info(round_data,
                                       draw_tile_list,
                                       hands_list,
                                       discarded_tiles_pool_list,
                                       four_players_open_hands_list,
                                       discarded_tile)
                        for k in range(len(draw_tile_list)):
                            res = {
                                u'draw_tile': int(draw_tile_list[k]),
                                u'hands': hands_list[k],
                                u'discarded_tiles_pool': discarded_tiles_pool_list[k],
                                u'four_players_open_hands': four_players_open_hands_list[k],
                                u'discarded_tile': int(discarded_tile[k])
                            }
                            assert type(draw_tile_list[k]) == int, "draw tile wrong type"
                            assert type(hands_list[k]) == list, "hands_list wrong type"
                            assert type(discarded_tiles_pool_list[k]) == list, "discarded_tiles_pool_list wrong type"
                            assert type(
                                four_players_open_hands_list[k]) == list, "four_players_open_hands_list wrong type"
                            assert type(discarded_tile[k]) == int, "discarded_tile wrong type"
                            # res_str = json.dumps(res)
                            yield res
                            # output.append(res)
                            # if CLOUD:
                            #     error = client.insert_rows_json('mahjong.discarded', [res])  # Make an API request.
                            #     if error:
                            #         print("Encountered errors while inserting rows: {}".format(error))
                            # else:
                            #     file.write(res_str + '\n')
            except ET.ParseError:
                pass
        # return output

# if __name__ == "__main__":
#     main()

# In[ ]:
