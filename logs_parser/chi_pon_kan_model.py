#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import xml.etree.ElementTree as ET

import apache_beam as beam

from .parser import parse_mjlog


# In[2]:


# In[3]:


# In[4]:


# In[5]:


# def could_anKan(cur_player_closed_hands):
#     if (len(cur_player_closed_hands) >= 4):      
#         cur_player_closed_hands.sort()
#         for i in range(len(cur_player_closed_hands) - 3):
#             if (cur_player_closed_hands[i] == cur_player_closed_hands[i+1] and cur_player_closed_hands[i] == cur_player_closed_hands[i+2] and cur_player_closed_hands[i] == cur_player_closed_hands[i+3]):
#                 return True
#     return False


# # from draw, match with cur tile
# def could_kaKan(draw_tile, cur_player_open_hands):

#     if (len(cur_player_open_hands) < 3):
#         return False

#     draw_tile_34 = draw_tile // 4

#     match_count = 0
#     for hand in cur_player_open_hands:
#         hand_34 = hand // 4
#         if hand_34 == draw_tile_34:
#             match_count += 1
#             if match_count == 3:
#                 return True
#     return False


# from any player, match with cur tile
def could_pon(last_three_tiles_34, cur_player_closed_hands_34):
    for tile in last_three_tiles_34:
        if tile == None:
            continue

        match_count = 0

        for hand in cur_player_closed_hands_34:
            if hand == tile:
                match_count += 1
                if match_count == 2:
                    return True
    return False


# from any player, match with cur tile
def could_minKan(last_three_tiles_34, cur_player_closed_hands_34):
    for tile in last_three_tiles_34:
        if tile == None:
            continue

        match_count = 0

        for hand in cur_player_closed_hands_34:
            if hand == tile:
                match_count += 1
                if match_count == 3:
                    return True
    return False


# In[6]:


def transfrom_136_to_34(player_hands_136):
    player_hands_34 = []
    for hand in player_hands_136:
        hand_34 = hand // 4
        player_hands_34.append(hand_34)

    return player_hands_34


# In[ ]:


# In[7]:


# In[8]:


# def main():
#     CLOUD = environ.get("TF_KERAS_RUNNING_REMOTELY")
#     # environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/junlin/key.json"
#
#     start_year = 2021
#     if CLOUD:
#         start_year = 2009
#         client = bigquery.Client(project="mahjong-305819")
#
#     count = 1
#     for year in range(2009, 2022):
#         print('processing {}'.format(year))
#         if year == 2020:
#             continue
#         input_csv_path = path.join(create_or_join('dataset'), '{}.csv'.format(year))
#
#         df = pd.read_csv(input_csv_path)
#
#         output_json_path = path.join(create_or_join("processed_data"), 'tiles_state_and_action_sum.json')
#         if not CLOUD:
#             file = open(output_json_path, 'a+', encoding='utf-8')
#
#         # raw data extract from logs
#         global player_id_list
#         global four_closed_hands_list
#         global four_discarded_hands_list
#         global four_open_hands_list
#         global dora_list
#         global action_list
#         global dealer_list
#         global raw_repeat_dealer_list
#         global raw_riichi_bets_list
#         global scores_list
#         global raw_player_wind_list
#         global prevailing_wind_list
#
#         # paired data
#         global player_tiles_list
#         global enemies_tiles_list
#         global player_wind_list
#         global repeat_dealer_list
#         global riichi_bets_list
#         global last_player_discarded_tile_list
#         global last_three_discarded_tile_list
#         global could_chi_list
#         global could_pon_list
#         global could_minkan_list
#         global valid_chi_mentsu
#
#
#         # last_player_tile cur_possible_tiles
#         valid_chi_mentsu = {
#              0: [[1, 2]],
#              1: [[2, 3], [0, 2]],
#              2: [[3, 4], [1, 3], [0, 1]],
#              3: [[4, 5], [2, 4], [1, 2]],
#              4: [[5, 6], [3, 5], [2, 3]],
#              5: [[6, 7], [4, 6], [3, 4]],
#              6: [[7, 8], [5, 7], [4, 5]],
#              7: [[6, 8], [5, 6]],
#              8: [[6, 7]],
#
#              9: [[10, 11]],
#              10: [[11, 12], [9, 11]],
#              11: [[12, 13], [10, 12], [9, 10]],
#              12: [[13, 14], [11, 13], [10, 11]],
#              13: [[14, 15], [12, 14], [11, 12]],
#              14: [[15, 16], [13, 15], [12, 13]],
#              15: [[16, 17], [14, 16], [13, 14]],
#              16: [[15, 17], [14, 15]],
#              17: [[15, 16]],
#
#
#              18: [[19, 20]],
#              19: [[20, 21], [18, 20]],
#              20: [[21, 22], [19, 21], [18, 19]],
#              21: [[22, 23], [20, 22], [19, 20]],
#              22: [[23, 24], [21, 23], [20, 21]],
#              23: [[24, 25], [22, 24], [21, 22]],
#              24: [[25, 26], [23, 25], [22, 23]],
#              25: [[24, 26], [23, 24]],
#              26: [[24, 25]]
#         }
#
#
#         for i in range(len(df["log_content"])):
#             xml_str = df["log_content"][i]
#             if type(xml_str) != str:
#                     continue
#             else:
#                 node = ET.fromstring(xml_str)
#                 data = parse_mjlog(node)
#
#                 # remove three-players games
#                 if len(data["meta"]['UN'][3]["name"]) == 0:
#                     continue
#                 else:
#
#                     player_id_list = []
#                     four_closed_hands_list = []
#                     four_discarded_hands_list = []
#                     four_open_hands_list = []
#                     dora_list = []
#                     action_list = []
#                     dealer_list = []
#                     raw_repeat_dealer_list = []
#                     raw_riichi_bets_list = []
#                     scores_list = []
#                     raw_player_wind_list = []
#                     prevailing_wind_list = []
#
#                     player_tiles_list = []
#                     enemies_tiles_list = []
#                     player_wind_list = []
#                     repeat_dealer_list = []
#                     riichi_bets_list = []
#                     last_player_discarded_tile_list = []
#                     last_three_discarded_tile_list = []
#                     could_chi_list = []
#                     could_pon_list = []
#                     could_minkan_list = []
#
#                     last_dealer = -1
#                     repeat_dealer = [0, 0, 0, 0]
#                     prevailing_wind = 'E'
#
#                     for j in range(len(data["rounds"])):
#
#                         dealer = int(data['rounds'][j][0]["data"]["oya"])
#                         player_winds = init_player_winds(dealer)
#
#                         if j > 3 and dealer == 0:
#                             prevailing_wind = 'S'
#
#                         if dealer == last_dealer:
#                             repeat_dealer[dealer] += 1
#
#                         get_round_info(data['rounds'][j], repeat_dealer, player_winds, prevailing_wind)
#
#                         last_dealer = dealer
#
#                     pair_data()
#
#                     for k in range(len(player_id_list)):
#                         res = {
#                             u'id': count,
#                             u'player_id': player_id_list[k],
#                             u'dealer': dealer_list[k],
#                             u'repeat_dealer': repeat_dealer_list[k],
#                             u'riichi_bets': riichi_bets_list[k],
#                             u'player_wind': player_wind_list[k],
#                             u'prevailing_wind': prevailing_wind_list[k],
#                             u'player_tiles': str(player_tiles_list[k]),
#                             u'enemies_tiles': str(enemies_tiles_list[k]),
#                             u'dora': str(dora_list[k]),
#                             u'scores': str(scores_list[k]),
#                             u'last_player_discarded_tile': last_player_discarded_tile_list[k],
#                             u'last_three_discarded_tile': str(last_three_discarded_tile_list[k]),
#                             u'could_chi': could_chi_list[k],
#                             u'could_pon': could_pon_list[k],
#                             u'could_minkan': could_minkan_list[k],
#                             u'action': str(action_list[k])
#                         }
#
#                         res_str = json.dumps(res)
#                         yield res
#                         # if CLOUD:
#                         #     error = client.insert_rows_json('mahjong.chi_pon_kan', [res])  # Make an API request.
#                         #     if error:
#                         #         print("Encountered errors while inserting rows: {}".format(error))
#                         # else:
#                         #     file.write(res_str + '\n')
#                         count += 1


def get_player_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands):
    return {
        'closed_hand:': four_closed_hands[player_id],
        'open_hand': four_open_hands[player_id],
        'discarded_tiles': four_discarded_hands[player_id]
    }


def get_enemy(enemy_id, four_closed_hands, four_open_hands, four_discarded_hands):
    return {
        'enemy_seat': enemy_id,
        'closed_hand:': four_closed_hands[enemy_id],
        'open_hand': four_open_hands[enemy_id],
        'discarded_tiles': four_discarded_hands[enemy_id]
    }


def get_three_enemies_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands):
    enemies_tiles = []
    for i in range(4):
        if player_id != i:
            enemies_tiles.append(get_enemy(i, four_closed_hands, four_open_hands, four_discarded_hands))

    return enemies_tiles


def get_player_wind(player_id, raw_player_wind):
    return raw_player_wind[player_id]


def get_repeat_dealer_count(player_id, raw_repeat_dealer):
    return raw_repeat_dealer[player_id]


def get_riichi_bets(player_id, raw_riichi_bets):
    return raw_riichi_bets[player_id]


def get_last_player_id(player_id):
    if player_id == 0:
        return 3
    else:
        return player_id - 1


def get_last_discarded_tile(player_id, four_discarded_hands):
    if len(four_discarded_hands[player_id]) == 0:
        return None
    return four_discarded_hands[player_id][-1]


def get_last_three_discarded_tile(player_id, four_discarded_hands):
    last_three_discarded_tile = []

    for i in range(4):
        if i != player_id:
            last_three_discarded_tile.append(get_last_discarded_tile(i, four_discarded_hands))

    return last_three_discarded_tile


def closed_hands_append(closed_hands, player_id, tile):
    closed_hands[player_id] = copy.copy(closed_hands[player_id])
    closed_hands[player_id].append(tile)
    return closed_hands


def closed_hands_remove(closed_hands, player_id, tile):
    closed_hands[player_id] = copy.copy(closed_hands[player_id])
    closed_hands[player_id].remove(tile)
    return closed_hands


def open_hands_append(open_hands, caller, tile):
    open_hands[caller] = copy.copy(open_hands[caller])
    open_hands[caller].append(tile)
    return open_hands


# remove tile from closed_hands[caller], and add it into open_hands[caller]
def trans_to_open(tiles, closed_hands, open_hands, caller):
    for tile in tiles:
        if tile in closed_hands[caller]:
            closed_hands = closed_hands_remove(closed_hands, caller, tile)

        if tile not in open_hands[caller]:
            open_hands = open_hands_append(open_hands, caller, tile)

    return closed_hands, open_hands


def discarded_hands_append(discarded_hands, player_id, tile):
    discarded_hands[player_id] = copy.copy(discarded_hands[player_id])
    discarded_hands[player_id].append(tile)

    return discarded_hands


class ChiPonKanFeatureExtractor(beam.DoFn):
    def __init__(self, *unused_args, **unused_kwargs):
        super().__init__(*unused_args, **unused_kwargs)
        self.player_id_list = []
        self.four_closed_hands_list = []
        self.four_discarded_hands_list = []
        self.four_open_hands_list = []
        self.dora_list = []
        self.action_list = []
        self.dealer_list = []
        self.raw_repeat_dealer_list = []
        self.raw_riichi_bets_list = []
        self.scores_list = []
        self.raw_player_wind_list = []
        self.prevailing_wind_list = []

        self.player_tiles_list = []
        self.enemies_tiles_list = []
        self.player_wind_list = []
        self.repeat_dealer_list = []
        self.riichi_bets_list = []
        self.last_player_discarded_tile_list = []
        self.last_three_discarded_tile_list = []
        self.could_chi_list = []
        self.could_pon_list = []
        self.could_minkan_list = []
        self.valid_chi_mentsu = {
            0: [[1, 2]],
            1: [[2, 3], [0, 2]],
            2: [[3, 4], [1, 3], [0, 1]],
            3: [[4, 5], [2, 4], [1, 2]],
            4: [[5, 6], [3, 5], [2, 3]],
            5: [[6, 7], [4, 6], [3, 4]],
            6: [[7, 8], [5, 7], [4, 5]],
            7: [[6, 8], [5, 6]],
            8: [[6, 7]],

            9: [[10, 11]],
            10: [[11, 12], [9, 11]],
            11: [[12, 13], [10, 12], [9, 10]],
            12: [[13, 14], [11, 13], [10, 11]],
            13: [[14, 15], [12, 14], [11, 12]],
            14: [[15, 16], [13, 15], [12, 13]],
            15: [[16, 17], [14, 16], [13, 14]],
            16: [[15, 17], [14, 15]],
            17: [[15, 16]],

            18: [[19, 20]],
            19: [[20, 21], [18, 20]],
            20: [[21, 22], [19, 21], [18, 19]],
            21: [[22, 23], [20, 22], [19, 20]],
            22: [[23, 24], [21, 23], [20, 21]],
            23: [[24, 25], [22, 24], [21, 22]],
            24: [[25, 26], [23, 25], [22, 23]],
            25: [[24, 26], [23, 24]],
            26: [[24, 25]]
        }

    def init_player_winds(self, dealer):

        if dealer == 0:
            player_winds = ['E', 'N', 'W', 'S']
        elif dealer == 1:
            player_winds = ['S', 'E', 'N', 'W']
        elif dealer == 2:
            player_winds = ['W', 'S', 'E', 'N']
        else:
            player_winds = ['N', 'W', 'S', 'E']

        return player_winds

    def record_cur_state(self, player_id, closed_hands, open_hands, discarded_hands, doras, dealer, repeat_dealer,
                         riichi_bets, scores, player_winds, prevailing_wind):

        self.player_id_list.append(player_id)
        self.four_closed_hands_list.append(copy.copy(closed_hands))
        self.four_open_hands_list.append(copy.copy(open_hands))
        self.four_discarded_hands_list.append(copy.copy(discarded_hands))
        self.dora_list.append(copy.copy(doras))
        self.dealer_list.append(dealer)
        self.raw_repeat_dealer_list.append(copy.copy(repeat_dealer))
        self.scores_list.append(copy.copy(scores))
        self.raw_riichi_bets_list.append(copy.copy(riichi_bets))
        self.raw_player_wind_list.append(copy.copy(player_winds))
        self.prevailing_wind_list.append(prevailing_wind)

    # record cur four players' states, update closed and open hands after call
    def call_operations(self, action_data, closed_hands, open_hands):

        caller = action_data["caller"]
        callee = action_data["callee"]
        tiles = action_data["mentsu"]
        call_type = action_data["call_type"]

        self.action_list.append([call_type, tiles, callee])

        closed_hands = copy.copy(closed_hands)
        open_hands = copy.copy(open_hands)

        closed_hands, open_hands = trans_to_open(tiles, closed_hands, open_hands, caller)

        return closed_hands, open_hands

    def pair_data(self):
        for i in range(len(self.player_id_list)):
            player_id = self.player_id_list[i]
            four_closed_hands = self.four_closed_hands_list[i]
            four_open_hands = self.four_open_hands_list[i]
            four_discarded_hands = self.four_discarded_hands_list[i]
            raw_player_wind = self.raw_player_wind_list[i]
            raw_repeat_dealer = self.raw_repeat_dealer_list[i]
            raw_riichi_bets = self.raw_riichi_bets_list[i]

            # separate_player_enemies_hands
            self.player_tiles_list.append(
                get_player_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands))
            self.enemies_tiles_list.append(
                get_three_enemies_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands))

            # get corresponding palyer wind
            self.player_wind_list.append(get_player_wind(player_id, raw_player_wind))

            self.repeat_dealer_list.append(get_repeat_dealer_count(player_id, raw_repeat_dealer))

            self.riichi_bets_list.append(get_riichi_bets(player_id, raw_riichi_bets))

            # get discarded tiles
            last_player = get_last_player_id(player_id)
            last_player_tile_136 = get_last_discarded_tile(last_player, four_discarded_hands)
            self.last_player_discarded_tile_list.append(last_player_tile_136)

            last_three_tiles_136 = get_last_three_discarded_tile(player_id, four_discarded_hands)
            self.last_three_discarded_tile_list.append(last_three_tiles_136)

            # get could chi\pon\minKan features
            cur_player_closed_hands_34 = transfrom_136_to_34(four_closed_hands[player_id])

            if last_player_tile_136 == None:
                self.could_chi_list.append(0)
            else:
                last_player_tile_34 = last_player_tile_136 // 4
                if self.could_chi(last_player_tile_34, cur_player_closed_hands_34):
                    self.could_chi_list.append(1)
                else:
                    self.could_chi_list.append(0)

            last_three_tiles_34 = []
            for t in last_three_tiles_136:
                if t == None:
                    last_three_tiles_34.append(None)
                else:
                    last_three_tiles_34.append(t // 4)

            if could_pon(last_three_tiles_34, cur_player_closed_hands_34):
                self.could_pon_list.append(1)
            else:
                self.could_pon_list.append(0)

            if could_minKan(last_three_tiles_34, cur_player_closed_hands_34):
                self.could_minkan_list.append(1)
            else:
                self.could_minkan_list.append(0)

    def get_round_info(self, dataset, repeat_dealer, player_winds, prevailing_wind):

        init_data = dataset[0]["data"]
        dealer = int(init_data["oya"])
        scores = init_data["scores"]
        riichi_bets = [0, 0, 0, 0]
        doras = [init_data["dora"]]

        # player_tiles, get from init
        closed_hands = copy.copy(init_data["hands"])
        open_hands = [[], [], [], []]
        discarded_hands = [[], [], [], []]

        for k in range(len(dataset)):
            action = dataset[k]
            act = action["tag"]

            #       {'tag': 'REACH', 'data': {'player': 3, 'step': 1}}
            #       {'tag': 'REACH', 'data': {'player': 3, 'step': 2, 'scores': [17200, 26900, 40900, 14000]}}
            if act == 'REACH':

                if action['data']['step'] == 1:
                    self.record_cur_state(player_id, closed_hands, open_hands, discarded_hands, doras, dealer,
                                          repeat_dealer,
                                          riichi_bets, scores, player_winds, prevailing_wind)
                    self.action_list.append([act])

                if action['data']['step'] == 2:

                    player_id = action["data"]["player"]
                    riichi_bets[player_id] += 1

                    if 'scores' not in action['data']:
                        scores[player_id] -= 1000
                    else:
                        scores = action['data']['scores']

            #       {'tag': 'DORA', 'data': {'hai': 113}}
            elif act == 'DORA':
                doras.append(action['data']['hai'])

            elif act == "CALL":
                action_data = action["data"]
                self.record_cur_state(action_data["caller"], closed_hands, open_hands, discarded_hands, doras, dealer,
                                      repeat_dealer, riichi_bets, scores, player_winds, prevailing_wind)
                closed_hands, open_hands = self.call_operations(action_data, closed_hands, open_hands)


            # deal with draw and discard operations
            elif act in ["DRAW", "DISCARD"]:
                player_id = action["data"]["player"]
                tile = action["data"]["tile"]

                # 记录作出这个action之前场面的情况，当前玩家，与其他玩家，摸牌
                if act == "DRAW":
                    # record cur state for four players, and cur player id

                    self.record_cur_state(player_id, closed_hands, open_hands, discarded_hands, doras, dealer,
                                          repeat_dealer,
                                          riichi_bets, scores, player_winds, prevailing_wind)
                    self.action_list.append([act, tile])

                    # update state
                    closed_hands = closed_hands_append(closed_hands, player_id, tile)

                # update discarded_hands and four_closed_hands for cur player
                elif act == "DISCARD":
                    closed_hands = closed_hands_remove(closed_hands, player_id, tile)
                    discarded_hands = discarded_hands_append(discarded_hands, player_id, tile)

            elif act == 'AGARI' or act == 'RYUUKYOKU':
                continue

    # from last player
    def could_chi(self, last_player_tile_34, cur_player_closed_hands_34):
        if last_player_tile_34 >= 27:
            return False

        could_match_tiles = self.valid_chi_mentsu[last_player_tile_34]

        for tiles in could_match_tiles:
            match_count = 0
            for tile in tiles:
                for hand in cur_player_closed_hands_34:
                    if hand == tile:
                        match_count += 1
                        if match_count == 2:
                            return True
                        break

        return False

    def process(self, log_data, **kwargs):
        # raw data extract from logs
        # global player_id_list
        # global four_closed_hands_list
        # global four_discarded_hands_list
        # global four_open_hands_list
        # global dora_list
        # global action_list
        # global dealer_list
        # global raw_repeat_dealer_list
        # global raw_riichi_bets_list
        # global scores_list
        # global raw_player_wind_list
        # global prevailing_wind_list
        #
        # # paired data
        # global player_tiles_list
        # global enemies_tiles_list
        # global player_wind_list
        # global repeat_dealer_list
        # global riichi_bets_list
        # global last_player_discarded_tile_list
        # global last_three_discarded_tile_list
        # global could_chi_list
        # global could_pon_list
        # global could_minkan_list
        # global valid_chi_mentsu
        xml_str = log_data[4]
        node = ET.fromstring(xml_str)
        data = parse_mjlog(node)
        last_dealer = -1
        repeat_dealer = [0, 0, 0, 0]
        prevailing_wind = 'E'

        output = []

        for j in range(len(data["rounds"])):

            dealer = int(data['rounds'][j][0]["data"]["oya"])
            player_winds = self.init_player_winds(dealer)

            if j > 3 and dealer == 0:
                prevailing_wind = 'S'

            if dealer == last_dealer:
                repeat_dealer[dealer] += 1

            self.get_round_info(data['rounds'][j], repeat_dealer, player_winds, prevailing_wind)

            last_dealer = dealer

        self.pair_data()

        for k in range(len(self.player_id_list)):
            res = {
                u'player_id': self.player_id_list[k],
                u'dealer': self.dealer_list[k],
                u'repeat_dealer': self.repeat_dealer_list[k],
                u'riichi_bets': self.riichi_bets_list[k],
                u'player_wind': self.player_wind_list[k],
                u'prevailing_wind': self.prevailing_wind_list[k],
                u'player_tiles': self.player_tiles_list[k],
                u'enemies_tiles': self.enemies_tiles_list[k],
                u'dora': self.dora_list[k],
                u'scores': self.scores_list[k],
                u'last_player_discarded_tile': self.last_player_discarded_tile_list[k],
                u'last_three_discarded_tile': self.last_three_discarded_tile_list[k],
                u'could_chi': self.could_chi_list[k],
                u'could_pon': self.could_pon_list[k],
                u'could_minkan': self.could_minkan_list[k],
                u'action': self.action_list[k]
            }
            yield res
            # output.append(res)
        # return output
# In[ ]:


# In[ ]:

# if __name__ == "__main__":
#     main()


# In[ ]:


# In[ ]:


# def print_result(res, i):
#     print('cur state is shown below: ')
#     print('player is {} '.format(res[0][i]))
#     print('dealer is {} '.format(res[1][i]))
#     print('counters of repeat dealer is {} '.format(res[2][i]))
#     print('riichi bets for cur player is {} '.format(res[3][i]))
#     print('cur player wind is {} '.format(res[4][i]))
#     print('cur prevailing wind is {} '.format(res[5][i]))
#     print('cur player tiles: {} '.format(res[6][i]))
#     print('cur enemies tiles: {} '.format(res[7][i]))
#     print('dora: {} '.format(res[8][i]))
#     print('points on the panel: {} '.format(res[9][i]))
#     print('could chi? {}'.format(res[12][i]))
#     print('could pon? {}'.format(res[13][i]))
#     print('could min? {}'.format(res[14][i]))


#     print('action to take based on cur state is: {} '.format(res[-1][i]))

#     print(' ')


# In[ ]:


# In[ ]:


# def test_could_minKan(last_three_tiles_34, cur_player_closed_hands_34):
#     print(cur_player_closed_hands_34)
#     print(last_three_tiles_34)
#     for tile in last_three_tiles_34:  
#         if tile == None:
#             continue

#         match_count = 0

#         for hand in cur_player_closed_hands_34:
#             if hand == tile:
#                 match_count += 1
#                 if match_count == 3:
#                     return True
#     return False


# for i in range(len(res[0])):
#     if could_minkan_list[i] == 1:

#         print(player_tiles_list[i])
#         print(last_three_discarded_tile_list[i])
#         print(could_minkan_list[i])
#         print(action_list[i])
#         print( '  ')


# In[ ]:
