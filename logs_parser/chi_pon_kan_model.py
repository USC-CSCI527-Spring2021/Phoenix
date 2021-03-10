import copy
import xml.etree.ElementTree as ET

import apache_beam as beam

from .parser import parse_mjlog


# from last player, match with cur tile
def could_pon(last_player_tile_34, cur_player_closed_hands_34):
    match_count = 0

    for hand in cur_player_closed_hands_34:
        if hand == last_player_tile_34:
            match_count += 1
            if match_count == 2:
                return True
    return False

# from last player, match with cur tile
def could_minKan(last_player_tile_34, cur_player_closed_hands_34):

    match_count = 0

    for hand in cur_player_closed_hands_34:
        if hand == last_player_tile_34:
            match_count += 1
            if match_count == 3:
                return True
    return False

def transfrom_136_to_34(player_hands_136):
    player_hands_34 = []
    for hand in player_hands_136:
        hand_34 = hand // 4
        player_hands_34.append(hand_34)

    return player_hands_34


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


def get_last_discarded_tile(player_id, four_discarded_hands):
    if len(four_discarded_hands[player_id]) == 0:
        return None
    return four_discarded_hands[player_id][-1]



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

def get_tiles_for_mikan(tiles):
    start = tiles[0] // 4 * 4

    res = []
    for i in range(4):
        res.append(start + i)

    return res

def get_player_call_actions_record(player_id, call_actions_record):
    return call_actions_record[player_id]

def is_FCH(player_call_actions_record):
    non_FCH_actions_set = set(["Chi", "Pon", "MinKan", "KaKan"])
    if len(player_call_actions_record) == 0:
        return True

    for action in player_call_actions_record:
        if action in non_FCH_actions_set:
            return False

    return True


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

        self.could_chi_list = []
        self.could_pon_list = []
        self.could_minkan_list = []

        self.call_actions_record_list = []
        self.is_FCH_list  = []
        self.player_call_actions_record_list = []
        self.four_player_open_hands_detail_list = []
        self.last_player_id_list = []
        self.open_hands_detail_list = []




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
                         riichi_bets, scores, player_winds, prevailing_wind, taken_call_actions, open_hands_details):

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
        self.call_actions_record_list.append(copy.copy(taken_call_actions))
        self.four_player_open_hands_detail_list.append(copy.copy(open_hands_details))

    # record cur four players' states, update closed and open hands after call
    def call_operations(self, action_data, closed_hands, open_hands, taken_call_actions, open_hands_details):

        caller = action_data["caller"]
        callee = action_data["callee"]
        tiles = action_data["mentsu"]
        call_type = action_data["call_type"]

        if call_type == "AnKan":
            tiles = get_tiles_for_mikan(tiles)

        self.action_list.append([call_type, tiles, callee])

        # record the actions has been taken
        taken_call_actions[caller] = copy.copy(taken_call_actions[caller])
        taken_call_actions[caller].append(call_type)

        closed_hands = copy.copy(closed_hands)
        open_hands = copy.copy(open_hands)

        reacted_tile = None
        for tile in tiles:
            if tile not in closed_hands[caller]:
                reacted_tile = tile

        closed_hands, open_hands = trans_to_open(tiles, closed_hands, open_hands, caller)

        open_hands_details[caller] = copy.copy(open_hands_details[caller])
        open_hands_details[caller].append({"tiles": tiles, "meld_type": call_type, "reacted_tile" : reacted_tile})

        return closed_hands, open_hands, open_hands_details

    def pair_data(self):
        last_player =  -1

        for i in range(len(self.player_id_list)):

            self.last_player_id_list.append(last_player)
            player_id = self.player_id_list[i]
            four_closed_hands = self.four_closed_hands_list[i]
            four_open_hands = self.four_open_hands_list[i]
            four_discarded_hands = self.four_discarded_hands_list[i]
            raw_player_wind = self.raw_player_wind_list[i]
            raw_repeat_dealer = self.raw_repeat_dealer_list[i]
            raw_riichi_bets = self.raw_riichi_bets_list[i]

            call_actions_record = self.call_actions_record_list[i]
            open_hands_details = self.four_player_open_hands_detail_list[i]

            # separate_player_enemies_hands
            self.player_tiles_list.append(
                get_player_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands))
            self.enemies_tiles_list.append(
                get_three_enemies_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands))

            # get corresponding palyer wind
            self.player_wind_list.append(get_player_wind(player_id, raw_player_wind))

            self.repeat_dealer_list.append(get_repeat_dealer_count(player_id, raw_repeat_dealer))

            self.riichi_bets_list.append(get_riichi_bets(player_id, raw_riichi_bets))

            self.open_hands_detail_list.append(open_hands_details[player_id])

            player_call_actions_record = get_player_call_actions_record(player_id, call_actions_record)
            self.player_call_actions_record_list.append(player_call_actions_record)


            if is_FCH(player_call_actions_record):
                self.is_FCH_list.append(1)
            else:
                self.is_FCH_list.append(0)

            # get discarded tiles
            if i == 0:
                last_player_tile_136 = None
                self.last_player_discarded_tile_list.append(last_player_tile_136)

                if (i + 1) < len(self.player_id_list) and player_id == self.player_id_list[i + 1]:
                    last_player = last_player
                else:
                    last_player = player_id
            else:
                last_player_tile_136 = get_last_discarded_tile(last_player, four_discarded_hands)
                self.last_player_discarded_tile_list.append(last_player_tile_136)

                if (i + 1) < len(self.player_id_list) and player_id == self.player_id_list[i+1]:
                    last_player = last_player
                else:
                    last_player = player_id

            # get could chi\pon\minKan features
            cur_player_closed_hands_34 = transfrom_136_to_34(four_closed_hands[player_id])

            if last_player_tile_136 == None:
                self.could_chi_list.append(0)
                self.could_pon_list.append(0)
                self.could_minkan_list.append(0)
            else:
                last_player_tile_34 = last_player_tile_136 // 4
                if self.could_chi(last_player_tile_34, cur_player_closed_hands_34):
                    self.could_chi_list.append(1)
                else:
                    self.could_chi_list.append(0)

                if could_pon(last_player_tile_34, cur_player_closed_hands_34):
                    self.could_pon_list.append(1)
                else:
                    self.could_pon_list.append(0)

                if could_minKan(last_player_tile_34, cur_player_closed_hands_34):
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

        taken_call_actions = [[], [], [], []]
        open_hands_details = [[], [], [], []]
        last_player_id = -1

        for k in range(len(dataset)):
            action = dataset[k]
            act = action["tag"]

            if act == 'REACH':
                player_id = action["data"]["player"]

                if action['data']['step'] == 1:
                    self.record_cur_state(player_id, closed_hands, open_hands, discarded_hands, doras, dealer,
                                          repeat_dealer, riichi_bets, scores, player_winds,
                                          prevailing_wind, taken_call_actions, open_hands_details)
                    self.action_list.append([act])
                    taken_call_actions[player_id] = copy.copy(taken_call_actions[player_id])
                    taken_call_actions[player_id].append(act)

                if action['data']['step'] == 2:

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
                self.record_cur_state(player_id, closed_hands, open_hands, discarded_hands, doras, dealer,
                                          repeat_dealer, riichi_bets, scores, player_winds,
                                          prevailing_wind, taken_call_actions, open_hands_details)
                closed_hands, open_hands, open_hands_details = self.call_operations(action_data, closed_hands, open_hands, taken_call_actions, open_hands_details)


            # deal with draw and discard operations
            elif act in ["DRAW", "DISCARD"]:
                player_id = action["data"]["player"]
                tile = action["data"]["tile"]

                # 记录作出这个action之前场面的情况，当前玩家，与其他玩家，摸牌
                if act == "DRAW":
                    # record cur state for four players, and cur player id
                    self.record_cur_state(player_id, closed_hands, open_hands, discarded_hands, doras, dealer,
                                          repeat_dealer, riichi_bets, scores, player_winds,
                                          prevailing_wind, taken_call_actions, open_hands_details)
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

        xml_str = log_data[4]

        # if type(xml_str) != str:
        #     continue

        try:
            node = ET.fromstring(xml_str)
            data = parse_mjlog(node)

            # remove three-players games
            # if len(data["meta"]['UN'][3]["name"]) == 0:
            #     continue

            last_dealer = -1
            repeat_dealer = [0, 0, 0, 0]
            prevailing_wind = 'E'

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
                    u'open_hands_detail': self.open_hands_detail_list[k],
                    u'enemies_tiles': self.enemies_tiles_list[k],
                    u'dora': self.dora_list[k],
                    u'scores': self.scores_list[k],
                    u'last_player_discarded_tile': self.last_player_discarded_tile_list[k],
                    u'could_chi': self.could_chi_list[k],
                    u'could_pon': self.could_pon_list[k],
                    u'could_minkan': self.could_minkan_list[k],
                    u'is_FCH': self.is_FCH_list[k],
                    u'action': self.action_list[k]
                }
                yield res
                # output.append(res)
            # return output
        except ET.ParseError:
            return
# In[ ]:
