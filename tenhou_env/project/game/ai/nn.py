from __future__ import absolute_import

import random

import numpy as np
from mahjong.tile import TilesConverter
from mahjong.utils import is_pon
from utils.decisions_logger import MeldPrint


def getGeneralFeature(player):
    def _indicator2dora(dora_indicator):
        dora = dora_indicator // 4
        if dora < 27:  # EAST
            if dora == 8:
                dora = -1
            elif dora == 17:
                dora = 8
            elif dora == 26:
                dora = 17
        else:
            dora -= 9 * 3
            if dora == 3:
                dora = -1
            elif dora == 6:
                dora = 3
            dora += 9 * 3
        dora += 1
        return dora

    def _transscore(score):
        feature = np.zeros((34))
        if score > 50000:
            score = 50000 
        a = score // (50000.0 / 33)
        feature[int(a)] = 1
        return feature

    def _getPlayerTiles(player):
        closed_hand_feature = np.zeros((4, 34))
        open_hand_feature = np.zeros((4, 34))
        discarded_tiles_feature = np.zeros((4, 34))

        open_hand = [tile for meld in player.melds for tile in meld.tiles]
        closed_hand = [tile for tile in player.tiles if tile not in open_hand]
        discarded_tiles = player.discards
        
        for val in closed_hand:
            idx=0
            while(closed_hand_feature[idx][val//4] == 1):
                idx+=1
            closed_hand_feature[idx][val//4] = 1
        for val in open_hand:
            idx = 0
            while (open_hand_feature[idx][val // 4] == 1):
                idx += 1
            open_hand_feature[idx][val // 4] = 1
        for val in discarded_tiles:
            # Got Tile class need to extract value
            val = val.value
            idx = 0
            while (discarded_tiles_feature[idx][val // 4] == 1):
                idx += 1
            discarded_tiles_feature[idx][val // 4] = 1
        
        return np.concatenate((closed_hand_feature,open_hand_feature,discarded_tiles_feature))

    def _getSelfTiles(player): 
        return _getPlayerTiles(player) 

    def _getEnemiesTiles(player): 
        np.concatenate((
            _getPlayerTiles(player.table.get_player(1)),
            _getPlayerTiles(player.table.get_player(2)),
            _getPlayerTiles(player.table.get_player(3))
            ))
    
    def _getDoraList(player):  
        dora_indicators = player.table.dora_indicators
        dora_feature = np.zeros((5, 34))
        for idx, dora_indicator in enumerate(dora_indicators):
            dora_feature[idx][_indicator2dora(dora_indicator)] = 1
        return dora_feature

    def _getScoreList1(player): 
        scores_feature = np.zeros((4, 34))
        for i in range(4):
            scores_feature[i] = _transscore(player.table.get_player(i).scores)
        return scores_feature

    def _getBoard1(player): 
        
        dealer_feature = np.zeros((1, 34))
        dealer_seat = player.table.dealer_seat
        player_seat = player.seat
        dealer_feature[0][(dealer_seat+4-player_seat)%4] = 1


        repeat_dealer_feature = np.zeros((1, 34))
        repeat_dealer = player.table.count_of_honba_sticks
        repeat_dealer_feature[0][repeat_dealer] = 1

        riichi_bets_feature = np.zeros((1, 34))
        riichi_bets = player.table.count_of_riichi_sticks
        riichi_bets_feature[0][riichi_bets] = 1

        player_wind_feature = np.zeros((1, 34))
        player_wind_feature[27 + (4-player.table.dealer_seat)%4] = 1

        prevailing_wind_feature = np.zeros((1, 34))
        prevailing_wind_feature[27 + player.table.round_wind_number//4] = 1

        return np.concatenate((
            dealer_feature,
            repeat_dealer_feature,
            riichi_bets_feature,
            player_wind_feature,
            prevailing_wind_feature
        ))

    return np.concatenate((            
        _getSelfTiles(player), #(12,34)
        _getDoraList(player), #(5,34)
        _getBoard1(player), #(5,34)
        _getEnemiesTiles(player), #(36,34)
        _getScoreList1(player) #(4,34)
    ))

class Chi:
    def __init__(self, player):
        self.player = player
        # self.input_shape = (63, 34, 1)
        # self.strategy = 'local'        # fix here
        # self.model = models.make_or_restore_model(self.input_shape, "chi", self.strategy)
        # load models from current working dir
        # self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'chi'))
        print('###### Chi model initialized #######')

    def should_call_chi(self, tile_136, is_kamicha_discard):
        # random make a decision
        return (True, 10) if random.randint(0, 1) else (False, 10)
        # features = self.getFeature(tile_136)
        # p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        # if p_do > p_donot:
        #     return True, p_do
        # else:
        #     return False, p_donot

    def getFeature(self, tile_136):
        tile_136_feature = np.zeros((1, 34))
        tile_136_feature[0][tile_136//4] = 1
        x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))


class Pon:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        # self.input_shape = (63, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "pon", self.strategy)
        # load models from current working dir
        # self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'pon'))
        print('###### Pon model initialized #######')

    def should_call_pon(self, tile_136, is_kamicha_discard):
        # random make a decision
        return (True, 10) if random.randint(0, 1) else (False, 0)
        # features = self.getFeature(tile_136)
        # p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        # if p_do > p_donot:
        #     return True, p_do
        # else:
        #     return False, p_donot

    def getFeature(self, tile_136):
        tile_136_feature = np.zeros((1, 34))
        tile_136_feature[0][tile_136 // 4] = 1
        x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))



class Kan:
    def __init__(self, player):
        self.player = player
        # self.input_shape = (66, 34, 1)
        # self.strategy = 'local'
        # self.model = models.make_or_restore_model(self.input_shape, "kan", self.strategy)
        # load models from current working dir
        # self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'kan'))
        print('###### Kan model initialized #######')

    def should_call_kan(self, tile_136, open_kan, from_riichi=False):
        # we can't call kan on the latest tile
        # if self.player.table.count_of_remaining_tiles <= 1:
        #     return None

        # tile_34 = tile_136 // 4
        # closed_hand_34 = TilesConverter.to_34_array(self.player.closed_hand)

        # # let's check can we upgrade opened pon to the kan
        # pon_melds = [x for x in self.player.meld_34_tiles if is_pon(x)]
        # has_shouminkan_candidate = False
        # for meld in pon_melds:
        #     # tile is equal to our already opened pon
        #     if tile_34 in meld:
        #         has_shouminkan_candidate = True


        # if not open_kan and not has_shouminkan_candidate and closed_hand_34[tile_34] != 4:
        #     return None

        # if open_kan and closed_hand_34[tile_34] != 3:
        #     return None

        # kan_type = has_shouminkan_candidate and MeldPrint.SHOUMINKAN or MeldPrint.KAN 
        # return kan_type

        # features = self.getFeature(tile_136, open_kan)
        # p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        # if p_do > p_donot:
        #     return kan_type
        # else:
        #     return None
        def _can_minkan(tile, closed_hand):
            count = np.zeros(34)
            for t in closed_hand:
                count[t//4]+=1
            if count[tile//4]==3:
                return True
            return False
        def _can_ankan(tile, closed_hand):
            count = np.zeros(34)
            for t in closed_hand:
                count[t//4]+=1
            if count[tile//4]==4:
                return True
            return False
        def _can_kakan(tile, melds):
            for meld in melds:
                if (meld.type == MeldPrint.PON) and (tile//4 == meld.tiles[0]//4):
                    return True
            return False
        kan_type_feature = np.zeros((3, 34))
        closed_hand = self.player.closed_hand
        if not open_kan:
            if _can_kakan(tile_136, self.player.melds): #KaKan
                kan_type_feature[2] = 1
                kan_type = MeldPrint.SHOUMINKAN
            elif _can_ankan(tile_136, closed_hand): #AnKan
                kan_type_feature[1] = 1
                kan_type = MeldPrint.KAN
            else:
                return None
        else: 
            if _can_minkan(tile_136, closed_hand):#MinKan
                kan_type_feature[0] = 1 
                kan_type = MeldPrint.SHOUMINKAN
            else:
                return None
        # random make a decision
        return kan_type #if random.randint(0, 1) else None

    def getFeature(self, tile136, kan_type_feature):
        tile136_feature = np.zeros((1, 34))
        tile136_feature[0][tile136 // 4] = 1        
        x = np.concatenate((kan_type_feature, tile136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))


class Riichi:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        # self.input_shape = (62, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "riichi", self.strategy)
        # load models from current working dir
        # self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'riichi'))
        print('###### Riichi model initialized #######')

    def should_call_riichi(self):
        # random make a decision
        return (True, 10) if random.randint(0, 1) else (False, 10)
        # features = self.getFeature()
        # p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        # if p_do > p_donot:
        #     return True, p_do
        # else:
        #     return False, p_donot

    def getFeature(self):
        x = getGeneralFeature(self.player)
        return x.reshape((x.shape[0], x.shape[1], 1))

class Discard:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        # self.input_shape = (16, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "discard", self.strategy)
        # load models from current working dir
        # self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'discarded'))
        print('###### Discarded model initialized #######')

    def discard_tile(self, all_hands_136=None, closed_hands_136=None, with_riichi=False):
        '''
        The reason why these two should be input:
        if the "discard" action is from "discard after meld", since we have decided to meld, what
        to do next is to decide what to discard, but the meld has not happened (and could also be 
        interrupted), so we could only use discard model based on supposed hands after melding

        '''
        # random output a tile
        tile_to_discard_136 = random.choice(self.player.closed_hand)
        # features = self.getFeature(all_hands_136, closed_hands_136, with_riichi)
        # tile_to_discard = np.argmax(self.model.predict(np.expand_dims(features, axis=0))[0])
        # tile_to_discard_136 = [h for h in closed_hands_136 if h // 4 == tile_to_discard][-1]
        # if multiple tiles exists, return the one which is not red dora
        return tile_to_discard_136

    def getFeature(self, open_hands_136, closed_hands_136, with_riichi):
        from trainer.models import transform_discard_features
        data = {
            "draw_tile": self.player.tiles[-1],
            "discarded_tiles_pool": self.player.table.revealed_tiles_136,
            "four_players_open_hands": self.player.table.melded_tiles,
            ## fake data here, pass in just to get function running
            "discarded_tile": 0,
        }
        # if hands are none, get hands from self.player
        return transform_discard_features(data)['features']  # change this according to actual shape
