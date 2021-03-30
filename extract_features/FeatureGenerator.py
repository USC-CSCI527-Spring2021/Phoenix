# Author: Jingwen Sun
# This is a class used for extract features from a tiles_state_and_action list for chi/pon/kan/riichi model.
# usage:
#   at the bottom of this file

import json

import numpy as np
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from tensorflow.keras.utils import to_categorical


class FeatureGenerator:
    def __init__(self):
        """
        changed the input from filename to tiles_state_and_action data
        By Jun Lin
        """
        self.shanten_calculator = Shanten()

    def getPlayerTiles(self, player_tiles):
        closed_hand_136 = player_tiles['closed_hand:']
        open_hand_136 = player_tiles['open_hand']
        discarded_tiles_136 = player_tiles['discarded_tiles']

        closed_hand_feature = np.zeros((4, 34))
        open_hand_feature = np.zeros((4, 34))
        discarded_tiles_feature = np.zeros((4, 34))
        
        for val in closed_hand_136:
            idx=0
            while(closed_hand_feature[idx][val//4] == 1):
                idx+=1
            closed_hand_feature[idx][val//4] = 1
        for val in open_hand_136:
            idx=0
            while(open_hand_feature[idx][val//4] == 1):
                idx+=1
            open_hand_feature[idx][val//4] = 1
        for val in discarded_tiles_136:
            idx=0
            while(discarded_tiles_feature[idx][val//4] == 1):
                idx+=1
            discarded_tiles_feature[idx][val//4] = 1
        
        return np.concatenate((closed_hand_feature,open_hand_feature,discarded_tiles_feature))
    
    def getSelfTiles(self, tiles_state_and_action):
        player_tiles = tiles_state_and_action["player_tiles"]
        return self.getPlayerTiles(player_tiles)       

    def getEnemiesTiles(self,tiles_state_and_action):
        player_seat = tiles_state_and_action["player_id"]
        enemies_tiles_list = tiles_state_and_action["enemies_tiles"]
        if (len(enemies_tiles_list) == 3):
            enemies_tiles_list.insert(player_seat, {})
        enemies_tiles_feature = np.empty((0, 34))
        for i in range(3):
            player_seat = (player_seat + 1) % 4
            player_tiles = self.getPlayerTiles(enemies_tiles_list[player_seat])
            enemies_tiles_feature = np.concatenate((enemies_tiles_feature, player_tiles))
        return enemies_tiles_feature

    def getDoraIndicatorList(self, tiles_state_and_action):
        dora_indicator_list = tiles_state_and_action["dora"]
        dora_indicator_feature = np.zeros((5, 34))
        for idx, val in enumerate(dora_indicator_list):
            dora_indicator_feature[idx][val // 4] = 1
        return dora_indicator_feature

    def getDoraList(self, tiles_state_and_action):
        def indicator2dora(dora_indicator):
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
        dora_indicator_list = tiles_state_and_action["dora"]
        dora_feature = np.zeros((5, 34))
        for idx, dora_indicator in enumerate(dora_indicator_list):
            dora_feature[idx][indicator2dora(dora_indicator)] = 1
        return dora_feature

    def getScoreList1(self,tiles_state_and_action):
        def trans_score(score):
            feature = np.zeros((34))
            if score > 50000:
                score = 50000 
            a = score//(50000.0/33)
            # alpha = a + 1 - (score*33.0/50000)
            # alpha = round(alpha, 2)
            # feature[int(a)] = alpha
            # if a<33:
            #     feature[int(a+1)] = 1-alpha
            feature[int(a)] = 1
            return feature
            
        player_seat = tiles_state_and_action["player_id"]
        scores_list = tiles_state_and_action["scores"]
        scores_feature = np.zeros((4, 34))
        for i in range(4):
            score = scores_list[player_seat]
            scores_feature[i] = trans_score(score)
            player_seat += 1
            player_seat %= 4
        return scores_feature        
        
    def getBoard1(self, tiles_state_and_action):
        def wind(c):
            if c=='E' : 
                return 27
            if c=='S' : 
                return 28
            if c=='W' : 
                return 29
            if c=='N' : 
                return 30
        player_seat = tiles_state_and_action["player_id"]
        dealer_seat = tiles_state_and_action["dealer"]
        repeat_dealer = tiles_state_and_action["repeat_dealer"]
        riichi_bets = tiles_state_and_action["riichi_bets"]
        player_wind = tiles_state_and_action["player_wind"]
        prevailing_wind = tiles_state_and_action["prevailing_wind"]
        
        dealer_feature = np.zeros((1, 34))
        dealer_feature[0][(dealer_seat+4-player_seat)%4] = 1
        
        repeat_dealer_feature = np.zeros((1, 34))
        repeat_dealer_feature[0][repeat_dealer] = 1
        
        riichi_bets_feature = np.zeros((1, 34))
        riichi_bets_feature[0][riichi_bets] = 1
        
        player_wind_feature = np.zeros((1, 34))
        player_wind_feature[0][wind(player_wind)] = 1
        
        prevailing_wind_feature = np.zeros((1, 34))
        prevailing_wind_feature[0][wind(prevailing_wind)] = 1
        
        return np.concatenate((
            dealer_feature,
            repeat_dealer_feature,
            riichi_bets_feature,
            player_wind_feature,
            prevailing_wind_feature
        ))

    def getGeneralFeature(self, tiles_state_and_action):
        return np.concatenate((
            self.getSelfTiles(tiles_state_and_action),  # (12,34)
            self.getDoraList(tiles_state_and_action),  # (5,34)
            self.getBoard1(tiles_state_and_action),  # (5,34)
            self.getEnemiesTiles(tiles_state_and_action),  # (36,34)
            self.getScoreList1(tiles_state_and_action)  # (4,34)
        ))

    def ChiFeatureGenerator(self, tiles_state_and_action):
        """
        changed the input from filename to tiles_state_and_action data
        By Jun Lin
        """
        def _chilist(last_player_discarded_tile, closed_hand_136):
            res = []
            pairs = [(a, b) for idx, a in enumerate(closed_hand_136) for b in closed_hand_136[idx + 1:]]
            for p in pairs:
                meld = [last_player_discarded_tile,p[0],p[1]]
                if all(tile < 36 for tile in meld) or all(36 <= tile < 72 for tile in meld) or all(36 <= tile < 72 for tile in meld):
                    meld34 = [tile//4 for tile in meld]
                    if any(tile+1 in meld34 and tile-1 in meld34 for tile in meld34):
                        res.append(meld)
            return res
        could_chi = tiles_state_and_action["could_chi"]
        last_player_discarded_tile = tiles_state_and_action["last_player_discarded_tile"]
        closed_hand_136 = tiles_state_and_action["player_tiles"]['closed_hand:']
        action = tiles_state_and_action["action"]
        if could_chi == 1:
            for chimeld in _chilist(last_player_discarded_tile, closed_hand_136):
                last_player_discarded_tile_feature = np.zeros((1, 34))
                for chitile in chimeld:
                    last_player_discarded_tile_feature[0][chitile // 4] = 1
                x = np.concatenate(
                    (last_player_discarded_tile_feature, self.getGeneralFeature(tiles_state_and_action)))
                if action[0] == 'Chi' && all(chitile in action[1] for chitile in chimeld):
                    y = 1
                else:
                    y = 0
                # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
                #        "labels": to_categorical(y, num_classes=2)}
                yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)

    def PonFeatureGenerator(self, tiles_state_and_action):
        """
        changed the input from filename to tiles_state_and_action data
        By Jun Lin
        """
        last_discarded_tile = tiles_state_and_action["last_player_discarded_tile"]
        closed_hand_136 = tiles_state_and_action["player_tiles"]['closed_hand:']
        action = tiles_state_and_action["action"]
        if tiles_state_and_action["could_pon"] == 1:
            last_discarded_tile_feature = np.zeros((1, 34))
            last_discarded_tile_feature[0][last_discarded_tile // 4] = 1
            x = np.concatenate(
                (last_discarded_tile_feature, self.getGeneralFeature(tiles_state_and_action)))
            if action[0] == 'Pon':
                y = 1
            else:
                y = 0
            # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
            #        "labels": to_categorical(y, num_classes=2)}
            yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)

    def KanFeatureGenerator(self, tiles_state_and_action):
        """
        changed the input from filename to tiles_state_and_action data
        By Jun Lin
        """
        def could_ankan(closed_hand_136):
            count = np.zeros(34)
            for tile in closed_hand_136:
                count[tile//4]+=1
                if count[tile//4]==4:
                    return True
            return False

        def could_kakan(closed_hand_136, open_hand_136):
            count = np.zeros(34)
            for tile in open_hand_136:
                count[tile // 4] += 1
            for tile in closed_hand_136:
                if count[tile // 4] == 3:
                    return True
            return False

        last_discarded_tile = tiles_state_and_action["last_player_discarded_tile"]
        closed_hand_136 = tiles_state_and_action["player_tiles"]['closed_hand:']
        open_hand_136 = tiles_state_and_action["player_tiles"]['open_hand']
        action = tiles_state_and_action["action"]
        if tiles_state_and_action["could_minkan"] == 1:  # Minkan
            kan_type_feature = np.zeros((3, 34))
            kan_type_feature[0] = 1
            last_discarded_tile_feature = np.zeros((1, 34))
            last_discarded_tile_feature[0][last_discarded_tile // 4] = 1
            x = np.concatenate(
                (kan_type_feature, last_discarded_tile_feature,
                 self.getGeneralFeature(tiles_state_and_action)))

            if action[0] == 'MinKan' and (last_discarded_tile in action[1]):
                y = 1
            else:
                y = 0
            # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
            #        "labels": to_categorical(y, num_classes=2)}
            yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)
        else:
            if could_ankan(closed_hand_136):  # AnKan
                kan_type_feature = np.zeros((3, 34))
                kan_type_feature[1] = 1
                last_discarded_tile_feature = np.zeros((1, 34))
                x = np.concatenate(
                    (
                        kan_type_feature, last_discarded_tile_feature,
                        self.getGeneralFeature(tiles_state_and_action)))

                if action[0] == 'AnKan':
                    y = 1
                else:
                    y = 0
                # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
                #        "labels": to_categorical(y, num_classes=2)}
                yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)
            else:
                if could_kakan(closed_hand_136, open_hand_136):  # KaKan
                    kan_type_feature = np.zeros((3, 34))
                    kan_type_feature[2] = 1
                    last_discarded_tile_feature = np.zeros((1, 34))
                    x = np.concatenate(
                        (kan_type_feature, last_discarded_tile_feature,
                         self.getGeneralFeature(tiles_state_and_action)))
                    if action[0] == 'KaKan':
                        y = 1
                    else:
                        y = 0
                    # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
                    #        "labels": to_categorical(y, num_classes=2)}
                    yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)

    def RiichiFeatureGenerator(self,tiles_state_and_action):
        action = tiles_state_and_action["action"]
        if tiles_state_and_action["is_FCH"] == 1:
            tiles_34 = TilesConverter.to_34_array(tiles_state_and_action["player_tiles"]["closed_hand:"])
            min_shanten = self.shanten_calculator.calculate_shanten(tiles_34)
            if min_shanten == 0:
                x = self.getGeneralFeature(tiles_state_and_action)
                if action[0] == 'REACH':
                    y = 1
                else:
                    y = 0
                # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
                #        "labels": to_categorical(y, num_classes=2)}
                yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)

if __name__ == "__main__":
    filename = "assist/chi_pon_kan_reach_2021.json"
    fg = FeatureGenerator()
    with open(filename) as infile:
        for line in infile:
            tiles_state_and_action = json.loads(line)
            fg.ChiFeatureGenerator(tiles_state_and_action)
            fg.PonFeatureGenerator(tiles_state_and_action)
            fg.KanFeatureGenerator(tiles_state_and_action)
            fg.RiichiFeatureGenerator(tiles_state_and_action)
