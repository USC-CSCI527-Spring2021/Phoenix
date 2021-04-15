# Author: Jingwen Sun
# This is a class used for extract features from a tiles_state_and_action list for chi/pon/kan/riichi model.
# usage:
#   at the bottom of this file

import hashlib
import json
import marshal

import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, HandConstants
from mahjong.hand_calculating.scores import ScoresCalculator
from mahjong.meld import Meld
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from tensorflow.keras.utils import to_categorical

set_loky_pickler('dill')


class FeatureGenerator:
    def __init__(self):
        """
        changed the input from filename to tiles_state_and_action data
        By Jun Lin
        """
        self.shanten_calculator = Shanten()
        self.hc = HandCalculator()
        self.sc = ScoresCalculator()
        self.hand_cache_shanten = {}
        self.hand_cache_points = {}

    def build_cache_key(self, tiles_34):
        return hashlib.md5(marshal.dumps(tiles_34)).hexdigest()

    def calculate_shanten_or_get_from_cache(self, closed_hand_34):
        key = self.build_cache_key(closed_hand_34)
        if key in self.hand_cache_shanten:
            return self.hand_cache_shanten[key]
        result = self.shanten_calculator.calculate_shanten(closed_hand_34)
        self.hand_cache_shanten[key] = result
        return result

    def calculate_ponits_or_get_from_cache(self, closed_left_tiles_34,win_tile,melds,dora_indicators):
        tiles_34 = closed_left_tiles_34[:]
        for meld in melds:
            for x in meld.tiles_34:
                tiles_34[x] += 1
        key = self.build_cache_key(tiles_34+[win_tile]+dora_indicators)
        if key in self.hand_cache_points:
            return self.hand_cache_points[key]
        # print(closed_left_tiles_34)
        # print(melds)
        hc_result = self.hc.estimate_hand_value(TilesConverter.to_136_array(tiles_34),win_tile,melds,dora_indicators)
        sc_result = self.sc.calculate_scores(hc_result.han, hc_result.fu, HandConfig(HandConstants()))
        result = sc_result["main"]
        self.hand_cache_points[key] = result
        return result

    def canwinbyreplace(self, closed_left_tiles_34, melds, dora_indicators, tiles_could_draw, replacelimit):
        def _draw(closed_left_tiles_34, melds, dora_indicators, tiles_could_draw, replacelimit):
            if self.calculate_shanten_or_get_from_cache(closed_left_tiles_34) > replacelimit:
                return 0
            result = 0
            if replacelimit == 0: 
                for idx in range(34):
                    if tiles_could_draw[idx] > 0:
                        closed_left_tiles_34[idx] += 1
                        if self.calculate_shanten_or_get_from_cache(closed_left_tiles_34) == -1:
                            ponits = self.calculate_ponits_or_get_from_cache(closed_left_tiles_34,idx*4,melds,dora_indicators)
                            result = max(result,ponits)
                        closed_left_tiles_34[idx] -= 1
                
            else:
                for idx,count in enumerate(tiles_could_draw):
                    if count > 0:
                        tiles_could_draw[idx] -= 1
                        closed_left_tiles_34[idx] += 1
                        ponits = _discard(closed_left_tiles_34, melds, dora_indicators, tiles_could_draw, replacelimit)
                        result = max(result,ponits)
                        closed_left_tiles_34[idx] -= 1
                        tiles_could_draw[idx] += 1
            return result
        
        def _discard(closed_left_tiles_34, melds, dora_indicators, tiles_could_draw, replacelimit):
            result = 0
            for idx,count in enumerate(closed_left_tiles_34):
                if count > 0:
                    closed_left_tiles_34[idx] -= 1
                    replacelimit -= 1
                    ponits = _draw(closed_left_tiles_34, melds, dora_indicators, tiles_could_draw, replacelimit)
                    result = max(result,ponits)
                    replacelimit += 1
                    closed_left_tiles_34[idx] += 1
            return result

        return _draw(closed_left_tiles_34, melds, dora_indicators, tiles_could_draw, replacelimit)

    def open_hands_detail_to_melds(self, open_hands_detail):
        melds = []
        for ohd in open_hands_detail:
            tiles = ohd["tiles"]
            if ohd["meld_type"] == "Pon":
                meld_type = "pon"
                opened = True
            elif ohd["meld_type"] == "Chi":
                meld_type = "chi"
                opened = True
            elif ohd["meld_type"] == "AnKan":
                meld_type = "kan"
                opened = False 
            else:
                meld_type = "kan"
                opened = True
            meld = Meld(meld_type,tiles,opened)
            melds.append(meld)
        return melds

    def getPlayerTiles(self, player_tiles):
        closed_hand_136 = player_tiles.get('closed_hand:',[])
        open_hand_136 = player_tiles.get('open_hand',[])
        discarded_tiles_136 = player_tiles.get('discarded_tiles',[])

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
            enemies_tiles_list.insert(player_seat, tiles_state_and_action["player_tiles"])
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

    def getLookAheadFeature(self, tiles_state_and_action):
        # 0 for whether can be discarded
        # 1 2 3 for shanten
        # 4 5 6 7 for whether can get 2k 4k 6k 8k points with replacing 3 tiles ---- need review: takes too long!
        # 8 9 10 for in Shimocha Toimen Kamicha discarded
        # lookAheadFeature = np.zeros((11, 34))
        player_tiles = tiles_state_and_action["player_tiles"]
        closed_hand_136 = player_tiles.get('closed_hand:',[])
        open_hands_detail = tiles_state_and_action["open_hands_detail"]
        melds = self.open_hands_detail_to_melds(open_hands_detail)
        discarded_tiles_136 = player_tiles.get('discarded_tiles',[])
        player_seat = tiles_state_and_action["player_id"]
        enemies_tiles_list = tiles_state_and_action["enemies_tiles"]
        dora_indicators = tiles_state_and_action["dora"]
        if (len(enemies_tiles_list) == 3):
            enemies_tiles_list.insert(player_seat, player_tiles)
        tiles_could_draw = np.ones(34) * 4
        for player in enemies_tiles_list:
            for tile_set in [player.get('closed_hand:',[]), player.get('open_hand',[]), player.get('discarded_tiles',[])]:
                for tile in tile_set:
                    tiles_could_draw[tile//4] -= 1
        for dora_tile in dora_indicators:
            tiles_could_draw[tile//4] -= 1

        def feature_process(i, closed_hand_136, melds, dora_indicators, tiles_could_draw, player_seat, enemies_tiles_list):
            feature = np.zeros((11, 1))
            discard_tiles = [x for x in [i*4,i*4+1,i*4+2,i*4+3] if x in closed_hand_136]
            if len(discard_tiles) != 0 :
                discard_tile = discard_tiles[0]
                feature[0] = 1
                closed_left_tiles_34 = TilesConverter.to_34_array([t for t in closed_hand_136 if t != discard_tile])
                shanten = self.calculate_shanten_or_get_from_cache(closed_left_tiles_34)
                for i in range(3):
                    if shanten <= i:
                        feature[i+1] = 1
                maxscore = self.canwinbyreplace(closed_left_tiles_34, melds, dora_indicators,
                                                tiles_could_draw, replacelimit=2)
                scores = [2000, 4000, 6000, 8000]
                for i in range(4):
                    if maxscore >= scores[i]:
                        feature[i + 4] = 1
                seat = player_seat
                for i in range(3):
                    seat = (seat + 1) % 4
                    if discard_tile // 4 in [t // 4 for t in enemies_tiles_list[seat].get('discarded_tiles', [])]:
                        feature[i + 8] = 1
            return feature

        # results = [feature_process(i, closed_hand_136, melds, dora_indicators, tiles_could_draw, player_seat, enemies_tiles_list) for i in range(34)]
        results = Parallel(n_jobs=-1)(
            delayed(feature_process)(i, closed_hand_136, melds, dora_indicators, tiles_could_draw, player_seat,
                                     enemies_tiles_list)
            for i in range(34))
        return np.concatenate(results, axis=1)

    def getGeneralFeature(self, tiles_state_and_action):
        return np.concatenate((
            self.getLookAheadFeature(tiles_state_and_action), #(11,34)
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
                if action[0] == 'Chi' and all(chitile in action[1] for chitile in chimeld):
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
            # min_shanten = self.shanten_calculator.calculate_shanten(tiles_34)
            min_shanten = self.calculate_shanten_or_get_from_cache(tiles_34)
            if min_shanten == 0:
                x = self.getGeneralFeature(tiles_state_and_action)
                if action[0] == 'REACH':
                    y = 1
                else:
                    y = 0
                # yield {'features': x.reshape((x.shape[0], x.shape[1], 1)),
                #        "labels": to_categorical(y, num_classes=2)}
                yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=2)

    def DiscardFeatureGenerator(self,tiles_state_and_action):
        x = self.getGeneralFeature(tiles_state_and_action)
        y = tiles_state_and_action["discarded_tile"]//4
        yield x.reshape((x.shape[0], x.shape[1], 1)), to_categorical(y, num_classes=34)


if __name__ == "__main__":
    filename = "assist/chi_pon_kan_reach_2021.json"
    fg = FeatureGenerator()
    with open(filename) as infile:
        for line in infile:
            tiles_state_and_action = json.loads(line)
            fg.ChiFeatureGenerator(tiles_state_and_action) #(74,34)
            fg.PonFeatureGenerator(tiles_state_and_action) #(74,34)
            fg.KanFeatureGenerator(tiles_state_and_action) #(77,34)
            fg.RiichiFeatureGenerator(tiles_state_and_action) #(73,34)
            fg.DiscardFeatureGenerator(tiles_state_and_action) #(73,34)
