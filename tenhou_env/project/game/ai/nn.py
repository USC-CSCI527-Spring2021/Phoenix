from __future__ import absolute_import

import os
import random

import numpy as np
from mahjong.utils import is_aka_dora
from tensorflow import keras
from utils.decisions_logger import MeldPrint
from game.ai.exp_buffer import ExperienceCollector
from game.ai.models import rcpk_model, discard_model


def getGeneralFeature(player, additional_data = None):
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

        if additional_data and additional_data.has_key("open_hands_136"):
            open_hand = additional_data["open_hands_136"]
        else:
            open_hand = [tile for meld in player.melds for tile in meld.tiles]
        if additional_data and additional_data.has_key("closed_hands_136"):
            closed_hand = additional_data["closed_hands_136"]
        else:
            closed_hand = player.closed_hand #[tile for tile in player.tiles if tile not in open_hand]
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
        return np.concatenate((
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
        player_wind_feature[0][27 + (4 - player.table.dealer_seat) % 4] = 1

        prevailing_wind_feature = np.zeros((1, 34))
        prevailing_wind_feature[0][27 + player.table.round_wind_number // 4] = 1

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
        self.input_shape = (63, 34, 1)
        # self.strategy = 'local'        # fix here
        # self.model = models.make_or_restore_model(self.input_shape, "chi", self.strategy)
        # load models from current working dir
        if 'chi' not in player.config:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'chi'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['chi'])
        print('###### Chi model initialized #######')
        self.collector = ExperienceCollector('chi', player.config.buffer)
        self.collector.start_episode()

    def should_call_chi(self, tile_136, melds_chi):
        features = self.getFeature(melds_chi)
        predictions = self.model.predict(features)
        pidx = np.argmax(predictions[:,1])
        choice = np.argmax(predictions[pidx])
        actions = np.eye(predictions.shape[-1])[choice]
        self.collector.record_decision(features, actions, predictions)
        if choice == 0:
            print("Chi Model choose not to chi")
            return False, predictions[pidx][choice], melds_chi[pidx]
        else:
            print("Chi Model choose to chi")
            return True, predictions[pidx][choice], melds_chi[pidx]

    def getFeature(self, melds_chi):
        def _get_x(meld_chi):
            tile_136_feature = np.zeros((1, 34))
            for tile in meld_chi:
                tile_136_feature[0][tile//4] = 1
            x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
            return np.expand_dims(x.reshape((x.shape[0], x.shape[1], 1)), axis=0)
        return np.concatenate([_get_x(meld_chi) for meld_chi in melds_chi],axis=0)

class Pon:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        self.input_shape = (63, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "pon", self.strategy)
        # load models from current working dir
        if 'pon' not in player.config:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'pon'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['pon'])

        print('###### Pon model initialized #######')
        self.collector = ExperienceCollector('pon', player.config.buffer)

    def should_call_pon(self, tile_136, is_kamicha_discard):
        features = self.getFeature(tile_136)
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        choice = np.argmax(predictions)
        actions = np.eye(predictions.shape[-1])[choice]
        self.collector.record_decision(features, actions, predictions)
        if choice == 0:
            print("Pon Model choose not to pon")
            return False, predictions[choice]
        else:
            print("Pon Model choose to pon")
            return True, predictions[choice]

    def getFeature(self, tile_136):
        tile_136_feature = np.zeros((1, 34))
        tile_136_feature[0][tile_136 // 4] = 1
        x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))

class Kan:
    def __init__(self, player):
        self.player = player
        self.input_shape = (66, 34, 1)
        # self.strategy = 'local'
        # self.model = models.make_or_restore_model(self.input_shape, "kan", self.strategy)
        # load models from current working dir
        if 'kan' not in player.config:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'kan'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['kan'])

        print('###### Kan model initialized #######')
        self.collector = ExperienceCollector('kan', player.config.buffer)

    def should_call_kan(self, tile_136, open_kan, from_riichi=False):
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
            if count[tile // 4] == 4:
                return True
            return False

        def _can_kakan(tile, melds):
            for meld in melds:
                if (meld.type == MeldPrint.PON) and (tile // 4 == meld.tiles[0] // 4):
                    return True
            return False
        kan_type_feature = np.zeros((3, 34))
        closed_hand = self.player.closed_hand
        features = self.getFeature(tile_136, kan_type_feature)
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        model_predict = np.argmax(predictions)
        if not open_kan:
            if _can_kakan(tile_136, self.player.melds):  # KaKan
                kan_type_feature[2] = 1
                kan_type = MeldPrint.SHOUMINKAN
                if model_predict:
                    print("Kan model choose to ", kan_type)
                else:
                    print("Kan model choose not to ", kan_type)
            elif _can_ankan(tile_136, closed_hand):  # AnKan
                kan_type_feature[1] = 1
                kan_type = MeldPrint.KAN
                if model_predict:
                    print("Kan model choose to ", kan_type)
                else:
                    print("Kan model choose not to ", kan_type)
            else:
                return None
        else: 
            if _can_minkan(tile_136, closed_hand):#MinKan
                kan_type_feature[0] = 1
                kan_type = MeldPrint.SHOUMINKAN
                if model_predict:
                    print("Kan model choose to ", kan_type)
                else:
                    print("Kan model choose not to ", kan_type)
            else:
                return None
        actions = np.eye(predictions.shape[-1])[model_predict]
        self.collector.record_decision(features, actions, predictions)
        return kan_type 

    def getFeature(self, tile136, kan_type_feature):
        tile136_feature = np.zeros((1, 34))
        tile136_feature[0][tile136 // 4] = 1        
        x = np.concatenate((kan_type_feature, tile136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))

class Riichi:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        self.input_shape = (62, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "riichi", self.strategy)
        # load models from current working dir
        if 'riichi' not in player.config:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'riichi'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['riichi'])

        print('###### Riichi model initialized #######')
        self.collector = ExperienceCollector('riichi', player.config.buffer)

    def should_call_riichi(self):
        features = self.getFeature()
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        choice = np.argmax(predictions)
        actions = np.eye(predictions.shape[-1])[choice]
        self.collector.record_decision(features, actions, predictions)
        if choice == 0:
            print("Riichi Model choose not to riichi")
            return False, predictions[choice]
        else:
            print("Riichi Model choose to riichi")
            return True, predictions[choice]

    def getFeature(self):
        x = getGeneralFeature(self.player)
        return x.reshape((x.shape[0], x.shape[1], 1))

class Discard:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        self.input_shape = (16, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "discard", self.strategy)
        # load models from current working dir
        if 'discard' not in player.config:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'discard'))
        else:
            self.model = discard_model(self.input_shape)
            self.model.set_weights(player.config.weights['discard'])

        print('###### Discarded model initialized #######')
        self.collector = ExperienceCollector('discard', player.config.buffer)

    def discard_tile(self, all_hands_136=None, closed_hands_136=None, with_riichi_options=None):
        # discard_options
        if with_riichi_options:
            discard_options = with_riichi_options
        else:
            discard_options = closed_hands_136 if closed_hands_136 else self.player.closed_hand

        # get feature and prediction
        features = self.getFeature(all_hands_136, closed_hands_136)
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        max_score = 0
        choice = discard_options[0] # type: tile_136
        for option in discard_options:
            score = predictions[option // 4]
            if score > max_score or (score == max_score and is_aka_dora(choice, True)):
                max_score = score
                choice = option
        print("Discarded Model:", "discard", option, "from", closed_hands_136 if closed_hands_136 else self.player.closed_hand)
        actions = np.eye(predictions.shape[-1])[choice // 4]
        self.collector.record_decision(features, actions, predictions)
        return choice

    def getFeature(self, all_hands_136=None, closed_hands_136=None):
        open_hands_136 = [tile for tile in all_hands_136 if tile not in closed_hands_136]
        x = getGeneralFeature(self.player, {"open_hands_136":open_hands_136, "closed_hands_136":closed_hands_136})
        return x.reshape((x.shape[0], x.shape[1], 1))

class GlobalRewardPredictor:
    def __init__(self):
        self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'grp'))
        # self.model.summary()


    def get_global_reward(self, features):
        '''
        input shape: (None, constants.round_num, constants.pred_dim)/(None, 15, 15)
        '''
        
        # init_score = np.asarray(init_score)/1e5
        # gains = np.asarray(gains)/1e5
        # embed = np.expand_dims(np.concatenate((init_score, gains, dan, dealer, repeat_dealer, riichi_bets), axis=-1), axis=0)
        # # print('input_shape', embed.shape)
        # assert embed.shape[-2:] == (15, 15)
        return self.model.predict(features)[0]

# class Discard:
#     def __init__(self, player):
#         self.player = player
#         # self.strategy = 'local'
#         # self.input_shape = (16, 34, 1)
#         # self.model = models.make_or_restore_model(self.input_shape, "discard", self.strategy)
#         # load models from current working dir
#         self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'discarded'))
#         print('###### Discarded model initialized #######')
#         self.collector = ExperienceCollector('discard')

#     def discard_tile(self, all_hands_136=None, closed_hands_136=None, with_riichi=False):
#         '''
#         The reason why these two should be input:
#         if the "discard" action is from "discard after meld", since we have decided to meld, what
#         to do next is to decide what to discard, but the meld has not happened (and could also be 
#         interrupted), so we could only use discard model based on supposed hands after melding

#         '''
#         # random output a tile
#         # tile_to_discard_136 = random.choice(self.player.closed_hand)
#         features = self.getFeature(all_hands_136, closed_hands_136, with_riichi)
#         tile_to_discard = np.argmax(self.model.predict(np.expand_dims(features, axis=0))[0])
#         tile_to_discard_136 = [h for h in self.player.closed_hand if h // 4 == tile_to_discard]
#         akadora = [tile for tile in tile_to_discard_136 if is_aka_dora(tile, True)]
#         # No discard candidate
#         if not tile_to_discard_136:
#             ran = random.choice(self.player.closed_hand)
#             print("Discarded Model predict {} not in closed hand.\nRandom choose {} from {} to discard".format(
#                 tile_to_discard * 4, ran, self.player.closed_hand))
#             self.collector.record_decision(features, np.eye(34)[ran // 4])
#             return ran
#         # only if this game has akadora
#         if self.player.table.has_aka_dora and akadora:
#             if len(tile_to_discard_136) > 1:
#                 # if multiple tiles exists, return the one which is not red dora
#                 tile_to_discard_136.remove(akadora[-1])
#                 print("Multiple tiles exists, AkaDora:", akadora[-1], "Discard the non red dora:",
#                       tile_to_discard_136[0])
#                 self.collector.record_decision(features, np.eye(34)[tile_to_discard_136[0]])
#                 return tile_to_discard_136[0]
#             else:
#                 print("Discard the AkaDora:", tile_to_discard_136[0])
#                 self.collector.record_decision(features, np.eye(34)[tile_to_discard_136[0]])
#                 return tile_to_discard_136[0]
#         else:
#             print("Discarded Model:", "discard", tile_to_discard_136[-1], "from", self.player.closed_hand)
#             self.collector.record_decision(features, np.eye(34)[tile_to_discard_136[-1]])
#             return tile_to_discard_136[-1]

#     def getFeature(self, open_hands_136, closed_hands_136, with_riichi):
#         from trainer.models import transform_discard_features
#         data = {
#             "draw_tile": self.player.tiles[-1],
#             "hands": self.player.tiles,
#             "discarded_tiles_pool": self.player.table.revealed_tiles_136,
#             "four_players_open_hands": self.player.table.melded_tiles,
#             ## fake data here, pass in just to get function running
#             "discarded_tile": 0,
#         }
#         # if hands are none, get hands from self.player
#         return transform_discard_features(data)['features']  # change this according to actual shape
