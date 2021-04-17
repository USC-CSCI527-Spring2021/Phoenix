from __future__ import absolute_import

import os
import time

import numpy as np
import utils.decisions_constants as log
from game.ai.exp_buffer import ExperienceCollector
from game.ai.models import rcpk_model, discard_model
from mahjong.utils import is_aka_dora
from tensorflow import keras
from utils.decisions_logger import MeldPrint


def getGeneralFeature(player, additional_data=None):
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

        if additional_data and "open_hands_136" in additional_data:
            open_hand = additional_data["open_hands_136"]
        else:
            open_hand = [tile for meld in player.melds for tile in meld.tiles]
        if additional_data and "closed_hands_136" in additional_data:
            closed_hand = additional_data["closed_hands_136"]
        else:
            closed_hand = player.closed_hand  # [tile for tile in player.tiles if tile not in open_hand]
        discarded_tiles = player.discards

        for val in closed_hand:
            idx = 0
            while (closed_hand_feature[idx][val // 4] == 1):
                idx += 1
            closed_hand_feature[idx][val // 4] = 1
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

        return np.concatenate((closed_hand_feature, open_hand_feature, discarded_tiles_feature))

    def _getSelfTiles(player):
        return _getPlayerTiles(player)

    def _getEnemiesTiles(player):
        """
        If online, all the enemies tiles will be zeros(12,34), otherwise get their hands infos
        :param player:
        :return:
        """
        if player.config.isOnline:
            return np.zeros((12, 34))
        else:
            return np.concatenate((
                _getPlayerTiles(player.players[1]),
                _getPlayerTiles(player.players[2]),
                _getPlayerTiles(player.players[3])
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
        dealer_feature[0][(dealer_seat + 4 - player_seat) % 4] = 1

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
        _getSelfTiles(player),  # (12,34)
        _getDoraList(player),  # (5,34)
        _getBoard1(player),  # (5,34)
        _getEnemiesTiles(player),  # (36,34)
        _getScoreList1(player)  # (4,34)
    ))


class Chi:
    def __init__(self, player):
        self.player = player
        self.input_shape = (63, 34, 1)
        # self.strategy = 'local'        # fix here
        # self.model = models.make_or_restore_model(self.input_shape, "chi", self.strategy)
        # load models from current working dir
        if 'chi' not in player.config.weights:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'chi'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['chi'])
        # print('###### Chi model initialized #######')
        self.collector = ExperienceCollector('chi', player.config.buffer)
        self.collector.start_episode()

    def should_call_chi(self, tile_136, melds_chi):
        features = self.getFeature(melds_chi)
        start_time = time.time()
        predictions = self.model.predict(features)
        print("---Chi inference time:  %s seconds ---" % (time.time() - start_time))

        pidx = np.argmax(predictions[:, 1])
        choice = np.argmax(predictions[pidx])
        actions = np.eye(predictions.shape[-1])[choice]
        self.collector.record_decision(features, actions, predictions)
        if choice == 0:
            self.player.logger.debug(
                log.MELD_CALL,
                "Chi Model choose not to chi",
                context=[
                    f"Hand: {self.player.format_hand_for_print(tile_136)}",
                ],
            )
            return False, predictions[pidx][choice], melds_chi[pidx]
        else:
            self.player.logger.debug(
                log.MELD_CALL,
                "Chi Model choose to chi",
                context=[
                    f"Hand: {self.player.format_hand_for_print(tile_136)}",
                ],
            )
            return True, predictions[pidx][choice], melds_chi[pidx]

    def getFeature(self, melds_chi):
        def _get_x(meld_chi):
            tile_136_feature = np.zeros((1, 34))
            for tile in meld_chi:
                tile_136_feature[0][tile // 4] = 1
            x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
            return np.expand_dims(x.reshape((x.shape[0], x.shape[1], 1)), axis=0)

        return np.concatenate([_get_x(meld_chi) for meld_chi in melds_chi], axis=0)


class Pon:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        self.input_shape = (63, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "pon", self.strategy)
        # load models from current working dir
        if 'pon' not in player.config.weights:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'pon'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['pon'])

        # print('###### Pon model initialized #######')
        self.collector = ExperienceCollector('pon', player.config.buffer)

    def should_call_pon(self, tile_136, is_kamicha_discard):
        features = self.getFeature(tile_136)
        start_time = time.time()
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        print("---Pon inference time:  %s seconds ---" % (time.time() - start_time))
        choice = np.argmax(predictions)
        actions = np.eye(predictions.shape[-1])[choice]

        self.collector.record_decision(features, actions, predictions)
        if choice == 0:
            self.player.logger.debug(
                log.MELD_CALL,
                "Pon Model choose not to pon",
                context=[
                    f"Hand: {self.player.format_hand_for_print(tile_136)}",
                ],
            )
            return False, predictions[choice]
        else:
            self.player.logger.debug(
                log.MELD_CALL,
                "Pon Model choose to pon",
                context=[
                    f"Hand: {self.player.format_hand_for_print(tile_136)}",
                ],
            )
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
        if 'kan' not in player.config.weights:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'kan'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['kan'])

        # print('###### Kan model initialized #######')
        self.collector = ExperienceCollector('kan', player.config.buffer)

    def should_call_kan(self, tile_136, open_kan, from_riichi=False):
        def _can_minkan(tile, closed_hand):
            count = np.zeros(34)
            for t in closed_hand:
                count[t // 4] += 1
            if count[tile // 4] == 3:
                return True
            return False

        def _can_ankan(tile, closed_hand):
            count = np.zeros(34)
            for t in closed_hand:
                count[t // 4] += 1
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
        start_time = time.time()
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        print("---Chi inference time:  %s seconds ---" % (time.time() - start_time))

        model_predict = np.argmax(predictions)
        if not open_kan:
            if _can_kakan(tile_136, self.player.melds):  # KaKan
                kan_type_feature[2] = 1
                kan_type = MeldPrint.SHOUMINKAN
                if model_predict:
                    self.player.logger.debug(
                        log.MELD_CALL,
                        "Kan Model choose to " + kan_type,
                        context=[
                            f"Hand: {self.player.format_hand_for_print(tile_136)}",
                        ],
                    )
                else:
                    self.player.logger.debug(
                        log.MELD_CALL,
                        "Kan Model choose not to " + kan_type,
                        context=[
                            f"Hand: {self.player.format_hand_for_print(tile_136)}",
                        ],
                    )
            elif _can_ankan(tile_136, closed_hand):  # AnKan
                kan_type_feature[1] = 1
                kan_type = MeldPrint.KAN
                if model_predict:
                    self.player.logger.debug(
                        log.MELD_CALL,
                        "Kan Model choose to " + kan_type,
                        context=[
                            f"Hand: {self.player.format_hand_for_print(tile_136)}",
                        ],
                    )
                else:
                    self.player.logger.debug(
                        log.MELD_CALL,
                        "Kan Model choose not to " + kan_type,
                        context=[
                            f"Hand: {self.player.format_hand_for_print(tile_136)}",
                        ],
                    )
            else:
                return None
        else:
            if _can_minkan(tile_136, closed_hand):  # MinKan
                kan_type_feature[0] = 1
                kan_type = MeldPrint.SHOUMINKAN
                if model_predict:
                    self.player.logger.debug(
                        log.MELD_CALL,
                        "Kan Model choose to " + kan_type,
                        context=[
                            f"Hand: {self.player.format_hand_for_print(tile_136)}",
                        ],
                    )
                else:
                    self.player.logger.debug(
                        log.MELD_CALL,
                        "Kan Model choose not to " + kan_type,
                        context=[
                            f"Hand: {self.player.format_hand_for_print(tile_136)}",
                        ],
                    )
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
        if 'riichi' not in player.config.weights:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'riichi'))
        else:
            self.model = rcpk_model(self.input_shape)
            self.model.set_weights(player.config.weights['riichi'])

        # print('###### Riichi model initialized #######')
        self.collector = ExperienceCollector('riichi', player.config.buffer)

    def should_call_riichi(self):
        features = self.getFeature()
        start_time = time.time()
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        print("---Riichi inference time:  %s seconds ---" % (time.time() - start_time))
        choice = np.argmax(predictions)
        actions = np.eye(predictions.shape[-1])[choice]
        self.collector.record_decision(features, actions, predictions)
        if choice == 0:
            self.player.logger.debug(
                log.MELD_CALL,
                "Riichi Model choose not to riichi",
                context=[
                    f"Hand: {self.player.format_hand_for_print()}",
                ],
            )
            return False, predictions[choice]
        else:
            self.player.logger.debug(
                log.MELD_CALL,
                "Riichi Model choose to riichi",
                context=[
                    f"Hand: {self.player.format_hand_for_print()}",
                ],
            )
            return True, predictions[choice]

    def getFeature(self):
        x = getGeneralFeature(self.player)
        return x.reshape((x.shape[0], x.shape[1], 1))


class Discard:
    def __init__(self, player):
        self.player = player
        # self.strategy = 'local'
        self.input_shape = (62, 34, 1)
        # self.model = models.make_or_restore_model(self.input_shape, "discard", self.strategy)
        # load models from current working dir
        if 'discarded' not in player.config.weights:
            self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'discarded'))
        else:
            self.model = discard_model(self.input_shape)
            self.model.set_weights(player.config.weights['discarded'])
        # print('###### Discarded model initialized #######')
        self.collector = ExperienceCollector('discard', player.config.buffer)

    def discard_tile(self, all_hands_136=None, closed_hands_136=None, with_riichi_options=None):
        # discard_options
        if with_riichi_options:
            discard_options = with_riichi_options
        else:
            discard_options = closed_hands_136 if closed_hands_136 else self.player.closed_hand

        # get feature and prediction

        if not all_hands_136:
            all_hands_136 = self.player.tiles
            closed_hands_136 = self.player.closed_hand

        features = self.getFeature(all_hands_136, closed_hands_136)
        start_time = time.time()
        predictions = self.model.predict(np.expand_dims(features, axis=0))[0]
        print("---Discard inference time:  %s seconds ---" % (time.time() - start_time))

        max_score = 0
        choice = discard_options[0]  # type: tile_136
        for option in discard_options:
            score = predictions[option // 4]
            if score > max_score or (score == max_score and is_aka_dora(choice, True)):
                max_score = score
                choice = option
        self.player.logger.debug(log.DISCARD, context="Discard Model: discard {} from {}"
                                 .format(option, closed_hands_136 if closed_hands_136 else self.player.closed_hand))
        actions = np.eye(predictions.shape[-1])[choice // 4]
        self.collector.record_decision(features, actions, predictions)
        return choice

    def getFeature(self, all_hands_136, closed_hands_136):
        '''
        Here all_hands_136 and closed_hands_136 are "virtual" results after meld.
        Since "discard_tile" could be called in try_to_call_meld, and your meld choice could be
        interruptted. Like chi is interruptted by other's pon. So you can only use virtual results
        instead of current table data
        '''

        open_hands_136 = [tile for tile in all_hands_136 if tile not in closed_hands_136]
        x = getGeneralFeature(self.player, {"open_hands_136": open_hands_136, "closed_hands_136": closed_hands_136})
        return x.reshape((x.shape[0], x.shape[1], 1))

class GlobalRewardPredictor:
    def __init__(self):
        self.model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'grp'))
        # print("###### Global Reward Predictor model initialized #######")
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
        start_time = time.time()
        reward = self.model.predict(features)[0]
        print("---Global Reward Predictor inference time:  %s seconds ---" % (time.time() - start_time))
        return reward
