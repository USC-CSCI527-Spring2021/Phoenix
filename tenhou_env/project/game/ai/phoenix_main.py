from typing import List
import numpy as np

import utils.decisions_constants as log
from game.ai.hand_builder import HandBuilder
from game.ai.nn import Chi, Pon, Kan, Riichi, Discard, GlobalRewardPredictor
from mahjong.constants import DISPLAY_WINDS
from mahjong.hand_calculating.divider import HandDivider
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.utils import is_chi, is_man, is_pin, is_pon, is_sou
from utils.cache import build_estimate_hand_value_cache_key, build_shanten_cache_key
from game.ai.utils import RANKS, pred_emb_dim, round_num
import logging

game_logger = logging.getLogger("game")

class Phoenix:
    def __init__(self, player):
        self.player = player
        self.table = player.table

        self.chi = Chi(player)
        self.pon = Pon(player)
        self.kan = Kan(player)
        self.riichi = Riichi(player)
        self.discard = Discard(player)
        self.grp = GlobalRewardPredictor()
        self.hand_builder = HandBuilder(player, self)
        self.shanten_calculator = Shanten()
        self.hand_cache_shanten = {}
        self.placement = player.config.PLACEMENT_HANDLER_CLASS(player)
        self.finished_hand = HandCalculator()
        self.hand_divider = HandDivider()

        self.erase_state()

    def erase_state(self):
        self.hand_cache_shanten = {}
        self.hand_cache_estimation = {}
        self.finished_hand = HandCalculator()
        self.grp_features = []

    def collect_experience(self):

        # collect round info
        init_scores = np.array([p.init_score for p in self.table.players]) / 1e5
        gains = np.array(self.table.gains) / 1e5
        if self.player.config.isOnline:
            dans = np.array([RANKS.index(p.rank) for p in self.player.table.players])
        else:
            dans = np.array([0, 0, 0, 0])
        dealer = int(self.player.dealer_seat)
        repeat_dealer = self.player.table.count_of_honba_sticks
        riichi_bets = self.player.table.count_of_riichi_sticks

        features = np.concatenate((init_scores, gains, dans, np.array([dealer, repeat_dealer, riichi_bets])), axis=0)
        self.grp_features.append(features)

        #prepare input
        grp_input = [np.zeros(pred_emb_dim)] * max(round_num-len(self.grp_features), 0) + self.grp_features[:]

        reward = self.grp.get_global_reward(np.expand_dims(np.asarray(grp_input), axis=0))[self.player.seat]

        for model in [self.chi, self.pon, self.kan, self.riichi, self.discard]:
            model.collector.complete_episode(reward)

    
    def write_buffer(self):
        for model in [self.chi, self.pon, self.kan, self.riichi, self.discard]:
            model.collector.to_buffer()

    def init_hand(self):
        self.player.logger.debug(
            log.INIT_HAND,
            context=[
                f"Round  wind: {DISPLAY_WINDS[self.table.round_wind_tile]}",
                f"Player wind: {DISPLAY_WINDS[self.player.player_wind]}",
                f"Hand: {self.player.format_hand_for_print()}",
            ],
        )
        self.shanten, _ = self.hand_builder.calculate_shanten_and_decide_hand_structure(
            TilesConverter.to_34_array(self.player.tiles)
        )

    def draw_tile(self):
        pass

    def discard_tile(self, discard_tile):
        '''
        return discarded_tile and with_riichi
        '''
        if discard_tile is not None:    #discard after meld
            return discard_tile, False
        if self.player.is_open_hand:    #can not riichi
            return self.discard.discard_tile(), False

        shanten = self.calculate_shanten_or_get_from_cache(TilesConverter.to_34_array(self.player.closed_hand))      
        if shanten != 0:                #can not riichi
            return self.discard.discard_tile(), False

        # if not self.player.in_tempai:
            # game_logger.info(' '.join([str(i) for i in self.player.closed_hand]))
            # return self.discard.discard_tile(), False

        with_riichi, p = self.riichi.should_call_riichi()
        if with_riichi:
            # fix here: might need review
            riichi_options = [tile for tile in self.player.closed_hand if self.calculate_shanten_or_get_from_cache(TilesConverter.to_34_array([t for t in self.player.closed_hand if t != tile])) == 0]
            tile_to_discard = self.discard.discard_tile(with_riichi_options=riichi_options)
        else:
            tile_to_discard = self.discard.discard_tile()
        return tile_to_discard, with_riichi

    def try_to_call_meld(self, tile_136, is_kamicha_discard, meld_type):
        # 1 pon
        # 2 kan (it is a closed kan and can be send only to the self draw)
        # 4 chi
        # there is two return value, meldPrint() and discardOption(), 
        # while the second would not be used by client.py
        meld_chi, meld_pon = None, None
        should_chi, should_pon = False, False

        # print(tile_136)
        # print(self.player.closed_hand)
        melds_chi, melds_pon = self.get_possible_meld(tile_136, is_kamicha_discard)
        if melds_chi and meld_type & 4:
            should_chi, chi_score, tiles_chi = self.chi.should_call_chi(tile_136, melds_chi)
            # fix here: tiles_chi is now the first possible meld ---fixed! 
            # tiles_chi = melds_chi[0]
            meld_chi = Meld(meld_type="chi", tiles=tiles_chi) if meld_chi else None
        if melds_pon and meld_type & 1:
            should_pon, pon_score = self.pon.should_call_pon(tile_136, is_kamicha_discard)
            tiles_pon = melds_pon[0]
            meld_pon = Meld(meld_type="pon", tiles=tiles_pon) if meld_pon else None

        if not should_chi and not should_pon:
            return None, None

        if should_chi and should_pon:
            meld = meld_chi if chi_score > pon_score else meld_pon
        elif should_chi:
            meld = meld_chi
        else:
            meld = meld_pon

        all_tiles_copy, meld_tiles_copy = self.player.tiles[:], self.player.meld_tiles[:]
        all_tiles_copy.append(tile_136)
        meld_tiles_copy.append(meld)
        closed_hand_copy = [item for item in all_tiles_copy if item not in meld_tiles_copy]
        discard_option = self.discard.discard_tile(all_hands_136=all_tiles_copy, closed_hands_136=closed_hand_copy)

        return meld, discard_option

    def should_call_kyuushu_kyuuhai(self):
        #try kokushi strategy if with 10 types
        tiles_34 = TilesConverter.to_34_array(self.player.tiles)
        types = sum([1 for t in tiles_34 if t > 0])
        if types >= 10:
            return False
        else:
            return True


    def should_call_win(self, tile, is_tsumo, enemy_seat=None, is_chankan=False):
        # don't skip win in riichi
        if self.player.in_riichi:
            return True

        # currently we don't support win skipping for tsumo
        if is_tsumo:
            return True

        # fast path - check it first to not calculate hand cost
        cost_needed = self.placement.get_minimal_cost_needed()
        if cost_needed == 0:
            return True

        # 1 and not 0 because we call check for win this before updating remaining tiles
        is_hotei = self.player.table.count_of_remaining_tiles == 1

        hand_response = self.calculate_exact_hand_value_or_get_from_cache(
            tile,
            tiles=self.player.tiles,
            call_riichi=self.player.in_riichi,
            is_tsumo=is_tsumo,
            is_chankan=is_chankan,
            is_haitei=is_hotei,
        )
        assert hand_response is not None
        assert not hand_response.error, hand_response.error
        cost = hand_response.cost
        return self.placement.should_call_win(cost, is_tsumo, enemy_seat)

    def calculate_shanten_or_get_from_cache(self, closed_hand_34: List[int], use_chiitoitsu=True):
        """
        Sometimes we are calculating shanten for the same hand multiple times
        to save some resources let's cache previous calculations
        """
        key = build_shanten_cache_key(closed_hand_34, use_chiitoitsu)
        if key in self.hand_cache_shanten:
            return self.hand_cache_shanten[key]
        # if use_chiitoitsu and not self.player.is_open_hand:
        #     result = self.shanten_calculator.calculate_shanten_for_chiitoitsu_hand(closed_hand_34)
        # else:
        #     result = self.shanten_calculator.calculate_shanten_for_regular_hand(closed_hand_34)
        
        # fix here: a little bit strange in use_chiitoitsu
        shanten_results = []
        if use_chiitoitsu and not self.player.is_open_hand:
            shanten_results.append(self.shanten_calculator.calculate_shanten_for_chiitoitsu_hand(closed_hand_34))
        shanten_results.append(self.shanten_calculator.calculate_shanten_for_regular_hand(closed_hand_34))
        result = min(shanten_results)
        self.hand_cache_shanten[key] = result
        return result


    def calculate_exact_hand_value_or_get_from_cache(
        self,
        win_tile_136,
        tiles=None,
        call_riichi=False,
        is_tsumo=False,
        is_chankan=False,
        is_haitei=False,
        is_ippatsu=False,
    ):
        if not tiles:
            tiles = self.player.tiles[:]
        else:
            tiles = tiles[:]

        tiles += [win_tile_136]

        additional_han = 0
        if is_chankan:
            additional_han += 1
        if is_haitei:
            additional_han += 1
        if is_ippatsu:
            additional_han += 1

        config = HandConfig(
            is_riichi=call_riichi,
            player_wind=self.player.player_wind,
            round_wind=self.player.table.round_wind_tile,
            is_tsumo=is_tsumo,
            options=OptionalRules(
                has_aka_dora=self.player.table.has_aka_dora,
                has_open_tanyao=self.player.table.has_open_tanyao,
                has_double_yakuman=False,
            ),
            is_chankan=is_chankan,
            is_ippatsu=is_ippatsu,
            is_haitei=is_tsumo and is_haitei or False,
            is_houtei=(not is_tsumo) and is_haitei or False,
            tsumi_number=self.player.table.count_of_honba_sticks,
            kyoutaku_number=self.player.table.count_of_riichi_sticks,
        )

        return self._estimate_hand_value_or_get_from_cache(
            win_tile_136, tiles, call_riichi, is_tsumo, additional_han, config
        )

    def _estimate_hand_value_or_get_from_cache(
        self, win_tile_136, tiles, call_riichi, is_tsumo, additional_han, config, is_rinshan=False, is_chankan=False
    ):
        cache_key = build_estimate_hand_value_cache_key(
            tiles,
            call_riichi,
            is_tsumo,
            self.player.melds,
            self.player.table.dora_indicators,
            self.player.table.count_of_riichi_sticks,
            self.player.table.count_of_honba_sticks,
            additional_han,
            is_rinshan,
            is_chankan,
        )
        if self.hand_cache_estimation.get(cache_key):
            return self.hand_cache_estimation.get(cache_key)

        result = self.finished_hand.estimate_hand_value(
            tiles,
            win_tile_136,
            self.player.melds,
            self.player.table.dora_indicators,
            config,
            use_hand_divider_cache=True,
        )

        self.hand_cache_estimation[cache_key] = result
        return result

    @property
    def enemy_players(self):
        """
        Return list of players except our bot
        """
        return self.player.table.players[1:]

    def enemy_called_riichi(self, enemy_seat):
        """
        After enemy riichi we had to check will we fold or not
        it is affect open hand decisions
        :return:
        """
        pass

    def get_possible_meld(self, tile, is_kamicha_discard):

        closed_hand = self.player.closed_hand[:]

        # we can't open hand anymore
        if len(closed_hand) == 1:
            return None, None

        discarded_tile = tile // 4
        closed_hand_34 = TilesConverter.to_34_array(closed_hand + [tile])

        combinations = []
        first_index = 0
        second_index = 0
        if is_man(discarded_tile):
            first_index = 0
            second_index = 8
        elif is_pin(discarded_tile):
            first_index = 9
            second_index = 17
        elif is_sou(discarded_tile):
            first_index = 18
            second_index = 26

        if second_index == 0:
            # honor tiles
            if closed_hand_34[discarded_tile] == 3:
                combinations = [[[discarded_tile] * 3]]
        else:
            # to avoid not necessary calculations
            # we can check only tiles around +-2 discarded tile
            first_limit = discarded_tile - 2
            if first_limit < first_index:
                first_limit = first_index

            second_limit = discarded_tile + 2
            if second_limit > second_index:
                second_limit = second_index

            combinations = self.hand_divider.find_valid_combinations(
                closed_hand_34, first_limit, second_limit, True
            )

        if combinations:
            combinations = combinations[0]        
        # possible_melds = []
        melds_chi, melds_pon = [], []
        for best_meld_34 in combinations:
            # we can call pon from everyone
            if is_pon(best_meld_34) and discarded_tile in best_meld_34:
                if best_meld_34 not in melds_pon:
                    melds_pon.append(best_meld_34)

            # we can call chi only from left player
            if is_chi(best_meld_34) and is_kamicha_discard and discarded_tile in best_meld_34:
                if best_meld_34 not in melds_chi:
                    melds_chi.append(best_meld_34)

        return melds_chi, melds_pon          

    def enemy_called_riichi(self, enemy_seat):
        """
        After enemy riichi we had to check will we fold or not
        it is affect open hand decisions
        :return:
        """
        pass
