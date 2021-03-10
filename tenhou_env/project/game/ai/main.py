from typing import List

import utils.decisions_constants as log
from game.ai.defence.main import TileDangerHandler
from game.ai.hand_builder import HandBuilder
from game.ai.helpers.kabe import Kabe
from game.ai.helpers.suji import Suji
from game.ai.kan import Kan
from game.ai.riichi import Riichi
from game.ai.strategies.chinitsu import ChinitsuStrategy
from game.ai.strategies.common_open_tempai import CommonOpenTempaiStrategy
from game.ai.strategies.formal_tempai import FormalTempaiStrategy
from game.ai.strategies.honitsu import HonitsuStrategy
from game.ai.strategies.main import BaseStrategy
from game.ai.strategies.tanyao import TanyaoStrategy
from game.ai.strategies.yakuhai import YakuhaiStrategy
from mahjong.agari import Agari
from mahjong.constants import AKA_DORA_LIST, DISPLAY_WINDS
from mahjong.hand_calculating.divider import HandDivider
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from utils.cache import build_estimate_hand_value_cache_key, build_shanten_cache_key


class MahjongAI:
    version = "0.5.1"

    agari = None
    shanten_calculator = None
    defence = None
    riichi = None
    hand_divider = None
    finished_hand = None

    shanten = 7
    ukeire = 0
    ukeire_second = 0
    waiting = None

    current_strategy = None
    last_discard_option = None

    hand_cache_shanten = {}
    hand_cache_estimation = {}

    def __init__(self, player):
        self.player = player
        self.table = player.table

        self.kan = Kan(player)
        self.agari = Agari()
        self.shanten_calculator = Shanten()
        self.defence = TileDangerHandler(player)
        self.riichi = Riichi(player)
        self.hand_divider = HandDivider()
        self.finished_hand = HandCalculator()
        self.hand_builder = HandBuilder(player, self)
        self.placement = player.config.PLACEMENT_HANDLER_CLASS(player)

        self.suji = Suji(player)
        self.kabe = Kabe(player)

        self.erase_state()

    def erase_state(self):
        self.shanten = 7
        self.ukeire = 0
        self.ukeire_second = 0
        self.waiting = None

        self.current_strategy = None
        self.last_discard_option = None

        self.hand_cache_shanten = {}
        self.hand_cache_estimation = {}

        # to erase hand cache
        self.finished_hand = HandCalculator()

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

    def draw_tile(self, tile_136):
        if not self.player.in_riichi:
            self.determine_strategy(self.player.tiles)

    def discard_tile(self, discard_tile):
        # we called meld and we had discard tile that we wanted to discard
        if discard_tile is not None:
            if not self.last_discard_option:
                return discard_tile, False

            return self.hand_builder.process_discard_option(self.last_discard_option)

        return self.hand_builder.discard_tile()

    def try_to_call_meld(self, tile_136, is_kamicha_discard, meld_type=None):
        tiles_136_previous = self.player.tiles[:]
        closed_hand_136_previous = self.player.closed_hand[:]
        tiles_136 = tiles_136_previous + [tile_136]
        self.determine_strategy(tiles_136, meld_tile=tile_136)

        if not self.current_strategy:
            self.player.logger.debug(log.MELD_DEBUG, "We don't have active strategy. Abort melding.")
            return None, None

        closed_hand_34_previous = TilesConverter.to_34_array(closed_hand_136_previous)
        previous_shanten, _ = self.hand_builder.calculate_shanten_and_decide_hand_structure(closed_hand_34_previous)

        if previous_shanten == Shanten.AGARI_STATE and not self.current_strategy.can_meld_into_agari():
            return None, None

        meld, discard_option = self.current_strategy.try_to_call_meld(tile_136, is_kamicha_discard, tiles_136)
        if discard_option:
            self.last_discard_option = discard_option

            self.player.logger.debug(
                log.MELD_CALL,
                "We decided to open hand",
                context=[
                    f"Hand: {self.player.format_hand_for_print(tile_136)}",
                    f"Meld: {meld.serialize()}",
                    f"Discard after meld: {discard_option.serialize()}",
                ],
            )

        return meld, discard_option

    def determine_strategy(self, tiles_136, meld_tile=None):
        # for already opened hand we don't need to give up on selected strategy
        if self.player.is_open_hand and self.current_strategy:
            return False

        old_strategy = self.current_strategy
        self.current_strategy = None

        # order is important, we add strategies with the highest priority first
        strategies = []

        if self.player.table.has_open_tanyao:
            strategies.append(TanyaoStrategy(BaseStrategy.TANYAO, self.player))

        strategies.append(YakuhaiStrategy(BaseStrategy.YAKUHAI, self.player))
        strategies.append(HonitsuStrategy(BaseStrategy.HONITSU, self.player))
        strategies.append(ChinitsuStrategy(BaseStrategy.CHINITSU, self.player))

        strategies.append(FormalTempaiStrategy(BaseStrategy.FORMAL_TEMPAI, self.player))
        strategies.append(CommonOpenTempaiStrategy(BaseStrategy.COMMON_OPEN_TEMPAI, self.player))

        for strategy in strategies:
            if strategy.should_activate_strategy(tiles_136, meld_tile=meld_tile):
                self.current_strategy = strategy
                break

        if self.current_strategy and (not old_strategy or self.current_strategy.type != old_strategy.type):
            self.player.logger.debug(
                log.STRATEGY_ACTIVATE,
                context=self.current_strategy,
            )

        if not self.current_strategy and old_strategy:
            self.player.logger.debug(log.STRATEGY_DROP, context=old_strategy)

        return self.current_strategy and True or False

    def estimate_hand_value_or_get_from_cache(
        self, win_tile_34, tiles=None, call_riichi=False, is_tsumo=False, is_rinshan=False, is_chankan=False
    ):
        win_tile_136 = win_tile_34 * 4

        # we don't need to think, that our waiting is aka dora
        if win_tile_136 in AKA_DORA_LIST:
            win_tile_136 += 1

        if not tiles:
            tiles = self.player.tiles[:]
        else:
            tiles = tiles[:]

        tiles += [win_tile_136]

        config = HandConfig(
            is_riichi=call_riichi,
            player_wind=self.player.player_wind,
            round_wind=self.player.table.round_wind_tile,
            is_tsumo=is_tsumo,
            is_rinshan=is_rinshan,
            is_chankan=is_chankan,
            options=OptionalRules(
                has_aka_dora=self.player.table.has_aka_dora,
                has_open_tanyao=self.player.table.has_open_tanyao,
                has_double_yakuman=False,
            ),
            tsumi_number=self.player.table.count_of_honba_sticks,
            kyoutaku_number=self.player.table.count_of_riichi_sticks,
        )

        return self._estimate_hand_value_or_get_from_cache(
            win_tile_136, tiles, call_riichi, is_tsumo, 0, config, is_rinshan, is_chankan
        )

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

    def estimate_weighted_mean_hand_value(self, discard_option):
        weighted_hand_cost = 0
        number_of_tiles = 0
        for waiting in discard_option.waiting:
            tiles = self.player.tiles[:]
            tiles.remove(discard_option.tile_to_discard_136)

            hand_cost = self.estimate_hand_value_or_get_from_cache(
                waiting, tiles=tiles, call_riichi=discard_option.with_riichi, is_tsumo=True
            )

            if not hand_cost.cost:
                continue

            weighted_hand_cost += (
                                      hand_cost.cost["main"] + 2 * hand_cost.cost["additional"]
                                  ) * discard_option.wait_to_ukeire[waiting]
            number_of_tiles += discard_option.wait_to_ukeire[waiting]

        cost = number_of_tiles and int(weighted_hand_cost / number_of_tiles) or 0

        # we are karaten, or we don't have yaku
        # in that case let's add possible tempai cost
        if cost == 0 and self.player.round_step > 12:
            cost = 1000

        if self.player.round_step > 15 and cost < 2500:
            cost = 2500

        return cost

    def should_call_kyuushu_kyuuhai(self) -> bool:
        """
        Kyuushu kyuuhai 「九種九牌」
        (9 kinds of honor or terminal tiles)
        """
        # TODO aim for kokushi
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

    def enemy_called_riichi(self, enemy_seat):
        """
        After enemy riichi we had to check will we fold or not
        it is affect open hand decisions
        :return:
        """
        pass

    def calculate_shanten_or_get_from_cache(self, closed_hand_34: List[int], use_chiitoitsu: bool):
        """
        Sometimes we are calculating shanten for the same hand multiple times
        to save some resources let's cache previous calculations
        """
        key = build_shanten_cache_key(closed_hand_34, use_chiitoitsu)
        if key in self.hand_cache_shanten:
            return self.hand_cache_shanten[key]
        if use_chiitoitsu and not self.player.is_open_hand:
            result = self.shanten_calculator.calculate_shanten_for_chiitoitsu_hand(closed_hand_34)
        else:
            result = self.shanten_calculator.calculate_shanten_for_regular_hand(closed_hand_34)
        self.hand_cache_shanten[key] = result
        return result

    @property
    def enemy_players(self):
        """
        Return list of players except our bot
        """
        return self.player.table.players[1:]

    
    
