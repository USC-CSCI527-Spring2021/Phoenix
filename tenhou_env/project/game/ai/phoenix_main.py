from game.ai.nn import Chi, Pon, Kan, Riichi, Discard
from game.ai.hand_builder import HandBuilder
from mahjong.meld import Meld
from mahjong.utils import is_honor, is_terminal
from mahjong.tile import TilesConverter

class Phoenix:
    def __init__(self, player):
        self.player = player
        self.table = player.table

        self.chi = Chi(player)
        self.pon = Pon(player)
        self.kan = Kan(player)
        self.riichi = Riichi(player)
        self.discard = Discard(player)
        self.hand_builder = HandBuilder(player, self)
        
        self.erase_state()

    def erase_state(self):
        pass

    def init_hand(self):
        pass

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
        
        waiting, shanten = self.hand_builder.calculate_waits(self.player.closed_hand, self.player.tiles)
        if shanten != 0:                #can not riichi
            return self.discard.discard_tile(), False
        with_riichi, p = self.riichi.should_call_riichi()
        tile_to_discard = self.discard.discard_tile(with_riichi=with_riichi)
        return tile_to_discard
            
    def try_to_call_meld(self, tile_136, is_kamicha_discard, meld_type):
        # 1 pon
        # 2 kan (it is a closed kan and can be send only to the self draw)
        # 4 chi
        # there is two return value, meldPrint() and discardOption(), 
        # while the second would not be used by client.py
        if meld_type & 4:
            meld_chi, chi_score = self.chi.should_call_chi(tile_136, is_kamicha_discard)
            meld_chi = Meld(meld_type="chi", tiles=meld_chi) if meld_chi else None
        if meld_type & 1:
            meld_pon, pon_score = self.pon.should_call_pon(tile_136, is_kamicha_discard)
            meld_pon = Meld(meld_type="pon", tiles=meld_pon) if meld_pon else None


        if meld_chi == None and meld_pon == None:
            return None, None

        if meld_chi and meld_pon:
            meld = meld_chi if chi_score > pon_score else meld_pon
        elif meld_chi:
            meld = meld_chi
        else:
            meld = meld_pon

        all_tiles_copy, meld_tiles_copy = self.player.tiles, self.player.meld_tiles
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


    @property
    def enemy_players(self):
        """
        Return list of players except our bot
        """
        return self.player.table.players[1:]
