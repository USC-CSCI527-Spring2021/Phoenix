import game.ai.models as models
import numpy as np
from utils.decisions_logger import MeldPrint
input_shape = (None, 34, 1)   ######### fix input_shape afterwards

    

def getGeneralFeature(player):
    def _indicator2dora(dora_indicator):
        dora = dora_indicator//4
        if dora < 27: #EAST
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
        a = score//(50000.0/33)
        alpha = a + 1 - (score*33.0/50000)
        alpha = round(alpha, 2)
        feature[int(a)] = alpha
        if a<33:
            feature[int(a)] = 1-alpha
        return feature

    def _getPlayerTiles(player):
        closed_hand_feature = np.zeros((4, 34))
        open_hand_feature = np.zeros((4, 34))
        discarded_tiles_feature = np.zeros((4, 34))
        
        open_hand=[]
        for meld in player.melds:
            for tile in meld.tiles:
                open_hand.append(tile)
        closed_hand=[]
        for tile in player.tiles:
            if tile not in open_hand:
                closed_hand.append(tile)
        discarded_tiles=player.discards
        
        for val in closed_hand:
            idx=0
            while(closed_hand_feature[idx][val//4] == 1):
                idx+=1
            closed_hand_feature[idx][val//4] = 1
        for val in open_hand:
            idx=0
            while(open_hand_feature[idx][val//4] == 1):
                idx+=1
            open_hand_feature[idx][val//4] = 1
        for val in discarded_tiles:
            idx=0
            while(discarded_tiles_feature[idx][val//4] == 1):
                idx+=1
            discarded_tiles_feature[idx][val//4] = 1
        
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
        for idx, dora_indicator in enumerate(dora_list):
            dora_feature[idx][_indicator2dora(dora_indicator)] = 1
        return dora_feature

    def _getScoreList1(player): 
        scores_feature = np.zeros((4, 34))
        for i in range(4):
            scores_feature[i] = _transscore(player.table.get_player(i).scores)
        return scores_feature

    def _getBoard1(player): 
        dealer_feature = np.zeros((4, 34))
        dealer_feature[player.table.dealer_seat] = 1

        repeat_dealer_feature = np.zeros((8, 34))
        repeat_dealer = player.table.count_of_honba_sticks
        if repeat_dealer > 7 :
            repeat_dealer = 7
        repeat_dealer_feature[repeat_dealer] = 1

        riichi_bets_feature = np.zeros((8, 34))
        riichi_bets = player.table.count_of_riichi_sticks
        if riichi_bets > 7 :
            repeat_dealer = 7
        riichi_bets_feature[riichi_bets] = 1

        player_wind_feature = np.zeros((4, 34))
        player_wind_feature[(4-player.table.dealer_seat)%4] = 1

        prevailing_wind_feature = np.zeros((4, 34))
        prevailing_wind_feature[round_wind_number//4] = 1

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
        self.model = models.make_or_restore_model(input_shape, "chi")
    
    def should_call_chi(self, tile_136, is_kamicha_discard):
        features = self.getFeature(tile_136)
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self, tile_136):
        tile_136_feature = np.zeros((1, 34))
        tile_136_feature[0][tile_136//4] = 1
        x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))


class Pon:
    def __init__(self, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "pon")
    def should_call_pon(self, tile_136, is_kamicha_discard):
        features = self.getFeature(tile_136)
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self, tile_136):
        tile_136_feature = np.zeros((1, 34))
        tile_136_feature[0][tile_136 // 4] = 1
        x = np.concatenate((tile_136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))



class Kan:
    def __init__(self, input_shape, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "kan")

    def should_call_kan(self, tile, open_kan, from_riichi=False):
        features = self.getFeature(tile, open_kan)
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self, tile136, open_kan):
        def _is_kakan(tile, melds):
            for meld in melds:
                if (meld.type == MeldPrint.PON) and (tile136//4 == meld.tiles[0]//4):
                    return True
            return False

        tile136_feature = np.zeros((1, 34))
        tile136_feature[0][tile136 // 4] = 1
        kan_type_feature = np.zeros((3, 34))
        if not open_kan:
            kan_type_feature[1] = 1
        else:
            if _is_kakan(tile136, self.player.melds):
                kan_type_feature[2] = 1
            else:
                kan_type_feature[0] = 1
        x = np.concatenate((kan_type_feature, tile136_feature, getGeneralFeature(self.player)))
        return x.reshape((x.shape[0], x.shape[1], 1))


class Riichi:
    def __init__(self, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "riichi")

    def should_call_riichi(self):
        features = self.getFeature()
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self):
        x = getGeneralFeature(self.player)
        return x.reshape((x.shape[0], x.shape[1], 1))

class Discard:
    def __init__(self, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "discard")

    def discard_tile(self, all_hands_136=None, closed_hands_136=None, with_riichi=False):
        '''
        The reason why these two should be input:
        if the "discard" action is from "discard after meld", since we have decided to meld, what
        to do next is to decide what to discard, but the meld has not happened (and could also be 
        interrupted), so we could only use discard model based on supposed hands after melding

        '''
        features = self.getFeature(all_hands_136, closed_hands_136, with_riichi)
        tile_to_discard = np.argmax(self.model.predict(np.expand_dims(features, axis=0))[0])
        tile_to_discard_136 = [h for h in closed_hands_136 if h // 4 == tile_to_discard][-1] 
        #if multiple tiles exists, return the one which is not red dora
        return tile_to_discard_136
                



    def getFeature(self, open_hands_136, closed_hands_136, with_riichi):
        #if hands are none, get hands from self.player
        return np.random.randn(62, 34, 1)       #change this according to actual shape