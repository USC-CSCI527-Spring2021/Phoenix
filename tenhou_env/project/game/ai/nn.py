import game.ai.models as models
import numpy as np
input_shape = (None, 34, 1)   ######### fix input_shape afterwards

class Chi:
    def __init__(self, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "chi")
    
    def should_call_chi(self, tile_136, is_kamicha_discard):
        features = self.getFeature()
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self):
        return np.random.randn(62, 34, 1)       #change this according to actual shape


class Pon:
    def __init__(self, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "pon")
    def should_call_pon(self, tile_136, is_kamicha_discard):
        features = self.getFeature()
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self):
        return np.random.randn(62, 34, 1)       #change this according to actual shape



class Kan:
    def __init__(self, input_shape, player):
        self.player = player
        self.model = models.make_or_restore_model(input_shape, "kan")

    def should_call_kan(self, tile, open_kan, from_riichi=False):
        features = self.getFeature()
        p_donot, p_do = self.model.predict(np.expand_dims(features, axis=0))[0]
        if p_do > p_donot:
            return True, p_do
        else:
            return False, p_donot

    def getFeature(self):
        return np.random.randn(62, 34, 1)       #change this according to actual shape


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
        return np.random.randn(62, 34, 1)       #change this according to actual shape

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