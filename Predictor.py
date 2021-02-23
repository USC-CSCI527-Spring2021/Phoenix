

import tensorflow as tf
import tensorflow.keras as keras

model = keras.model.load('pred_model')

class GlobalRewardPredictor:
    def __init__(self):
        self.model = keras.model.load('pred_model')

    def getModel(self):
        return self.model

    def inference(init_score, gains, dealer, repeat_dealer, riichi_bets, dan):
        '''
        input shape: (None, constants.round_num, constants.pred_dim)/(None, 15, 15)
        dan: 四位玩家的段位
        
        '''
        
        init_score = np.asarray(init_score)/1e5
        gains = np.asarray(gains)/1e5
        embed = np.concatenate((init_score, gains, dan, dealer, repeat_dealer, riichi_bets), axis=-1)
        print('input_shape', embed.shape)
        return self.model.predict(embed)

