
from keras import backend as K


ENTROPY_LOSS = 5e-3
LOSS_CLIPPING = 0.2

def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1-LOSS_CLIPPING, max_value=1+LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

input_shape_dict = {'chi': (63,34,1), 'pon':(63,34,1), 'kan':(66,34,1), 'riichi':(62,34,1), 'discard': (16,34,1)}
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 1

RANKS = [
    "新人",
    "9級",
    "8級",
    "7級",
    "6級",
    "5級",
    "4級",
    "3級",
    "2級",
    "1級",
    "初段",
    "二段",
    "三段",
    "四段",
    "五段",
    "六段",
    "七段",
    "八段",
    "九段",
    "十段",
    "天鳳位",
]

pred_emb_dim = 15
round_num = 15
model_types = ['chi', 'pon', 'kan', 'riichi', 'discard']