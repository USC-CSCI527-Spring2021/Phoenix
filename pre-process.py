#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:





# In[2]:


import sys
# from io import load_mjlog
from parser import parse_mjlog
from viewer import print_node
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


# In[3]:


def set_tile_as_one_in_matrix(matrix, tile):
    col = tile // 4
    row = tile % 4
    matrix[row][col] = 1
    return matrix
    
def set_tile_as_zero_in_matrix(matrix, tile):
    col = tile // 4
    row = tile % 4
    matrix[row][col] = 0
    return matrix

def add_tiles(matrix, tiles):
    for tile in tiles:
        col = tile // 4
        row = tile % 4
        matrix[row][col] = 1
    return matrix 

def update_open_private_hands_after_call(encoded_open_hands, encoded_private_hands, tiles):
    for tile in tiles:
        encoded_open_hands = set_tile_as_one_in_matrix(encoded_open_hands, tile)
        encoded_private_hands = set_tile_as_zero_in_matrix(encoded_private_hands, tile)
    return encoded_open_hands, encoded_private_hands

def updated_calls(tiles, private_hands, open_hands, cur_discarded_tiles, caller, callee, encoded_hands, players_hands_after_draw_call, encoded_open_hands, encoded_private_hands):
    
    encoded_hands[caller] = add_tiles(encoded_hands[caller], tiles)
    players_hands_after_draw_call[caller].append(encoded_hands[caller])

    cur_discarded_tiles[callee].pop()

    updated_matrixs = update_open_private_hands_after_call(encoded_open_hands[caller], encoded_private_hands[caller], tiles)      
    open_hands[caller].append(updated_matrixs[0])       
    private_hands[caller].append(updated_matrixs[1])  


# In[4]:


def get_round_info(round_data):
    players_hands_after_draw_call = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    discarded_tiles = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    cur_discarded_tiles = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    private_hands = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    open_hands = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    
    gains = []
    init_socres = []
    dora_indicators = []
    final_results = {
        'AGARI' : [],
        'RYUUKYOKU' : []
    }

    # acquire init scores and states
    init_data = round_data[0]["data"]
    init_hands = init_data["hands"]
    dora_indicators.append(init_data["dora"])
    dealer = init_data["oya"]
    round_num = init_data["round"]    
    

    encoded_hands = np.zeros((4, 4, 34))

    encoded_private_hands = np.zeros((4, 4, 34))
    encoded_open_hands = np.zeros((4, 4, 34))

    # encoded init_hands into 4 *34 array
    for player_id in range(len(init_hands)):
        for index, tile in enumerate(init_hands[player_id]):
            encoded_hands[player_id][tile % 4][tile // 4] = 1
            encoded_private_hands[player_id][tile % 4][tile // 4] = 1

        players_hands_after_draw_call[player_id].append(encoded_hands[player_id])  
        private_hands[player_id].append(encoded_private_hands[player_id])  
        open_hands[player_id].append(encoded_open_hands[player_id])  

    # acquire actions and states after draw
    for action in round_data[1:]:
        pl_id = -1
        caller = -1
        act = action["tag"]
        
        if act == 'REACH':
            continue       
        elif act == 'AGARI':
            gains = action["data"]["gains"] 
            init_socres = action["data"]["scores"]
            final_results['AGARI'] = action["data"]
            
        # RYUUKYOKU, add the final gains to result 
        elif act == 'RYUUKYOKU':
            gains = action["data"]["gains"]  
            init_socres = action["data"]["scores"] 
            final_results['RYUUKYOKU'] = action["data"]
            
            
        elif act == 'DORA':
            dora_indicators.append(action["data"]["hai"])
            
        elif act == "CALL":
            caller = action["data"]["caller"]
            callee = action["data"]["callee"]
            tiles = action["data"]["mentsu"]

            if (action["data"]["call_type"] == 'Pon' or action["data"]["call_type"] == 'MinKan' or action["data"]["call_type"] == 'Chi'):   
                updated_calls(tiles, private_hands, open_hands, cur_discarded_tiles, caller, callee, encoded_hands, players_hands_after_draw_call, encoded_open_hands, encoded_private_hands)
                
            elif (action["data"]["call_type"] == 'AnKan'):
                # set two ankan tiles as open, the other two as private 
                an_kan_tiles = action["data"]["mentsu"]  
                updated_matrixs = update_open_private_hands_after_call(encoded_open_hands[caller], encoded_private_hands[caller], an_kan_tiles) 
                open_hands[caller].append(updated_matrixs[0])       
                private_hands[caller].append(updated_matrixs[1])  
                
            elif (action["data"]["call_type"] == 'KaKan'):
                ka_kan_tiles = action["data"]["mentsu"]          
                updated_matrixs = update_open_private_hands_after_call(encoded_open_hands[caller], encoded_private_hands[caller], ka_kan_tiles) 
                open_hands[caller].append(updated_matrixs[0])       
                private_hands[caller].append(updated_matrixs[1]) 

            else:     
                print(init_data)
                print(action)

        # deal with draw and discard operations  
        elif act in ["DRAW", "DISCARD"]:
            pl_id = action["data"]["player"]
            tile = action["data"]["tile"]

            col = tile // 4
            row = tile % 4

            if act == "DRAW":
                encoded_hands[pl_id][row][col] = 1
                players_hands_after_draw_call[pl_id].append(encoded_hands[pl_id])

                open_hands[pl_id].append(encoded_open_hands[pl_id])       
                private_hands[pl_id].append(encoded_private_hands[pl_id])  

            elif act == "DISCARD":
                encoded_hands[pl_id][row][col] = 0
                cur_discarded_tiles[pl_id].append(tile)
                discarded_tiles[pl_id].append(cur_discarded_tiles[pl_id])

                encoded_private_hands[pl_id] = set_tile_as_zero_in_matrix(encoded_private_hands[pl_id], tile)
                private_hands[pl_id].append(encoded_private_hands[pl_id])  
                open_hands[pl_id].append(encoded_open_hands[pl_id])  
        else :
            print(action)
                 
    
    return {
        "dealer" : dealer,
        "round_num" : round_num, 
        "init_socres" : init_socres,
        "players_hands_after_draw_call": players_hands_after_draw_call,
        "private_hands" : private_hands,
        "open_hands" : open_hands,
        "discarded_tiles" :  discarded_tiles,
        "gains" : gains,
        "dora_indicators" : dora_indicators,
        "final_results" : final_results
    }


# In[ ]:





# In[ ]:





# In[6]:


# note:
# 一个log对应一场game,一场game里可能有多轮rounds
# 我们是把一个round里的信息，放到一个dict里; 

# 然后同game里的rounds，被统一添加到res这个dict里
# 例：
# res {
#     0: round0_info
#     1: round1_info
# }

# 然后把每场game对应的res，都存到results这个list里

# 每round包含的信息有
#     {
#             "dealer" : dealer,  庄家
#             "round_num" : round_num,  在这场game里，是哪个round，0代表东一局， 1 代表东二局， 以此类推
#             "init_socres" : init_socres, 这场round初始分数
#             "players_hands_after_draw_call": players_hands_after_draw_call, 每次draw或者call之后，玩家手里的牌
#             "private_hands" : private_hands, 每次draw或者call之后，玩家手里的暗置牌
#             "open_hands" : open_hands, 每次draw或者call之后，玩家手里的明置牌
#             "discarded_tiles" :  discarded_tiles, 每次discard之后，该玩家牌河里打出的牌（如果被chi\pon\gan，被call的那个牌，会从牌河里移除）
#             "gains" : gains, 这场round结束后，四位玩家赢得的分数
#             "dora_indicators" : dora_indicators, 宝牌指示牌
#             "final_results" : final_results 该round的结果，可能是有人赢，可能是流局 对应的信息也都存在这里
#     }

#     上面那些信息的格式：
#     players_hands_after_draw_call = {
#         0: [],
#         1: [],
#         2: [],
#         3: []
#     }

#     private_hands = {
#         0: [],
#         1: [],
#         2: [],
#         3: []
#     }

#     open_hands = {
#         0: [],
#         1: [],
#         2: [],
#         3: []
#     }

#     discarded_tiles = {
#         0: [],
#         1: [],
#         2: [],
#         3: []
#     }

#     'final_results':
#         {'AGARI': 
#              {'winner': 0, 
#               'hand': [15, 19, 23, 40, 44, 48, 57, 59, 89, 95, 96], 
#               'machi': [40], 'dora': [78], 
#               'ura_dora': [], 
#               'yaku': [(8, 1)], 
#               'yakuman': [], 
#               'ten': {'fu': 30, 'point': 1000, 'limit': 0}, 
#               'ba': {'combo': 0, 'reach': 0}, 
#               'scores': [22500, 32500, 20000, 25000], 
#               'gains': [1000, -1000, 0, 0], 
#               'loser': 1, 
#               'result': {'scores': [23500, 31500, 20000, 25000], 
#                          'uma': [-16.0, 41.0, -30.0, 5.0]}
#              }, 
#          'RYUUKYOKU': []
#         }
    
        
if __name__ == "__main__":
    
    # read data from csv file
    df = pd.read_csv("2021.csv")
    results = []

    # get the frist log as example
    # one log represents one competition
    for i in range(len(df["log_content"])):

        xml_str = df["log_content"][i]    
        # transform data from xml format to dict
        node = ET.fromstring(xml_str)
        data = parse_mjlog(node)

        # remove three-players games 
        if len(data["meta"]['UN'][3]["name"]) == 0:
            continue

        res = {}
        for i in range(len(data["rounds"])):

            round_data = data["rounds"][i] 
            round_info = get_round_info(round_data)
            res[i] = round_info

        results.append(res)
    print(results[0])


# In[ ]:





# In[ ]:





# In[ ]:




