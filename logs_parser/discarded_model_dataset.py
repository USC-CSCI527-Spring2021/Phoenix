import sys

from parser import parse_mjlog
from viewer import print_node
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import copy
from pandas.core.frame import DataFrame


def get_round_info(dataset, 
                   draw_tile_list, 
                   hands_list, 
                   discarded_tiles_pool_list, 
                   four_players_open_hands_list,
                   discarded_tile
                  ):
       
    discarded_tiles_pool = []

    # acquire init scores and states
    init_data = dataset[0]["data"]
    four_player_hands = init_data["hands"]

    dealer = init_data["oya"]

    four_players_open_hands =  [[], [], [], []]
    
    last_act = 'init_act'
    
    # acquire actions and states after each action
    for k in range(len(dataset)):
        action = dataset[k]
        act = action["tag"]
        
        if act == 'REACH' or act == 'DORA' :
            continue       
            
        elif act == "CALL":
            caller = action["data"]["caller"]
            callee = action["data"]["callee"]
            tiles = action["data"]["mentsu"]
                  
            if (action["data"]["call_type"] == 'Pon' or action["data"]["call_type"] == 'Chi'):  


                for tile in tiles:   
                    if tile not in four_player_hands[caller]:
                        four_player_hands[caller].append(tile)
                        draw_tile_list.append(tile)
                        discarded_tiles_pool.remove(tile)
                    if tile not in four_players_open_hands[caller]:
                        four_players_open_hands[caller].append(tile)
                        
                four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])
    
                hands_list.append(copy.copy(four_player_hands[caller]))
                four_players_open_hands_list.append(copy.copy(four_players_open_hands))

            elif action["data"]["call_type"] == 'MinKan':

                for tile in tiles:   
                    if tile not in four_player_hands[caller]:
                        four_player_hands[caller].append(tile)
                        discarded_tiles_pool.remove(tile)
                    if tile not in four_players_open_hands[caller]:
                        four_players_open_hands[caller].append(tile)
                        
                             
            elif (action["data"]["call_type"] == 'AnKan' or action["data"]["call_type"] == 'KaKan'):
                
                # if ankan or kakan, then the last act must be draw. pop the last four_players_open_hands
                # from list to avoid duplicate
                hands_list.pop()
                draw_tile_list.pop()
                four_players_open_hands_list.pop()

                player_open_tile = four_players_open_hands[caller]
                
                for tile in tiles:           
                    if tile not in player_open_tile:
                        player_open_tile.append(tile)
        
        # deal with draw and discard operations  
        elif act in ["DRAW", "DISCARD"]:
            pl_id = action["data"]["player"]
            tile = action["data"]["tile"]
            
            if act == "DRAW":   
                
                four_player_hands[pl_id].append(tile)
                hands_list.append(copy.copy(four_player_hands[pl_id]))
                draw_tile_list.append(tile)
                four_players_open_hands_list.append(copy.copy(four_players_open_hands))
                
  
            elif act == "DISCARD":
                        
                discarded_tiles_pool_list.append(copy.copy(discarded_tiles_pool))

                discarded_tile.append(tile)
                discarded_tiles_pool.append(tile)
                four_player_hands[pl_id].remove(tile)
        
        
        elif act == 'AGARI' or act == 'RYUUKYOKU':     
            # if the tag is AGARI, then the last draw is meaningless for the tile_discarded_model
            if last_act == 'DRAW':
                hands_list.pop()
                draw_tile_list.pop()
                four_players_open_hands_list.pop()

                    
        last_act = act
                
                
    
# In[3]:


def encoded_tiles(matrix, tiles):
    for tile in tiles:
        col = tile // 4
        row = tile % 4
        matrix[row][col] = 1
    return matrix 

# encoded tiles from 136 format to 34*4 format
def transform_136_to_34m4(result):

    for i in range(len(result[1])):
        tiles = result[1][i]
        
        result[1][i] = encoded_tiles(np.zeros((4, 34)), tiles)
        
        four_players_tiles = result[3][i]
        encoded_matrix = np.zeros((4, 4, 34))         
        for j in range(4):
            encoded_matrix[j] = encoded_tiles(np.zeros((4, 34)), four_players_tiles[j])
        
        result[3][i] = encoded_matrix


if __name__ == "__main__":
    
    # path for input csv file
    input_csv_path = sys.argv[1]
    # path for output npy file
    output_npy_path = sys.argv[2]

    # read data from csv file
    df = pd.read_csv(input_csv_path)
    
    # features
    draw_tile_list = []
    hands_list = []
    discarded_tiles_pool_list = []
    four_players_open_hands_list = []
    
    # label
    discarded_tile = [] 

    for i in range(len(df["log_content"])):
        xml_str = df["log_content"][i]    
        # transform data from xml format to dict
        node = ET.fromstring(xml_str)
        data = parse_mjlog(node)

        # remove three-players games 
        if len(data["meta"]['UN'][3]["name"]) == 0:
            continue

        for j in range(len(data["rounds"])):  
            round_data = data["rounds"][j] 
            
            get_round_info(round_data, draw_tile_list, 
                           hands_list, 
                           discarded_tiles_pool_list, 
                           four_players_open_hands_list,
                           discarded_tile)
            
        
    result = [draw_tile_list,
              hands_list,
              discarded_tiles_pool_list, 
              four_players_open_hands_list, 
              discarded_tile]
    
    transform_136_to_34m4(result)
     
    encoded_np_result = np.array(result)
    
    np.save(output_npy_path, encoded_np_result)
       


