#!/usr/bin/env python
# coding: utf-8

# In[1]:


# script for json file


# In[2]:


import sys

from parser import parse_mjlog
from viewer import print_node
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import copy
from pandas.core.frame import DataFrame
import h5py

import json


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
                        four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])
                        four_players_open_hands[caller].append(tile)
                        
    
                hands_list.append(copy.copy(four_player_hands[caller]))
                four_players_open_hands_list.append(copy.copy(four_players_open_hands))

            elif action["data"]["call_type"] == 'MinKan':

                for tile in tiles:   
                    if tile not in four_player_hands[caller]:
                        four_player_hands[caller].append(tile)

                        discarded_tiles_pool.remove(tile)
                        
                    if tile not in four_players_open_hands[caller]:
                        four_players_open_hands[caller].append(tile)

                four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])
                        
                             
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

                four_players_open_hands[caller] = copy.copy(four_players_open_hands[caller])
        
        # deal with draw and discard operations  
        elif act in ["DRAW", "DISCARD"]:
            pl_id = action["data"]["player"]
            tile = action["data"]["tile"]
            
            if act == "DRAW":   
                
                four_player_hands[pl_id].append(tile)
                hands_list.append(copy.copy(four_player_hands[pl_id]))
                draw_tile_list.append(tile)
                four_players_open_hands = copy.copy(four_players_open_hands)
                four_players_open_hands_list.append(four_players_open_hands)
                
  
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





# In[ ]:





# In[3]:



if __name__ == "__main__":
    
    # path for dir
    dir_path = '../../dataset/'
    
    # path for output hdf5 file 
    output_json_path = './discarded_model_summary.json'
    
    file = open(output_json_path,'a+',encoding='utf-8')
 

    for year in range(2009, 2022):      
        if year == 2020: 
            continue

        print("processing " + str(year))
        input_csv_path = dir_path + str(year) + '.csv'
        # read data from csv file
        df = pd.read_csv(input_csv_path)

        for i in range(len(df["log_content"])):
            xml_str = df["log_content"][i]    

            if type(xml_str) != str:
                continue
            else:
                # transform data from xml format to dict
                node = ET.fromstring(xml_str)
                data = parse_mjlog(node)

                # remove three-players games 
                if len(data["meta"]['UN'][3]["name"]) == 0:
                    continue
                else:        

                    for j in range(len(data["rounds"])):  
                        
                        # features
                        draw_tile_list = []
                        hands_list = []
                        discarded_tiles_pool_list = []
                        four_players_open_hands_list = []

                        # label
                        discarded_tile = []  

                        round_data = data["rounds"][j] 

                        get_round_info(round_data, 
                                       draw_tile_list, 
                                       hands_list, 
                                       discarded_tiles_pool_list, 
                                       four_players_open_hands_list,
                                       discarded_tile)
                        
                        for k in range(len(draw_tile_list)):
                            res = {
                                'draw_tile': draw_tile_list[k],
                                'hands': hands_list[k],
                                'discarded_tiles_pool' : discarded_tiles_pool_list[k],
                                'four_players_open_hands': four_players_open_hands_list[k],
                                'discarded_tile': discarded_tile[k]                     
                            }
                            
                            res_str = json.dumps(res)
                            file.write(res_str+'\n')
                            
                        
            
        
        
        
        


# In[ ]:




