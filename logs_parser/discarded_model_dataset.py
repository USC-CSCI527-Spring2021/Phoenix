import copy
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd

from logs_parser.parser import parse_mjlog


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
                

def encoded_tiles(matrix, tiles):
    for tile in tiles:
        col = tile // 4
        row = tile % 4
        matrix[row][col] = 1
    return matrix 

# encoded tiles from 136 format to 34*4 format
def transform_136_to_34m4(hands_list, four_players_open_hands, discarded_tiles_pool_list):

    for i in range(len(hands_list)):
        tiles = hands_list[i]
        
        hands_list[i] = encoded_tiles(np.zeros((4, 34)), tiles)
        discarded_tiles_pool_list[i] =  encoded_tiles(np.zeros((4, 34)), discarded_tiles_pool_list[i])
        
        four_players_tiles = four_players_open_hands[i]
        encoded_matrix = np.zeros((4, 4, 34))         
        for j in range(4):
            encoded_matrix[j] = encoded_tiles(np.zeros((4, 34)), four_players_tiles[j])
        
        four_players_open_hands[i] = encoded_matrix



if __name__ == "__main__":
    
    # path for dir
    dir_path = '../data/'

    # path for output hdf5 file 
    output_hdf5_path = './discarded_model_dataset_sum_2021.hdf5'


    # create a hdf5 file    
    # "h5py_example.hdf5"
    f = h5py.File(output_hdf5_path, "a")
    is_init = True
    
    # for year in range(2015, 2022):
    for year in range(2012, 2013):

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
                    # features
                    draw_tile_list = []
                    hands_list = []
                    discarded_tiles_pool_list = []
                    four_players_open_hands_list = []
                    
                    # label
                    discarded_tile = []     

                    for j in range(len(data["rounds"])):  

                        round_data = data["rounds"][j] 

                        get_round_info(round_data, 
                                       draw_tile_list, 
                                       hands_list, 
                                       discarded_tiles_pool_list, 
                                       four_players_open_hands_list,
                                       discarded_tile)

                    if len(draw_tile_list) != len(hands_list) or len(draw_tile_list) != len(discarded_tiles_pool_list) or len(draw_tile_list) != len(four_players_open_hands_list) or len(draw_tile_list) != len(discarded_tile):
                        print("false")

                    transform_136_to_34m4(hands_list, four_players_open_hands_list, discarded_tiles_pool_list)

                    draw_tile_np = np.array(draw_tile_list)
                    hands_list_np = np.array(hands_list)
                    discarded_tiles_pool_np = np.array(discarded_tiles_pool_list)
                    four_players_open_hands_np = np.array(four_players_open_hands_list)
                    discarded_tile_np = np.array(discarded_tile)

                    if is_init:
                        f.create_dataset('draw_tile', data = draw_tile_np, shape = None, dtype = None, chunks=True, maxshape=(None,))
                        f.create_dataset('hands', data = hands_list_np, shape = None, dtype = None, chunks=True, maxshape=(None, 4, 34)) 
                        f.create_dataset('discarded_tiles_pool', data = discarded_tiles_pool_np, shape = None, dtype = None, chunks=True, maxshape=(None, 4, 34)) 
                        f.create_dataset('four_players_open_hands', data = four_players_open_hands_np, shape = None, dtype = None, chunks=True, maxshape=(None, 4, 4, 34)) 
                        f.create_dataset('discarded_tile', data = discarded_tile_np, shape = None, dtype = None, chunks=True, maxshape=(None,)) 
                        is_init = False

                    else:
                        f['draw_tile'].resize((f["draw_tile"].shape[0] + draw_tile_np.shape[0]), axis = 0)
                        f['draw_tile'][-draw_tile_np.shape[0]:] = draw_tile_np

                        f['hands'].resize((f["hands"].shape[0] + hands_list_np.shape[0]), axis = 0)
                        f['hands'][-hands_list_np.shape[0]:] = hands_list_np

                        f['discarded_tiles_pool'].resize((f["discarded_tiles_pool"].shape[0] + discarded_tiles_pool_np.shape[0]), axis = 0)
                        f['discarded_tiles_pool'][-discarded_tiles_pool_np.shape[0]:] = discarded_tiles_pool_np

                        f['four_players_open_hands'].resize((f["four_players_open_hands"].shape[0] + four_players_open_hands_np.shape[0]), axis = 0)
                        f['four_players_open_hands'][-four_players_open_hands_np.shape[0]:] = four_players_open_hands_np

                        f['discarded_tile'].resize((f["discarded_tile"].shape[0] + discarded_tile_np.shape[0]), axis = 0)
                        f['discarded_tile'][-discarded_tile_np.shape[0]:] = discarded_tile_np



       
        f.close() 


