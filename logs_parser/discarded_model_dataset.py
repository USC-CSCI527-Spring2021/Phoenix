import copy
import xml.etree.ElementTree as ET

import apache_beam as beam

from logs_parser.parser import parse_mjlog


def get_player_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands):
	return {
		'closed_hand:': four_closed_hands[player_id],
		'open_hand': four_open_hands[player_id],
		'discarded_tiles': four_discarded_hands[player_id]
	}


def get_enemy(enemy_id, four_open_hands, four_discarded_hands):
    
    return {
        'enemy_seat' : enemy_id,
        'open_hand' : four_open_hands[enemy_id],
        'discarded_tiles' : four_discarded_hands[enemy_id]
    }

def get_three_enemies_hands(player_id, four_open_hands, four_discarded_hands):
    enemies_tiles = []
    for i in range(4):
        if player_id != i:
            enemies_tiles.append(get_enemy(i, four_open_hands, four_discarded_hands))
            
    return enemies_tiles

def get_last_discarded_tile(player_id, four_discarded_hands):
    if len(four_discarded_hands[player_id]) == 0:
        return None
    return four_discarded_hands[player_id][-1]


def get_player_call_actions_record(player_id, call_actions_record):
    return call_actions_record[player_id]

def closed_hands_append(closed_hands, player_id, tile):
	    closed_hands[player_id] = copy.copy(closed_hands[player_id])
	    closed_hands[player_id].append(tile)
	    return closed_hands

def closed_hands_remove(closed_hands, player_id, tile):
    closed_hands[player_id] = copy.copy(closed_hands[player_id])
    closed_hands[player_id].remove(tile)
    return closed_hands

def open_hands_append(open_hands, caller, tile):
    open_hands[caller] = copy.copy(open_hands[caller])
    open_hands[caller].append(tile)
    return open_hands

def discarded_hands_append(discarded_hands, player_id, tile):
    discarded_hands[player_id] = copy.copy(discarded_hands[player_id])
    discarded_hands[player_id].append(tile)
    
    return discarded_hands

# remove tile from closed_hands[caller], and add it into open_hands[caller]
def trans_to_open(tiles, closed_hands, open_hands, caller): 
    
    for tile in tiles:   
        if tile in closed_hands[caller]:
            closed_hands = closed_hands_remove(closed_hands, caller, tile)
            
        if tile not in open_hands[caller]:
            open_hands = open_hands_append(open_hands, caller, tile)
    
    return closed_hands, open_hands

def get_tiles_for_mikan(tiles):
    start = tiles[0] // 4 * 4

    res = []
    for i in range(4):
        res.append(start + i)

    return res


def transfrom_136_to_34(player_hands_136):
    player_hands_34 = []
    for hand in player_hands_136:
        hand_34 = hand // 4
        player_hands_34.append(hand_34)
        
    return player_hands_34


def init_player_winds(dealer):
    if dealer == 0:
        player_winds = ['E', 'S', 'W', 'N']
    elif dealer == 1:
        player_winds = ['N', 'E', 'S', 'W']
    elif dealer == 2:
        player_winds = ['W', 'N', 'E', 'S']
    else:
        player_winds = ['S', 'W', 'N', 'E']
        
    return player_winds


def getPrevailingWind(round_num):
    prevailing_winds = ['E', 'S', 'W', 'N']

    return prevailing_winds[round_num // 4]


class DiscardedFeatureExtractor(beam.DoFn):

	def __init__(self, *unused_args, **unused_kwargs):
		super().__init__(*unused_args, **unused_kwargs)

		self.player_id_list = []
		self.four_closed_hands_list = []
		self.four_discarded_hands_list = []
		self.four_open_hands_list = []
		self.dora_list = []
		self.action_list = []
		self.scores_list = []
		self.call_actions_record_list = []
		self.four_player_open_hands_detail_list = []
		self.draw_list = []
		# paired data
		self.player_tiles_list = []
		self.enemies_tiles_list = []
		self.riichi_bets_list = []
		self.last_player_discarded_tile_list = []
		self.player_call_actions_record_list = []
		self.last_player_id_list = []
		self.open_hands_detail_list = []

	# record cur four players' states, update closed and open hands after call
	def call_operations(self, action_data, closed_hands, open_hands, taken_call_actions, open_hands_details):
		caller = action_data["caller"]
		tiles = action_data["mentsu"]
		call_type = action_data["call_type"]

		if call_type == "AnKan":
			tiles = get_tiles_for_mikan(tiles)

		# record the actions has been taken
		taken_call_actions[caller] = copy.copy(taken_call_actions[caller])
		taken_call_actions[caller].append(call_type)

		closed_hands = copy.copy(closed_hands)
		open_hands = copy.copy(open_hands)

		reacted_tile = None
		for tile in tiles:
			if tile not in closed_hands[caller]:
				reacted_tile = tile

		if call_type == 'Pon' or call_type == 'Chi' or call_type == "MinKan":
			self.draw_list.append([call_type, reacted_tile])

		closed_hands, open_hands = trans_to_open(tiles, closed_hands, open_hands, caller)

		open_hands_details[caller] = copy.copy(open_hands_details[caller])
		open_hands_details[caller].append({"tiles": tiles, "meld_type": call_type, "reacted_tile": reacted_tile})

		return closed_hands, open_hands, open_hands_details

	def record_cur_state(self, player_id, closed_hands, open_hands, discarded_hands, doras, riichi_bets, scores,
						 taken_call_actions, open_hands_details):
	    self.player_id_list.append(player_id)
	    self.four_closed_hands_list.append(copy.copy(closed_hands))
	    self.four_open_hands_list.append(copy.copy(open_hands))
	    self.four_discarded_hands_list.append(copy.copy(discarded_hands))
	    self.dora_list.append(copy.copy(doras))
	    self.scores_list.append(copy.copy(scores))
	    self.riichi_bets_list.append(riichi_bets)
	    self.call_actions_record_list.append(copy.copy(taken_call_actions))
	    self.four_player_open_hands_detail_list.append(copy.copy(open_hands_details))

	def get_round_info(self, dataset):
	    
	    init_data = dataset[0]["data"]
	    scores = init_data["scores"]
	    riichi_bets = init_data['reach']
	    doras = [init_data["dora"]]

	    # player_tiles, get from init 
	    closed_hands = copy.copy(init_data["hands"])
	    open_hands = [[], [], [], []]
	    discarded_hands = [[], [], [], []]

	    # updated 
	    taken_call_actions = [[], [], [], []]
	    open_hands_details = [[], [], [], []]

	    for k in range(len(dataset)):
	    
	        action = dataset[k]
	        act = action["tag"]

	#       {'tag': 'REACH', 'data': {'player': 3, 'step': 1}}
	#       {'tag': 'REACH', 'data': {'player': 3, 'step': 2, 'scores': [17200, 26900, 40900, 14000]}}
	        if act == 'REACH':
	            player_id = action["data"]["player"]
	            if action['data']['step'] == 1:

	                taken_call_actions[player_id] = copy.copy(taken_call_actions[player_id])
	                taken_call_actions[player_id].append(act)

	            if action['data']['step'] == 2:

	                riichi_bets += 1
	                
	                if 'scores' not in action['data']:
	                    scores[player_id] -= 1000
	                else:
	                    scores = action['data']['scores']

	#       {'tag': 'DORA', 'data': {'hai': 113}}
	        elif act == 'DORA': 
	            doras.append(action['data']['hai'])
	            
	        elif act == "CALL": 
	            action_data = action["data"]
	            # last_player_id =  player_id
	            closed_hands, open_hands, open_hands_details = self.call_operations(action_data, closed_hands, open_hands, taken_call_actions, open_hands_details)
	           
	        # deal with draw and discard operations  
	        elif act in ["DRAW", "DISCARD"]:
	            player_id = action["data"]["player"]
	            tile = action["data"]["tile"]
	            
	            # 记录作出这个action之前场面的情况，当前玩家，与其他玩家，摸牌
	            if act == "DRAW":          
	                self.draw_list.append([act, tile])
	                # update state 
	                closed_hands = closed_hands_append(closed_hands, player_id, tile)
	                
	            # update discarded_hands and four_closed_hands for cur player    
	            elif act == "DISCARD":

	                # record cur state for four players, and cur player id
	                self.record_cur_state(player_id, closed_hands, open_hands, discarded_hands, doras, riichi_bets, scores, taken_call_actions, open_hands_details) 
	                # last_player_id = player_id == last_player_id ? last_player_id : player_id      
	                self.action_list.append(tile)


	                closed_hands = closed_hands_remove(closed_hands, player_id, tile)
	                discarded_hands = discarded_hands_append(discarded_hands, player_id, tile)
	                
	        elif act == 'AGARI' or act == 'RYUUKYOKU':    
	            continue


	def pair_data(self):
	    last_player =  -1
	    for i in range(len(self.player_id_list)):

	        self.last_player_id_list.append(last_player)
	        player_id = self.player_id_list[i]
	        four_closed_hands  = self.four_closed_hands_list[i]
	        four_open_hands = self.four_open_hands_list[i]
	        four_discarded_hands = self.four_discarded_hands_list[i]

	        call_actions_record = self.call_actions_record_list[i]
	        open_hands_details = self.four_player_open_hands_detail_list[i]

			# separate_player_enemies_hands
			self.player_tiles_list.append(
				get_player_hands(player_id, four_closed_hands, four_open_hands, four_discarded_hands))
			self.enemies_tiles_list.append(get_three_enemies_hands(player_id, four_open_hands, four_discarded_hands))

			self.open_hands_detail_list.append(open_hands_details[player_id])

			# get player call_actions_record firstly, if reached, continue
			player_call_actions_record = get_player_call_actions_record(player_id, call_actions_record)
			self.player_call_actions_record_list.append(player_call_actions_record)

			# get discard tiles
			if i == 0:
				last_player_tile_136 = None
				self.last_player_discarded_tile_list.append(last_player_tile_136)

				if (i + 1) < len(self.player_id_list) and player_id == self.player_id_list[i + 1]:
					last_player = last_player
				else:
					last_player = player_id
			else:
				last_player_tile_136 = get_last_discarded_tile(last_player, four_discarded_hands)
	            self.last_player_discarded_tile_list.append(last_player_tile_136)

	            if (i + 1) < len(self.player_id_list) and player_id == self.player_id_list[i+1]:
	                last_player = last_player
	            else:
	                last_player = player_id

	def process(self, log_data, **kwargs):
		xml_str = log_data	

		if type(xml_str) == str:
			try:
				node = ET.fromstring(xml_str)
				data = parse_mjlog(node)
				# remove three-players games
				if len(data["meta"]['UN'][3]["name"]) != 0:
					for j in range(len(data["rounds"])):
						self.player_id_list = []
						self.four_closed_hands_list = []
						self.four_discarded_hands_list = []
						self.four_open_hands_list = []
						self.dora_list = []
						self.action_list = []
						self.scores_list = []
						self.player_tiles_list = []
						self.enemies_tiles_list = []
						self.riichi_bets_list = []
						self.last_player_discarded_tile_list = []

						self.call_actions_record_list = []
						self.player_call_actions_record_list = []
						self.four_player_open_hands_detail_list = []
						self.draw_list = []

						self.last_player_id_list = []
						self.open_hands_detail_list = []

						dealer = int(data['rounds'][j][0]["data"]["oya"])
						repeat_dealer = int(data['rounds'][j][0]["data"]['combo'])
						players_winds = init_player_winds(dealer)

						round_num = int(data['rounds'][j][0]["data"]['round'])
						prevailing_wind = getPrevailingWind(round_num)

						self.get_round_info(data['rounds'][j])
						self.pair_data()

						for k in range(len(self.player_id_list)):

							if "REACH" in self.player_call_actions_record_list[k]:
								continue

							res = {
	                            'player_id': self.player_id_list[k],
	                            'draw_tile': self.draw_list[k],
	                            'player_tiles': self.player_tiles_list[k],
	                            'open_hands_detail': self.open_hands_detail_list[k],
	                            'enemies_tiles': self.enemies_tiles_list[k],
	                            'dealer': dealer,
	                            'repeat_dealer': repeat_dealer,
	                            'riichi_bets' : self.riichi_bets_list[k],
	                            'player_wind': players_winds[self.player_id_list[k]],
	                            'prevailing_wind': prevailing_wind,
	                            'dora': self.dora_list[k],
	                            'scores': self.scores_list[k],
	                            'discarded_tile': self.action_list[k]
							}
							# print(res)
							yield res
			except:
				pass

