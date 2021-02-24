import numpy as np

#Author: Jingwen Sun
#This is a class used for extract features from a tiles_state_and_action list for chi/pon/kan model.
#getKanFeatures() is not completed yet
#usage: 
#   tiles_state_and_action = np.load('tiles_state_and_action_2021.npy')
#   ef = extractFeature(tiles_state_and_action)
#   xs, ys = ef.getChiFeatures()
#       xs is a list of (63,34) vectors and ys is a list of 1/0
#   xs, ys = ef.getPonFeatures()
#       xs is a list of (63,34) vectors and ys is a list of 1/0

class extractFeature:
    def __init__(self, tiles_state_and_action):
        self.tiles_state_and_action = tiles_state_and_action
    
    def getPlayerTiles(self, player_tiles):
        closed_hand_136 = player_tiles['closed_hand:']
        open_hand_136 = player_tiles['open_hand']
        discarded_tiles_136 = player_tiles['discarded_tiles']
        
        closed_hand_feature = np.zeros((4, 34))
        open_hand_feature = np.zeros((4, 34))
        discarded_tiles_feature = np.zeros((4, 34))
        
        for val in closed_hand_136:
            idx=0
            while(closed_hand_feature[idx][val//4] == 1):
                idx+=1
            closed_hand_feature[idx][val//4] = 1
        for val in open_hand_136:
            idx=0
            while(open_hand_feature[idx][val//4] == 1):
                idx+=1
            open_hand_feature[idx][val//4] = 1
        for val in discarded_tiles_136:
            idx=0
            while(discarded_tiles_feature[idx][val//4] == 1):
                idx+=1
            discarded_tiles_feature[idx][val//4] = 1
        
        return np.concatenate((closed_hand_feature,open_hand_feature,discarded_tiles_feature))
    
    def getSelfTiles(self, i):
        player_tiles = self.tiles_state_and_action[6][i]
        return self.getPlayerTiles(player_tiles)       

    def getEnemiesTiles(self,i):
        player_seat = self.tiles_state_and_action[0][i]
        enemies_tiles_list = self.tiles_state_and_action[7][i]
        if(len(enemies_tiles_list)==3):
            enemies_tiles_list.insert(player_seat, {})
        enemies_tiles_feature = np.empty((0, 34))
        for i in range(3):
            player_seat = (player_seat+1)%4
            player_tiles = self.getPlayerTiles(enemies_tiles_list[player_seat])
            enemies_tiles_feature = np.concatenate((enemies_tiles_feature,player_tiles))
        return enemies_tiles_feature
        
    def getDoraList(self, i):
        dora_list = self.tiles_state_and_action[8][i]
        dora_feature = np.zeros((5, 34))
        for idx, val in enumerate(dora_list):
            dora_feature[idx][val//4] = 1
        return dora_feature
    
    #get score features in one (1,34) vector
    def getScoreList1(self,i):
        def trans_score(score):
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
            
        player_seat = self.tiles_state_and_action[0][i]
        scores_list = self.tiles_state_and_action[9][i]
        scores_feature = np.zeros((4, 34))
        for i in range(4):
            score = scores_list[player_seat]
            scores_feature[i] = trans_score(score)
            player_seat += 1
            player_seat %= 4
        return scores_feature
    
    #get board information features in multiple (1,34) vectors
    def getBoard(self, i):
        def wind(c):
            if c=='E' : 
                return 0
            if c=='S' : 
                return 1
            if c=='W' : 
                return 2
            if c=='N' : 
                return 3
        player_seat = self.tiles_state_and_action[0][i]
        dealer_seat = self.tiles_state_and_action[1][i]
        repeat_dealer = self.tiles_state_and_action[2][i]
        riichi_bets = self.tiles_state_and_action[3][i]
        player_wind = self.tiles_state_and_action[4][i]
        prevailing_wind = self.tiles_state_and_action[5][i]
        
        dealer_feature = np.zeros((4, 34))
        dealer_feature[(dealer_seat+4-player_seat)%4] = 1
        
        repeat_dealer_feature = np.zeros((8, 34))
        if repeat_dealer > 7 :
            repeat_dealer = 7
        repeat_dealer_feature[repeat_dealer] = 1
        
        riichi_bets_feature = np.zeros((8, 34))
        if riichi_bets > 7 :
            repeat_dealer = 7
        riichi_bets_feature[riichi_bets] = 1
        
        player_wind_feature = np.zeros((4, 34))
        player_wind_feature[(wind(player_wind)+4-player_seat)%4] = 1
        
        prevailing_wind_feature = np.zeros((4, 34))
        prevailing_wind_feature[(wind(prevailing_wind)+4-player_seat)%4] = 1
        
        return np.concatenate((
            dealer_feature,
            repeat_dealer_feature,
            riichi_bets_feature,
            player_wind_feature,
            prevailing_wind_feature
        ))
    
    #get board information features in one (1,34) vector
    def getBoard1(self, i):
        def wind(c):
            if c=='E' : 
                return 27
            if c=='S' : 
                return 28
            if c=='W' : 
                return 29
            if c=='N' : 
                return 30
        player_seat = self.tiles_state_and_action[0][i]
        dealer_seat = self.tiles_state_and_action[1][i]
        repeat_dealer = self.tiles_state_and_action[2][i]
        riichi_bets = self.tiles_state_and_action[3][i]
        player_wind = self.tiles_state_and_action[4][i]
        prevailing_wind = self.tiles_state_and_action[5][i]
        
        dealer_feature = np.zeros((1, 34))
        dealer_feature[0][(dealer_seat+4-player_seat)%4] = 1
        
        repeat_dealer_feature = np.zeros((1, 34))
        repeat_dealer_feature[0][repeat_dealer] = 1
        
        riichi_bets_feature = np.zeros((1, 34))
        riichi_bets_feature[0][riichi_bets] = 1
        
        player_wind_feature = np.zeros((1, 34))
        player_wind_feature[0][wind(player_wind)] = 1
        
        prevailing_wind_feature = np.zeros((1, 34))
        prevailing_wind_feature[0][wind(prevailing_wind)] = 1
        
        return np.concatenate((
            dealer_feature,
            repeat_dealer_feature,
            riichi_bets_feature,
            player_wind_feature,
            prevailing_wind_feature
        ))
        
    def getGeneralFeature(self, i):
        return np.concatenate((            
            self.getSelfTiles(i),
            self.getDoraList(i),
            self.getBoard1(i),
            self.getEnemiesTiles(i),
            self.getScoreList1(i)
        ))
           
    def getChiFeatures(self):
        xs = []
        ys = []
        for i in range(len(self.tiles_state_and_action[12])):
            could_chi = self.tiles_state_and_action[12][i]
            if could_chi == 1:
                last_player_discarded_tile = self.tiles_state_and_action[10][i]
                last_player_discarded_tile_feature = np.zeros((1, 34))
                last_player_discarded_tile_feature[0][last_player_discarded_tile//4] = 1
                x = np.concatenate((last_player_discarded_tile_feature,self.getGeneralFeature(i)))
                xs.append(x)
                    
                action = self.tiles_state_and_action[15][i]
                if action[0]=='Chi':
                    y = 1
                else:
                    y = 0
                ys.append(y)
                
        return xs, ys
    
    def getPonFeatures(self):
        def could_pon(closed_hand_136, last_discarded_tile):
            if last_discarded_tile == None:
                return False
            count = np.zeros(34)
            for tile in closed_hand_136:
                count[tile//4]+=1
            if count[last_discarded_tile//4] >= 2:
                return True
            return False
        
        xs = []
        ys = []
        for i in range(len(self.tiles_state_and_action[13])):
            if self.tiles_state_and_action[13][i] == 1:
                last_three_discarded_tile_list = self.tiles_state_and_action[11][i]
                closed_hand_136 = self.tiles_state_and_action[6][i]['closed_hand:']
                for last_discarded_tile in last_three_discarded_tile_list:
                    if could_pon(closed_hand_136, last_discarded_tile):
                        last_discarded_tile_feature = np.zeros((1, 34))
                        last_discarded_tile_feature[0][last_discarded_tile//4] = 1
                        x = np.concatenate((last_discarded_tile_feature,self.getGeneralFeature(i)))
                        xs.append(x)
                        
                        action = self.tiles_state_and_action[15][i]
                        if action[0]=='Pon' and (last_discarded_tile in action[1]):
                            y = 1
                        else:
                            y = 0
                        ys.append(y)
                
        return xs, ys

