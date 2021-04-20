class ExperienceCollector:
    def __init__(self, model_type, buffer):
        self.model_type = model_type
        self.states = []
        self.actions = []
        self.importances = []
        self.rewards = []
        self.buffer = buffer

        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_importances = []
    
    def record_decision(self, state, action, importance):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)
        self.current_episode_importances.append(importance)

    def start_episode(self):
        self.current_episode_actions = []
        self.current_episode_states = []
        self.current_episode_importances = []

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        self.importances += self.current_episode_importances
        self.rewards += [reward] * num_states

        self.current_episode_actions = []
        self.current_episode_states = []
        self.current_episode_importances = []
    
    def to_buffer(self):
        
        for sample in list(zip(self.states, self.rewards, self.importances, self.actions)):
            self.buffer[self.model_type].store.remote(*sample)
        self.states, self.rewards, self.actions, self.importances = [], [], [], []