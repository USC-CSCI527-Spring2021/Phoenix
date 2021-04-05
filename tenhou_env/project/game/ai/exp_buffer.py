class ExperienceBuffer:
    def __init__(self, state, action, importance, rewards):
        self.state = state
        self.action = action
        self.importance = importance
        self.rewards = rewards

class ExperienceCollector:
    def __init__(self, model_type):
        self.model_type = model_type
        self.states = []
        self.actions = []
        self.importances = []
        self.rewards = []

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
        pass
