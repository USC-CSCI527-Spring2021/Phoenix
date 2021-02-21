from game.ai.configs.default import BotDefaultConfig
from game.ai.placement import PlacementHandler


class MaiConfig(BotDefaultConfig):
    name = "Mai"

    PLACEMENT_HANDLER_CLASS = PlacementHandler

    TUNE_DANGER_BORDER_TEMPAI_VALUE = 3
    TUNE_DANGER_BORDER_1_SHANTEN_VALUE = 0
    TUNE_DANGER_BORDER_2_SHANTEN_VALUE = 0
