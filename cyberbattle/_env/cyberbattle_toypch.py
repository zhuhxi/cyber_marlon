# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.toypch import toy_pch
from . import cyberbattle_env


class CyberBattleToyPch(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on a toy Pch exercise"""

    def __init__(self, config_list, **kwargs):
        super().__init__(
            initial_environment=toy_pch.new_environment(config_list=config_list),
            **kwargs)
