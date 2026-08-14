"""WolfPlay multi-agent Werewolf framework."""

from .engine import GameRuntime
from .models import Faction, Phase, Role, Winner

__all__ = ["Faction", "GameRuntime", "Phase", "Role", "Winner"]
__version__ = "1.0.0"
