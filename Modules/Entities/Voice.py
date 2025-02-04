import numpy as np
from dataclasses import dataclass

@dataclass
class VoiceDataContainer:
    h_num       : int
    q_num       : int
    sid         : str
    gender      : bool
    voice_data  : np.float32



