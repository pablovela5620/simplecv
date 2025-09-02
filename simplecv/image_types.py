from jaxtyping import UInt8
from numpy import ndarray

# Single BGR image type (H×W×3 array of uint8)
type ImageBGR = UInt8[ndarray, "H W 3"]
# List of BGR images type
type BGRList = list[ImageBGR]
