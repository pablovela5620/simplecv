AVP_LINKS = (
    # Spine
    (0, 61),  # hip -> spine1
    (61, 62),  # spine1 -> spine2
    (62, 63),  # spine2 -> spine3
    (63, 64),  # spine3 -> spine4
    (64, 65),  # spine4 -> spine5
    (65, 66),  # spine5 -> spine6
    (66, 67),  # spine6 -> spine7
    # Neck
    (67, 29),  # spine7 -> neck1
    (29, 30),  # neck1 -> neck2
    (30, 31),  # neck2 -> neck3
    (31, 32),  # neck3 -> neck4
    # Left Arm
    (67, 24),  # spine7 -> leftShoulder
    (24, 1),  # leftShoulder -> leftArm
    (1, 2),  # leftArm -> leftForearm
    (2, 3),  # leftForearm -> leftHand (wrist)
    # Right Arm
    (67, 56),  # spine7 -> rightShoulder
    (56, 33),  # rightShoulder -> rightArm
    (33, 34),  # rightArm -> rightForearm
    (34, 35),  # rightForearm -> rightHand (wrist)
    # Left Hand
    # Thumb
    (3, 27),  # leftHand -> leftThumbKnuckle
    (27, 25),  # leftThumbKnuckle -> leftThumbIntermediateBase
    (25, 26),  # leftThumbIntermediateBase -> leftThumbIntermediateTip
    (26, 28),  # leftThumbIntermediateTip -> leftThumbTip
    # Index Finger
    (3, 7),  # leftHand -> leftIndexFingerMetacarpal
    (7, 6),  # leftIndexFingerMetacarpal -> leftIndexFingerKnuckle
    (6, 4),  # leftIndexFingerKnuckle -> leftIndexFingerIntermediateBase
    (4, 5),  # leftIndexFingerIntermediateBase -> leftIndexFingerIntermediateTip
    (5, 8),  # leftIndexFingerIntermediateTip -> leftIndexFingerTip
    # Middle Finger
    (3, 17),  # leftHand -> leftMiddleFingerMetacarpal
    (17, 16),  # leftMiddleFingerMetacarpal -> leftMiddleFingerKnuckle
    (16, 14),  # leftMiddleFingerKnuckle -> leftMiddleFingerIntermediateBase
    (14, 15),  # leftMiddleFingerIntermediateBase -> leftMiddleFingerIntermediateTip
    (15, 18),  # leftMiddleFingerIntermediateTip -> leftMiddleFingerTip
    # Ring Finger
    (3, 22),  # leftHand -> leftRingFingerMetacarpal
    (22, 21),  # leftRingFingerMetacarpal -> leftRingFingerKnuckle
    (21, 19),  # leftRingFingerKnuckle -> leftRingFingerIntermediateBase
    (19, 20),  # leftRingFingerIntermediateBase -> leftRingFingerIntermediateTip
    (20, 23),  # leftRingFingerIntermediateTip -> leftRingFingerTip
    # Little Finger
    (3, 12),  # leftHand -> leftLittleFingerMetacarpal
    (12, 11),  # leftLittleFingerMetacarpal -> leftLittleFingerKnuckle
    (11, 9),  # leftLittleFingerKnuckle -> leftLittleFingerIntermediateBase
    (9, 10),  # leftLittleFingerIntermediateBase -> leftLittleFingerIntermediateTip
    (10, 13),  # leftLittleFingerIntermediateTip -> leftLittleFingerTip
    # Right Hand
    # Thumb
    (35, 59),  # rightHand -> rightThumbKnuckle
    (59, 57),  # rightThumbKnuckle -> rightThumbIntermediateBase
    (57, 58),  # rightThumbIntermediateBase -> rightThumbIntermediateTip
    (58, 60),  # rightThumbIntermediateTip -> rightThumbTip
    # Index Finger
    (35, 39),  # rightHand -> rightIndexFingerMetacarpal
    (39, 38),  # rightIndexFingerMetacarpal -> rightIndexFingerKnuckle
    (38, 36),  # rightIndexFingerKnuckle -> rightIndexFingerIntermediateBase
    (36, 37),  # rightIndexFingerIntermediateBase -> rightIndexFingerIntermediateTip
    (37, 40),  # rightIndexFingerIntermediateTip -> rightIndexFingerTip
    # Middle Finger
    (35, 49),  # rightHand -> rightMiddleFingerMetacarpal
    (49, 48),  # rightMiddleFingerMetacarpal -> rightMiddleFingerKnuckle
    (48, 46),  # rightMiddleFingerKnuckle -> rightMiddleFingerIntermediateBase
    (46, 47),  # rightMiddleFingerIntermediateBase -> rightMiddleFingerIntermediateTip
    (47, 50),  # rightMiddleFingerIntermediateTip -> rightMiddleFingerTip
    # Ring Finger
    (35, 54),  # rightHand -> rightRingFingerMetacarpal
    (54, 53),  # rightRingFingerMetacarpal -> rightRingFingerKnuckle
    (53, 51),  # rightRingFingerKnuckle -> rightRingFingerIntermediateBase
    (51, 52),  # rightRingFingerIntermediateBase -> rightRingFingerIntermediateTip
    (52, 55),  # rightRingFingerIntermediateTip -> rightRingFingerTip
    # Little Finger
    (35, 44),  # rightHand -> rightLittleFingerMetacarpal
    (44, 43),  # rightLittleFingerMetacarpal -> rightLittleFingerKnuckle
    (43, 41),  # rightLittleFingerKnuckle -> rightLittleFingerIntermediateBase
    (41, 42),  # rightLittleFingerIntermediateBase -> rightLittleFingerIntermediateTip
    (42, 45),  # rightLittleFingerIntermediateTip -> rightLittleFingerTip
)

AVP_ID2NAME: dict[int, str] = {
    0: "hip",
    1: "leftArm",
    2: "leftForearm",
    3: "leftHand",  # Typically wrist
    4: "leftIndexFingerIntermediateBase",
    5: "leftIndexFingerIntermediateTip",
    6: "leftIndexFingerKnuckle",
    7: "leftIndexFingerMetacarpal",
    8: "leftIndexFingerTip",
    9: "leftLittleFingerIntermediateBase",
    10: "leftLittleFingerIntermediateTip",
    11: "leftLittleFingerKnuckle",
    12: "leftLittleFingerMetacarpal",
    13: "leftLittleFingerTip",
    14: "leftMiddleFingerIntermediateBase",
    15: "leftMiddleFingerIntermediateTip",
    16: "leftMiddleFingerKnuckle",
    17: "leftMiddleFingerMetacarpal",
    18: "leftMiddleFingerTip",
    19: "leftRingFingerIntermediateBase",
    20: "leftRingFingerIntermediateTip",
    21: "leftRingFingerKnuckle",
    22: "leftRingFingerMetacarpal",
    23: "leftRingFingerTip",
    24: "leftShoulder",
    25: "leftThumbIntermediateBase",
    26: "leftThumbIntermediateTip",
    27: "leftThumbKnuckle",
    28: "leftThumbTip",
    29: "neck1",
    30: "neck2",
    31: "neck3",
    32: "neck4",
    33: "rightArm",
    34: "rightForearm",
    35: "rightHand",  # Typically wrist
    36: "rightIndexFingerIntermediateBase",
    37: "rightIndexFingerIntermediateTip",
    38: "rightIndexFingerKnuckle",
    39: "rightIndexFingerMetacarpal",
    40: "rightIndexFingerTip",
    41: "rightLittleFingerIntermediateBase",
    42: "rightLittleFingerIntermediateTip",
    43: "rightLittleFingerKnuckle",
    44: "rightLittleFingerMetacarpal",
    45: "rightLittleFingerTip",
    46: "rightMiddleFingerIntermediateBase",
    47: "rightMiddleFingerIntermediateTip",
    48: "rightMiddleFingerKnuckle",
    49: "rightMiddleFingerMetacarpal",
    50: "rightMiddleFingerTip",
    51: "rightRingFingerIntermediateBase",
    52: "rightRingFingerIntermediateTip",
    53: "rightRingFingerKnuckle",
    54: "rightRingFingerMetacarpal",
    55: "rightRingFingerTip",
    56: "rightShoulder",
    57: "rightThumbIntermediateBase",
    58: "rightThumbIntermediateTip",
    59: "rightThumbKnuckle",
    60: "rightThumbTip",
    61: "spine1",
    62: "spine2",
    63: "spine3",
    64: "spine4",
    65: "spine5",
    66: "spine6",
    67: "spine7",
}
AVP_IDS: list[int] = [int(key) for key in AVP_ID2NAME]
