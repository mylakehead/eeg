"""
Module: EEG Data Constants and File Mapping

Reference: https://bcmi.sjtu.edu.cn/~seed/seed-iv.html#

Copyright:
    MIT License

    Copyright © 2024 Lakehead University, Large Scale Data Analytics Group Project

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
    Kang Hong, XingJian Han, Minh Anh Nguyen
    hongkang@hongkang.name, xhan15@lakeheadu.ca, mnguyen9@lakeheadu.ca

Date:
    Created: 2024-10-02
    Last Modified: 2024-11-24
"""

import os
import re
from enum import Enum


class Gender(Enum):
    FEMALE = 0
    MALE = 1


class Emotion(Enum):
    NEUTRAL = 0
    SAD = 1
    FEAR = 2
    HAPPY = 3


class Band(Enum):
    DELTA = 0
    THETA = 1
    ALPHA = 2
    BETA = 3
    GAMMA = 4


class Subject(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    THIRTEEN = 13
    FOURTEEN = 14
    FIFTEEN = 15


class Session(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


gender_subject = {
    Gender.FEMALE: [
        Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.EIGHT, Subject.NINE,
        Subject.TEN, Subject.ELEVEN, Subject.FOURTEEN, Subject.FIFTEEN
    ],
    Gender.MALE: [
        Subject.ONE, Subject.TWO, Subject.SIX, Subject.SEVEN, Subject.TWELVE, Subject.THIRTEEN
    ]
}


# labels[<session>] = <trails>
session_label = {
    Session.ONE: [
        Emotion.SAD, Emotion.FEAR, Emotion.HAPPY, Emotion.NEUTRAL, Emotion.FEAR, Emotion.NEUTRAL,
        Emotion.NEUTRAL, Emotion.SAD, Emotion.NEUTRAL, Emotion.SAD, Emotion.FEAR, Emotion.SAD, Emotion.SAD,
        Emotion.SAD, Emotion.FEAR, Emotion.HAPPY, Emotion.FEAR, Emotion.FEAR, Emotion.HAPPY, Emotion.HAPPY,
        Emotion.NEUTRAL, Emotion.HAPPY, Emotion.NEUTRAL, Emotion.HAPPY
    ],
    Session.TWO: [
        Emotion.FEAR, Emotion.SAD, Emotion.HAPPY, Emotion.NEUTRAL, Emotion.NEUTRAL, Emotion.FEAR,
        Emotion.NEUTRAL, Emotion.FEAR, Emotion.HAPPY, Emotion.HAPPY, Emotion.FEAR, Emotion.HAPPY, Emotion.FEAR,
        Emotion.NEUTRAL, Emotion.SAD, Emotion.SAD, Emotion.FEAR, Emotion.SAD, Emotion.NEUTRAL, Emotion.HAPPY,
        Emotion.NEUTRAL, Emotion.SAD, Emotion.HAPPY, Emotion.SAD
    ],
    Session.THREE: [
        Emotion.SAD, Emotion.FEAR, Emotion.FEAR, Emotion.SAD, Emotion.HAPPY, Emotion.HAPPY, Emotion.HAPPY,
        Emotion.SAD, Emotion.SAD, Emotion.FEAR, Emotion.SAD, Emotion.NEUTRAL, Emotion.FEAR, Emotion.HAPPY,
        Emotion.HAPPY, Emotion.NEUTRAL, Emotion.FEAR, Emotion.HAPPY, Emotion.NEUTRAL, Emotion.NEUTRAL,
        Emotion.FEAR, Emotion.NEUTRAL, Emotion.SAD, Emotion.NEUTRAL
    ]
}


class FeatureMethod(Enum):
    DE_LDS = 0
    DE_MOVING_AVE = 1
    PSD_LDS = 2
    PSD_MOVING_AVE = 3


def subject_file_map(folder):
    """
    Maps EEG subject identifiers to their corresponding session file paths.

    Args:
        folder (str): Root directory containing EEG data organized by subject and session.
    """
    file_map = dict()
    for s in Subject:
        file_map[s] = dict()

    pattern = r'(\d+)_(\d{8})\.mat'
    for root, _, files in os.walk(folder):
        paths = root.split(os.sep)
        session = paths[-1]

        # Iterate through the files in the current folder
        for file_name in files:
            match = re.match(pattern, file_name)
            if match:
                s = int(match.group(1))
                _ = match.group(2)

                abs_path = os.path.join(root, file_name)

                # Map the file to the corresponding subject and session
                file_map[Subject(s)][Session(int(session))] = abs_path

    return file_map
