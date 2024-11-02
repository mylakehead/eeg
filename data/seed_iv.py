"""
Module Name: EEG Emotion Data Definition

Description:
    This module defines several enumerations and mappings for use in an EEG-based emotion recognition project.
    The enumerations represent categorical data such as gender, emotion types, subjects, sessions, and feature
    extraction methods. Additionally, mappings are defined to link gender with specific subjects and to label
    sessions with a sequence of emotional states.

    The module includes the following enumerations and dictionaries:
        - Gender: An enumeration for subject gender (FEMALE, MALE).
        - Emotion: An enumeration for different emotion labels (NEUTRAL, SAD, FEAR, HAPPY).
        - Subject: An enumeration for subject identifiers (ONE through FIFTEEN).
        - Session: An enumeration for session numbers (ONE, TWO, THREE).
        - FeatureMethod: An enumeration for feature extraction methods (DE_LDS, DE_MOVING_AVE, PSD_LDS, PSD_MOVING_AVE).
        - gender_subject: A dictionary mapping each gender to a list of associated subjects.
        - session_label: A dictionary mapping each session to a sequence of emotion labels for individual trials.

    Reference: https://bcmi.sjtu.edu.cn/~seed/seed-iv.html

Usage:
    These enumerations and mappings are used to provide structured data for training or evaluating emotion
    recognition models, and to ensure consistency in data processing and feature extraction.

License:
    MIT License

    Copyright © 2024 [Your University Name], [Course Name] Group Project

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
    Last Modified: 2024-11-02
"""

from enum import Enum


class Gender(Enum):
    FEMALE = 0
    MALE = 1


class Emotion(Enum):
    NEUTRAL = 0
    SAD = 1
    FEAR = 2
    HAPPY = 3


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
