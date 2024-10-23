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
