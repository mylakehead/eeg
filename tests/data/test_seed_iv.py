from data.seed_iv import sessions, session_label, Gender, gender_subject


def test_session_label():
    # real data from SEED-IV dataset
    origin = {
        1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    }

    assert len(session_label[1]) == len(origin[1])
    assert len(session_label[2]) == len(origin[2])
    assert len(session_label[3]) == len(origin[3])

    for s in sessions:
        for i, v in enumerate(session_label[s]):
            assert v.value == origin[s][i]


def test_gender_subject():
    origin = {
        0: [3, 4, 5, 8, 9, 10, 11, 14, 15],
        1: [1, 2, 6, 7, 12, 13]
    }

    for g in Gender:
        for i, s in enumerate(gender_subject[g]):
            assert s.value == origin[g.value][i]
