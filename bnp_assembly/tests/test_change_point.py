import pytest
import numpy as np
from bnp_assembly.change_point import  find_change_point
import matplotlib.pyplot as plt

from bnp_assembly.missing_data import find_clips

pairs = [([0, 1, 2, 5, 10, 20, 16, 19], 4),
         ([10, 20, 30, 1, 2, 3][::-1], 3)]


@pytest.mark.parametrize("data,expected", pairs)
def test_find_change_point(data, expected):
    assert find_change_point(data) == expected


def test_find_change_point2():
    data = np.array([5, 5, 5, 5])
    assert find_change_point(data) == 0


def test_find_change_point_real_case():
    data = np.load("missing_data_case.npy")
    data = data[::-1]

    change_point = find_change_point(data)
    correct = 26
    assert abs(change_point-correct) < 2

    no_change_point = data[26:]
    change_point = find_change_point(no_change_point)
    #assert change_point == 0


def test_find_change_point_real_case2():
    data = np.load("missing_data_case_bin_size1000.npy")
    change_point = find_change_point(data)
    print("CHange point", change_point)
    #plt.plot(data)
    #plt.show()

    clips = find_clips(data, None, None)
    print(clips)


def test_find_change_point_real_case3():
    data = np.load("missing_data_case_31.npy")
    change_point = find_change_point(data[:len(data)])
    print("CHange point", change_point)
    # plt.plot(data)
    # plt.show()

    clips = find_clips(data, None, None)
    assert abs(clips[0] - 564) < 10
    assert clips[1] == 0
    print(clips)


# to be run locally for debugging
@pytest.mark.skip
def test_find_change_points_real_data():
    path = "../../benchmarking/data/athalia_rosea/real/big/10/10000000/1/not_assembled/0/200/0.0/0/0.0/0.0/6000/bnp_scaffolding_dynamic_heatmaps/logging/missing_data/"

    correct = {
        0: (30, 16974),
        1: (834, None),
        2: (22, 15811),
        3: (30,  2350),
        4: (107, (2852, 2993)),
        5: (26, 2282),
        6: (22, None),
        7: (0, 4303),
        8: (31, 5034),
        9: (30, 6293),
        20: (185, None),
        21: (16, 10669),
        31: (573, None),
        32: (0, 4454),
        33: (44, None),
        34: (0, (833, 1090))


    }

    def clip_is_correct(clip, correct_clips, torrelance=0.1):
        if isinstance(correct_clips, int):
            correct_clips = (correct_clips,)

        for correct_clip in correct_clips:
            if correct_clip is None or correct_clip == 0:
                torrelance = 0
            else:
                torrelance = correct_clip * torrelance

            if abs(clip-correct_clip) <= torrelance:
                return True

        return False

    for contig in range(62):
        print("Contig", contig)
        bins = np.load(path + f"bin counts contig {contig}.npy")
        clips = find_clips(bins, None, None)
        if contig not in correct:
            continue

        start, end = clips
        correct_start = correct[contig][0]
        correct_end = correct[contig][1]
        if correct_end is None:
            correct_end = 0
        else:
            if isinstance(correct_end, int):
                correct_end = (correct_end,)

            correct_end = [len(bins)-e for e in correct_end]

        print(f"Correct: {correct_start}, {correct_end}. Found {start}, {end}")

        for dir, clip, correct_clip in (("start", start, correct_start), ("end", end, correct_end)):
            if not clip_is_correct(clip, correct_clip):
                print("Wrong clip", dir, clip, "Correct is", correct_clip)


        print("")


if __name__ == "__main__":
    test_find_change_points_real_data()

