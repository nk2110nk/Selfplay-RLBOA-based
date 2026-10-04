import csv

from results import FIELDS, result_path, write_results
from selfplay.evaluator import balanced_schedule, evaluation_seed
from train import parse_args


def test_balanced_schedule_has_both_slots():
    schedule = balanced_schedule(["Laptop"], ["Boulware"], ["case1"], 1)
    assert {row[3] for row in schedule} == {0, 1}


def test_evaluation_seed_stays_numpy_compatible_at_50000_steps():
    seed = evaluation_seed(0, 50000, 0, 0)
    assert seed == 50000 * 100003 % (2 ** 32)
    assert 0 <= seed < 2 ** 32
    assert evaluation_seed(0, 50000, 0, 1) == seed + 1


def test_exact_frequency_validation():
    try:
        parse_args(["--issue", "Laptop", "--agents", "Boulware",
                    "--n-envs", "3", "--snapshot-freq", "10000"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("non-divisible event frequency was accepted")


def test_tsv_header_matches_body(tmp_path):
    row = {field: 0 for field in FIELDS}
    path = write_results(tmp_path / "result.tsv", [row])
    with path.open() as handle:
        records = list(csv.reader(handle, delimiter="\t"))
    assert len(records[0]) == len(records[1]) == len(FIELDS)


def test_result_path_is_inside_model_csv(tmp_path):
    path = result_path(tmp_path, ("Boulware", "Linear"), "Laptop")
    assert path == (
        tmp_path / "csv" / "Boulware-Linear" / "Laptop" /
        "det=False_noise=False" / "Laptop-Boulware-Linear-dF-nF.tsv"
    )
