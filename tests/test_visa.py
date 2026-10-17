import csv
from pathlib import Path, PureWindowsPath

import pytest

from anomcal.data.visa import (
    EXPECTED_1CLS_COUNTS,
    VISA_CLASSES,
    VisaFormatError,
    compare_with_expected,
    counts_by_class,
    find_split_csv,
    main,
    make_image_id,
    missing_files,
    read_split_csv,
    resolve_visa_root,
    to_local_path,
)

HEADER = ["object", "split", "label", "image", "mask"]


def write_csv(path: Path, rows) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def test_expected_counts_match_published_totals():
    # VisA paper: 10,821 images, 9,621 normal, 1,200 anomalous; 1cls split 8659/962/1200.
    train = sum(c[0] for c in EXPECTED_1CLS_COUNTS.values())
    test_n = sum(c[1] for c in EXPECTED_1CLS_COUNTS.values())
    test_a = sum(c[2] for c in EXPECTED_1CLS_COUNTS.values())
    assert (train, test_n, test_a) == (8659, 962, 1200)
    assert train + test_n == 9621 and train + test_n + test_a == 10821
    assert set(EXPECTED_1CLS_COUNTS) == set(VISA_CLASSES) and len(VISA_CLASSES) == 12


def test_read_fake_split(fake_records):
    assert len(fake_records) == 2 * (12 + 4 + 4)
    counts = counts_by_class(fake_records)
    assert counts == {"candle": (12, 4, 4), "pcb1": (12, 4, 4)}


def test_image_ids_are_unique_and_include_label():
    # VisA reuses numbers across Normal/0000.JPG and Anomaly/000.JPG folders.
    a = make_image_id("candle", 0, "candle/Data/Images/Normal/000.JPG")
    b = make_image_id("candle", 1, "candle/Data/Images/Anomaly/000.JPG")
    assert a != b and a == "candle/normal/000" and b == "candle/anomaly/000"


def test_paths_are_built_with_pathlib(fake_visa_root, fake_records):
    r = next(x for x in fake_records if x.label == 1)
    p = r.image_path(fake_visa_root)
    assert p.is_file() and p.parts[-4:] == ("Data", "Images", "Anomaly", "000.JPG")
    assert r.mask_path(fake_visa_root).is_file()
    assert next(x for x in fake_records if x.label == 0).mask_path(fake_visa_root) is None


def test_posix_relpath_maps_to_windows_path():
    # The CSV uses '/', the laptop uses '\\'. Joining via parts gives a native path.
    win = PureWindowsPath("C:/data/visa").joinpath(*"candle/Data/Images/Normal/0001.JPG".split("/"))
    assert str(win) == r"C:\data\visa\candle\Data\Images\Normal\0001.JPG"


@pytest.mark.parametrize("bad", ["../escape.JPG", "/abs/x.JPG", "candle/../../x.JPG"])
def test_paths_outside_root_are_rejected(tmp_path, bad):
    with pytest.raises(VisaFormatError):
        to_local_path(tmp_path, bad)


@pytest.mark.parametrize(
    "row, msg",
    [
        (["candle", "val", "normal", "candle/a.JPG", ""], "split"),
        (["candle", "train", "good", "candle/a.JPG", ""], "label"),
        (["candle", "test", "anomaly", "candle/a.JPG", ""], "without a mask"),
        (["candle", "train", "normal", "candle/a.JPG", "candle/m.png"], "with a mask"),
        (["candle", "train", "anomaly", "candle/a.JPG", "candle/m.png"], "train split"),
        (["candle", "train", "normal", "pcb1/a.JPG", ""], "not under class"),
        (["candle", "train", "normal"], "5 fields"),
    ],
)
def test_malformed_rows_are_rejected(tmp_path, row, msg):
    with pytest.raises(VisaFormatError, match=msg):
        read_split_csv(write_csv(tmp_path / "s.csv", [row]))


def test_bad_header_rejected(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("a,b,c,d,e\n", encoding="utf-8")
    with pytest.raises(VisaFormatError, match="header"):
        read_split_csv(p)


def test_duplicate_rows_rejected(tmp_path):
    row = ["candle", "train", "normal", "candle/Data/Images/Normal/0001.JPG", ""]
    with pytest.raises(VisaFormatError, match="duplicate"):
        read_split_csv(write_csv(tmp_path / "s.csv", [row, row]))


def test_crlf_csv_is_read(tmp_path):
    p = tmp_path / "s.csv"
    p.write_bytes(b"object,split,label,image,mask\r\ncandle,train,normal,candle/x/0001.JPG,\r\n")
    assert len(read_split_csv(p)) == 1


def test_resolve_root_accepts_nested_visa_folder(fake_visa_root, tmp_path):
    assert resolve_visa_root(fake_visa_root) == fake_visa_root
    nested = tmp_path / "data"
    (nested / "VisA" / "candle").mkdir(parents=True)
    assert resolve_visa_root(nested) == nested / "VisA"
    with pytest.raises(VisaFormatError):
        resolve_visa_root(tmp_path / "empty")


def test_find_split_csv(fake_visa_root, tmp_path):
    assert find_split_csv(fake_visa_root).name == "1cls.csv"
    with pytest.raises(FileNotFoundError, match="spot-diff"):
        find_split_csv(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_split_csv(fake_visa_root, tmp_path / "nope.csv")


def test_compare_with_expected():
    assert compare_with_expected(EXPECTED_1CLS_COUNTS) == []
    off = dict(EXPECTED_1CLS_COUNTS, candle=(899, 100, 100))
    assert compare_with_expected(off) == ["candle: got (899, 100, 100), expected (900, 100, 100)"]


def test_missing_files_detected(fake_visa_root, fake_records, tmp_path):
    assert missing_files(fake_visa_root, fake_records) == []
    assert len(missing_files(tmp_path, fake_records[:3])) == 3


def test_cli_reports_mismatch_on_fake_data(fake_visa_root, capsys):
    # The fake set is not the real VisA, so the count check must fail loudly (exit 1)...
    assert main(["--data-root", str(fake_visa_root)]) == 1
    out = capsys.readouterr().out
    assert "COUNT MISMATCH" in out and "candle" in out
    assert "all images and masks listed in the split exist" in out
