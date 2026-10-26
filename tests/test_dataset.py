import csv
import pytest
from soundsight.dataset import records


def test_metadata_checks_duplicate_filenames(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "audio").mkdir()
    rows = []
    for target in range(50):
        for fold in range(1, 6):
            for take in range(8):
                name = f"{target}-{fold}-{take}.wav"
                (tmp_path / "audio" / name).touch()
                rows.append({"filename": name, "target": target, "fold": fold,
                             "category": f"class_{target}", "src_file": f"{target}-{fold}-{take}"})

    def write():
        with (tmp_path / "meta/esc50.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    write()
    assert len(records(tmp_path)) == 2000
    rows[8]["filename"] = rows[0]["filename"]
    write()
    with pytest.raises(ValueError, match="duplicate"):
        records(tmp_path)
