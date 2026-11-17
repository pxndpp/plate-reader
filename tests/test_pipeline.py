"""Model-free checks of the pipeline's text logic. Run: python tests/test_pipeline.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import clean_province, clean_registration, group_lines  # noqa: E402


def box(x, y, w=40, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_group_lines():
    # EasyOCR returned the number before the letters (as seen on a real plate), plus a lower line
    res = [(box(120, 5), "2484", 0.5), (box(10, 6), "สก", 0.7), (box(10, 60), "กรุงเทพมหานคร", 0.8)]
    lines = group_lines(res)
    assert [t for t, _ in lines] == ["สก 2484", "กรุงเทพมหานคร"], lines
    assert group_lines([]) == []


def test_tilted_plate_keeps_province_separate():
    # real EasyOCR output from a slightly tilted plate: (x1, x2, y1, y2) per token
    def b(x1, x2, y1, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    res = [(b(121, 220, 26, 95), "8575", 1.0), (b(47, 143, 41, 116), "3กศ", 0.7),
           (b(76, 201, 82, 131), "กระเทพมุทายาร", 0.2)]
    lines = [t for t, _ in group_lines(res)]
    assert lines == ["3กศ 8575", "กระเทพมุทายาร"], lines


def test_clean():
    assert clean_registration("4กย5098`| ") == "4กย5098"
    assert clean_registration("ศน  463") == "ศน 463"
    assert clean_province("กรุงเทพ-มหานคร 12") == "กรุงเทพมหานคร"


if __name__ == "__main__":
    test_group_lines()
    test_tilted_plate_keeps_province_separate()
    test_clean()
    print("ok")
