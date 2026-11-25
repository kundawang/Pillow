from __future__ import annotations

import signal
import subprocess
import sys
import time
import zlib
from io import BytesIO
from pathlib import Path

import pytest

from PIL.PdfParser import (
    IndirectObjectDef,
    IndirectReference,
    PdfArray,
    PdfBinary,
    PdfDict,
    PdfFormatError,
    PdfName,
    PdfParser,
    PdfStream,
    decode_text,
    encode_text,
    pdf_repr,
)


def test_text_encode_decode() -> None:
    assert encode_text("abc") == b"\xfe\xff\x00a\x00b\x00c"
    assert decode_text(b"\xfe\xff\x00a\x00b\x00c") == "abc"
    assert decode_text(b"abc") == "abc"
    assert decode_text(b"\x1b a \x1c") == "\u02d9 a \u02dd"


def test_indirect_refs() -> None:
    assert IndirectReference(1, 2) == IndirectReference(1, 2)
    assert IndirectReference(1, 2) != IndirectReference(1, 3)
    assert IndirectReference(1, 2) != IndirectObjectDef(1, 2)
    assert IndirectReference(1, 2) != (1, 2)
    assert IndirectObjectDef(1, 2) == IndirectObjectDef(1, 2)
    assert IndirectObjectDef(1, 2) != IndirectObjectDef(1, 3)
    assert IndirectObjectDef(1, 2) != IndirectReference(1, 2)
    assert IndirectObjectDef(1, 2) != (1, 2)


def test_parsing() -> None:
    assert PdfParser.interpret_name(b"Name#23Hash") == b"Name#Hash"
    assert PdfParser.interpret_name(b"Name#23Hash", as_text=True) == "Name#Hash"
    assert PdfParser.get_value(b"1 2 R ", 0) == (IndirectReference(1, 2), 5)
    assert PdfParser.get_value(b"true[", 0) == (True, 4)
    assert PdfParser.get_value(b"false%", 0) == (False, 5)
    assert PdfParser.get_value(b"null<", 0) == (None, 4)
    assert PdfParser.get_value(b"%cmt\n %cmt\n 123\n", 0) == (123, 15)
    assert PdfParser.get_value(b"<901FA3>", 0) == (b"\x90\x1f\xa3", 8)
    assert PdfParser.get_value(b"asd < 9 0 1 f A > qwe", 3) == (b"\x90\x1f\xa0", 17)
    assert PdfParser.get_value(b"(asd)", 0) == (b"asd", 5)
    assert PdfParser.get_value(b"(asd(qwe)zxc)zzz(aaa)", 0) == (b"asd(qwe)zxc", 13)
    assert PdfParser.get_value(b"(Two \\\nwords.)", 0) == (b"Two words.", 14)
    assert PdfParser.get_value(b"(Two\nlines.)", 0) == (b"Two\nlines.", 12)
    assert PdfParser.get_value(b"(Two\r\nlines.)", 0) == (b"Two\nlines.", 13)
    assert PdfParser.get_value(b"(Two\\nlines.)", 0) == (b"Two\nlines.", 13)
    assert PdfParser.get_value(b"(One\\(paren).", 0) == (b"One(paren", 12)
    assert PdfParser.get_value(b"(One\\)paren).", 0) == (b"One)paren", 12)
    assert PdfParser.get_value(b"(\\0053)", 0) == (b"\x053", 7)
    assert PdfParser.get_value(b"(\\053)", 0) == (b"\x2b", 6)
    assert PdfParser.get_value(b"(\\53)", 0) == (b"\x2b", 5)
    assert PdfParser.get_value(b"(\\53a)", 0) == (b"\x2ba", 6)
    assert PdfParser.get_value(b"(\\1111)", 0) == (b"\x491", 7)
    assert PdfParser.get_value(b" 123 (", 0) == (123, 4)
    assert PdfParser.get_value(b" 123.4 %", 0)[0] == pytest.approx(123.4)
    assert PdfParser.get_value(b" 123.4 %", 0)[1] == 6
    with pytest.raises(PdfFormatError):
        PdfParser.get_value(b"]", 0)
    with pytest.raises(PdfFormatError, match="key must be a name"):
        PdfParser.get_value(b"<<true[]>>", 0)
    d = PdfParser.get_value(b"<</Name (value) /N /V>>", 0)[0]
    assert isinstance(d, PdfDict)
    assert len(d) == 2
    assert d.Name == "value"
    assert d[b"Name"] == b"value"
    assert d.N == PdfName("V")
    a = PdfParser.get_value(b"[/Name (value) /N /V]", 0)[0]
    assert isinstance(a, list)
    assert len(a) == 4
    assert a[0] == PdfName("Name")
    s = PdfParser.get_value(
        b"<</Name (value) /Length 5>>\nstream\nabcde\nendstream<<...", 0
    )[0]
    assert isinstance(s, PdfStream)
    assert s.dictionary.Name == "value"
    assert s.decode() == b"abcde"
    for name in ["CreationDate", "ModDate"]:
        for date, value in {
            b"20180729214124": "20180729214124",
            b"D:20180729214124": "20180729214124",
            b"D:2018072921": "20180729210000",
            b"D:20180729214124Z": "20180729214124",
            b"D:20180729214124+08'00'": "20180729134124",
            b"D:20180729214124-05'00'": "20180730024124",
        }.items():
            b = b"<</" + name.encode() + b" (" + date + b")>>"
            d = PdfParser.get_value(b, 0)[0]
            assert time.strftime("%Y%m%d%H%M%S", getattr(d, name)) == value


def test_pdfstream_flatedecode() -> None:
    d = PdfDict({b"Filter": b"FlateDecode"})
    buf = zlib.compress(b"test")
    s = PdfStream(d, buf)
    assert s.decode() == b"test"

    with pytest.raises(ValueError, match="Decompressed data too large"):
        s.decode(3)


def test_pdf_repr() -> None:
    assert bytes(IndirectReference(1, 2)) == b"1 2 R"
    assert bytes(IndirectObjectDef(*IndirectReference(1, 2))) == b"1 2 obj"
    assert bytes(PdfName(b"Name#Hash")) == b"/Name#23Hash"
    assert bytes(PdfName("Name#Hash")) == b"/Name#23Hash"
    assert bytes(PdfDict({b"Name": IndirectReference(1, 2)})) == b"<<\n/Name 1 2 R\n>>"
    assert bytes(PdfDict({"Name": IndirectReference(1, 2)})) == b"<<\n/Name 1 2 R\n>>"
    assert pdf_repr(IndirectReference(1, 2)) == b"1 2 R"
    assert pdf_repr(IndirectObjectDef(*IndirectReference(1, 2))) == b"1 2 obj"
    assert pdf_repr(PdfName(b"Name#Hash")) == b"/Name#23Hash"
    assert pdf_repr(PdfName("Name#Hash")) == b"/Name#23Hash"
    assert (
        pdf_repr(PdfDict({b"Name": IndirectReference(1, 2)})) == b"<<\n/Name 1 2 R\n>>"
    )
    assert (
        pdf_repr(PdfDict({"Name": IndirectReference(1, 2)})) == b"<<\n/Name 1 2 R\n>>"
    )
    assert pdf_repr(123) == b"123"
    assert pdf_repr(True) == b"true"
    assert pdf_repr(False) == b"false"
    assert pdf_repr(None) == b"null"
    assert pdf_repr(b"a)/b\\(c") == rb"(a\)/b\\\(c)"
    assert pdf_repr([123, True, {"a": PdfName(b"b")}]) == b"[ 123 true <<\n/a /b\n>> ]"
    assert pdf_repr(PdfBinary(b"\x90\x1f\xa0")) == b"<901FA0>"


def test_linearize_page_tree() -> None:
    b = BytesIO()
    with PdfParser(f=b, mode="wb") as pdf:
        pdf.start_writing()
        pdf.write_header()

        pages_id = pdf.next_object_id(0)
        page_ids = [pdf.next_object_id(0) for _ in range(3)]
        for page_id in page_ids:
            pdf.write_page(page_id)
        pdf.write_obj(
            pages_id,
            Type=PdfName(b"Pages"),
            Count=2,
            Kids=[page_ids[0], page_ids[1]],
        )

        pdf.pages = [pages_id, page_ids[2]]
        pdf.write_catalog()
        pdf.write_xref_and_trailer()

    with PdfParser(f=b) as pdf:
        assert pdf.linearize_page_tree() == page_ids


def _build_pdf(
    objects: dict[int, bytes],
    pages_id: int = 1,
    omit_from_xref: tuple[int, ...] = (),
) -> bytes:
    """Build a minimal PDF in memory. ``objects`` maps object IDs (all
    generation 0) to their already-serialized object bodies. A Catalog
    pointing at the given Pages object ID is appended as the last object."""
    catalog_id = max(objects) + 1
    objects = dict(objects)
    objects[catalog_id] = bytes(
        PdfDict(
            {
                b"Type": PdfName(b"Catalog"),
                b"Pages": IndirectReference(pages_id, 0),
            }
        )
    )
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode("us-ascii")
        out += objects[object_id]
        out += b"\nendobj\n"
    xref_offset = len(out)
    max_id = max(objects)
    out += f"xref\n0 {max_id + 1}\n".encode("us-ascii")
    out += b"0000000000 65535 f \n"
    for object_id in range(1, max_id + 1):
        if object_id in omit_from_xref:
            out += b"0000000000 65535 f \n"
        else:
            out += f"{offsets[object_id]:010d} 00000 n \n".encode("us-ascii")
    catalog = PdfDict({b"Size": max_id + 1, b"Root": IndirectReference(catalog_id, 0)})
    out += (
        b"trailer\n"
        + bytes(catalog)
        + b"\nstartxref\n"
        + str(xref_offset).encode("us-ascii")
        + b"\n%%EOF\n"
    )
    return bytes(out)


def _pages_node(kids: list[IndirectReference], count: int | None = None) -> bytes:
    node = PdfDict(
        {
            b"Type": PdfName(b"Pages"),
            b"Count": len(kids) if count is None else count,
            b"Kids": PdfArray(kids),
        }
    )
    return bytes(node)


def _page_node(parent: IndirectReference) -> bytes:
    return bytes(
        PdfDict(
            {
                b"Type": PdfName(b"Page"),
                b"Parent": parent,
                b"MediaBox": PdfArray([0, 0, 10, 10]),
            }
        )
    )


class _AlarmTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: object) -> None:
    raise _AlarmTimeout


def test_page_tree_cycle() -> None:
    # Two-level cycle: the root Pages node lists an intermediate Pages node
    # whose Kids point back at the root.
    root_ref = IndirectReference(1, 0)
    inner_ref = IndirectReference(2, 0)
    objects = {
        1: _pages_node([inner_ref]),
        2: _pages_node([root_ref]),
    }
    data = _build_pdf(objects)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)
    try:
        with pytest.raises(
            PdfFormatError, match="reference cycle found at object 2 0 R"
        ):
            PdfParser(buf=data)
    finally:
        signal.alarm(0)


def test_page_tree_self_cycle() -> None:
    # A Pages node listing itself among its own Kids.
    root_ref = IndirectReference(1, 0)
    objects = {1: _pages_node([root_ref])}
    data = _build_pdf(objects)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)
    try:
        with pytest.raises(
            PdfFormatError, match="reference cycle found at object 1 0 R"
        ):
            PdfParser(buf=data)
    finally:
        signal.alarm(0)


def test_page_tree_duplicate_kid() -> None:
    # The same Page object is listed twice by the same parent.
    root_ref = IndirectReference(1, 0)
    page_ref = IndirectReference(2, 0)
    objects = {
        1: _pages_node([page_ref, page_ref]),
        2: _page_node(root_ref),
    }
    data = _build_pdf(objects)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)
    try:
        with pytest.raises(
            PdfFormatError, match="page tree node 2 0 R is referenced more than once"
        ):
            PdfParser(buf=data)
    finally:
        signal.alarm(0)


def test_page_tree_duplicate_kid_in_different_parents() -> None:
    # The same Pages subtree is referenced by two different parent nodes.
    left_ref = IndirectReference(2, 0)
    right_ref = IndirectReference(3, 0)
    shared_ref = IndirectReference(4, 0)
    page_ref = IndirectReference(5, 0)
    objects = {
        1: _pages_node([left_ref, right_ref]),
        2: _pages_node([shared_ref]),
        3: _pages_node([shared_ref]),
        4: _pages_node([page_ref]),
        5: _page_node(shared_ref),
    }
    data = _build_pdf(objects)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)
    try:
        with pytest.raises(
            PdfFormatError, match="page tree node 4 0 R is referenced more than once"
        ):
            PdfParser(buf=data)
    finally:
        signal.alarm(0)


def test_page_tree_missing_xref_entry() -> None:
    # A kid references an object that is marked free in the xref table: it
    # must not be treated as a valid page.
    root_ref = IndirectReference(1, 0)
    missing_ref = IndirectReference(2, 0)
    objects = {
        1: _pages_node([missing_ref]),
        2: _page_node(root_ref),
    }
    data = _build_pdf(objects, omit_from_xref=(2,))

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)
    try:
        with pytest.raises(PdfFormatError, match="references missing object 2 0 R"):
            PdfParser(buf=data)
    finally:
        signal.alarm(0)


def test_page_tree_nested_and_shared_resources() -> None:
    # A legal nested page tree: DFS order and page numbering are preserved,
    # and sharing an indirect Resources object between pages stays valid.
    root_ref = IndirectReference(1, 0)
    left_ref = IndirectReference(2, 0)
    right_ref = IndirectReference(3, 0)
    resources_ref = IndirectReference(4, 0)
    page_refs = [IndirectReference(i, 0) for i in range(5, 9)]
    objects = {
        1: _pages_node([left_ref, right_ref]),
        2: _pages_node([page_refs[0], page_refs[1]]),
        3: _pages_node([page_refs[2], page_refs[3]]),
        4: bytes(PdfDict({b"ProcSet": PdfArray([PdfName(b"PDF")])})),
    }
    for page_ref in page_refs:
        objects[page_ref.object_id] = bytes(
            PdfDict(
                {
                    b"Type": PdfName(b"Page"),
                    b"Parent": root_ref,
                    b"MediaBox": PdfArray([0, 0, 10, 10]),
                    b"Resources": resources_ref,
                }
            )
        )
    data = _build_pdf(objects)

    with PdfParser(buf=data) as pdf:
        assert pdf.pages == page_refs
        assert pdf.linearize_page_tree() == page_refs
        for page_ref in page_refs:
            assert pdf.read_indirect(page_ref)[b"Resources"] == resources_ref


def test_page_tree_deep_nesting_does_not_hit_recursion_limit() -> None:
    # A chain deeper than the default recursion limit must linearize because
    # the traversal is iterative rather than recursive.
    depth = sys.getrecursionlimit() + 100
    page_ref = IndirectReference(depth + 2, 0)
    objects: dict[int, bytes] = {}
    for object_id in range(1, depth + 1):
        kid_ref = IndirectReference(object_id + 1, 0)
        objects[object_id] = _pages_node([kid_ref])
    objects[depth + 1] = _pages_node([page_ref])
    objects[depth + 2] = _page_node(IndirectReference(depth + 1, 0))
    data = _build_pdf(objects)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(30)
    try:
        with PdfParser(buf=data) as pdf:
            assert pdf.pages == [page_ref]
    finally:
        signal.alarm(0)


def test_page_tree_cycle_in_subprocess() -> None:
    # Belt and braces: even with a very low recursion limit, a cyclic page
    # tree is rejected with PdfFormatError instead of hanging or raising
    # RecursionError, and the process terminates quickly on its own.
    code = (
        "import sys\n"
        "from PIL.PdfParser import IndirectReference, PdfFormatError, PdfParser\n"
        "from Tests.test_pdfparser import _build_pdf, _pages_node\n"
        "sys.setrecursionlimit(50)\n"
        "data = _build_pdf("
        "{1: _pages_node([IndirectReference(2, 0)]), "
        "2: _pages_node([IndirectReference(1, 0)])})\n"
        "try:\n"
        "    PdfParser(buf=data)\n"
        "except PdfFormatError as e:\n"
        "    assert 'cycle' in str(e); print('OK')\n"
        "else:\n"
        "    raise AssertionError('no error raised')\n"
    )
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert time.monotonic() - start < 30
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_duplicate_xref_entry() -> None:
    pdf = PdfParser("Tests/images/duplicate_xref_entry.pdf")
    assert pdf.xref_table.existing_entries[6][0] == 1197
    pdf.close()


def test_trailer_loop() -> None:
    with pytest.raises(PdfFormatError, match="trailer loop found"):
        PdfParser("Tests/images/trailer_loop.pdf")
