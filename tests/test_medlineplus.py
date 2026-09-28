"""
Tests for the MedlinePlus tool, run with `uv run pytest`.

These test the parsing only, against a saved sample reply, so they run
offline, cost no tokens, and never depend on a website being up. Same idea
as a JUnit test with a mocked HTTP client.
"""

from sympai.tools.medlineplus import parse_results

SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <list num="73" start="0" per="1">
    <document rank="0" url="https://medlineplus.gov/chestpain.html">
      <content name="title">&lt;span class="qt0"&gt;Chest&lt;/span&gt; &lt;span class="qt2"&gt;Pain&lt;/span&gt;</content>
      <content name="altTitle">&lt;span class="qt2"&gt;Pain&lt;/span&gt;, &lt;span class="qt0"&gt;Chest&lt;/span&gt;</content>
      <content name="FullSummary">&lt;p&gt;Having a &lt;span class="qt2"&gt;pain&lt;/span&gt; in your chest can be scary.&lt;/p&gt;&lt;p&gt;It does not always mean a heart attack &amp;amp; there are other causes.&lt;/p&gt;</content>
    </document>
  </list>
</nlmSearchResult>"""


def test_strips_html_and_keeps_the_link():
    [topic] = parse_results(SAMPLE)
    assert topic["title"] == "Chest Pain"
    assert topic["url"] == "https://medlineplus.gov/chestpain.html"
    assert topic["summary"] == (
        "Having a pain in your chest can be scary. "
        "It does not always mean a heart attack & there are other causes."
    )


def test_trims_long_summaries_on_a_word_boundary():
    [topic] = parse_results(SAMPLE, max_chars=20)
    assert topic["summary"] == "Having a pain in ..."


def test_no_results_gives_an_empty_list():
    empty = b'<?xml version="1.0"?><nlmSearchResult><list num="0"/></nlmSearchResult>'
    assert parse_results(empty) == []
