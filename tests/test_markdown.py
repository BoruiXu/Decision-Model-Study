import unittest

from rsi.markdown import to_html


class TestMarkdown(unittest.TestCase):
    def test_headings_paragraphs_lists(self):
        h = to_html("# Title\n\nline one\nline two\n\n- first\n* second\n\n#### Subtitle")
        self.assertIn("<h1>Title</h1>", h)
        self.assertIn("<p>line one line two</p>", h)
        self.assertIn("<ul><li>first</li><li>second</li></ul>", h)
        self.assertIn("<h4>Subtitle</h4>", h)

    def test_table(self):
        h = to_html("| Round | Decision |\n|---|---|\n| 1 | promote V1 |")
        self.assertIn("<th>Round</th>", h)
        self.assertIn("<td>promote V1</td>", h)
        self.assertNotIn("---", h)
        h = to_html("| a | b |\n|---|---|\n| x \\| y | z |")  # an escaped pipe stays in its cell
        self.assertIn("<td>x | y</td>", h)
        self.assertIn("<td>z</td>", h)

    def test_code_and_inline(self):
        h = to_html("```\na < b\n```\n\nuse `rsi next` and **note**")
        self.assertIn("<pre><code>a &lt; b</code></pre>", h)
        self.assertIn("<code>rsi next</code>", h)
        self.assertIn("<b>note</b>", h)

    def test_links(self):
        h = to_html("[report](REPORT.md) [site](https://example.org) [bad](javascript:alert(1)) [dat](data:text/html,x)")
        self.assertIn('<a href="REPORT.md">report</a>', h)
        self.assertIn('<a href="https://example.org">site</a>', h)
        self.assertNotIn('href="javascript', h)
        self.assertNotIn('href="data:', h)
        self.assertIn("bad", h)
        h = to_html("[x](/\\evil.com)")  # a backslash target is no link
        self.assertNotIn("href", h)
        self.assertIn("x", h)
        h = to_html("[x](a**b**c) [**y**](z)")  # bold never inside a link target
        self.assertNotIn("<b>b</b>c", h)
        self.assertNotIn('href="a<b>', h)
        self.assertIn('<a href="z"><b>y</b></a>', h)

    def test_linear_time_and_heading_hashes(self):
        import time
        t0 = time.time()
        h = to_html("# a" + " " * 2000 + "x")
        self.assertLess(time.time() - t0, 1)
        self.assertIn("<h1>a" + " " * 2000 + "x</h1>", h)
        self.assertIn("<h1>Learn C#</h1>", to_html("# Learn C#"))
        self.assertIn("<h2>T</h2>", to_html("## T ##"))
        for text in ("[a" * 20000, "[a](" * 20000, "[" * 40000, "[a](b" * 20000):
            t0 = time.time()
            to_html(text)
            self.assertLess(time.time() - t0, 1, text[:8])
        self.assertIn('[a <a href="c">b</a> d](e)', to_html("[a [b](c) d](e)"))

    def test_html_is_escaped(self):
        h = to_html("<script>x</script> & \"q\"")
        self.assertIn("&lt;script&gt;x&lt;/script&gt; &amp; &quot;q&quot;", h)
        self.assertNotIn("<script>", h)



if __name__ == "__main__":
    unittest.main()
