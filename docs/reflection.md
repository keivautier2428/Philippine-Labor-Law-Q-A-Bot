# Checkpoint 1 Reflection

Our raw dataset — Articles from Presidential Decree No. 442 (the Philippine Labor
Code) — was cleaner than typical scraped data in one sense (it's an authoritative,
well-formatted legal source), but it introduced its own kind of mess. Each article
is written as dense, single-paragraph legal prose with heavy cross-referencing
("the provisions of this Title," "as determined by the Secretary of Labor in
appropriate regulations"), inconsistent punctuation spacing, and leftover
`&nbsp;` HTML entities from the source page. Article boundaries also weren't
explicit in the raw HTML — we had to detect them from the "Article N." pattern
and separately track which Book/Title/Chapter each article belonged to just by
scanning heading-like lines, since the source page had no structured markup for
that hierarchy.

The biggest judgment call was scoping: the full Labor Code has 300+ articles
across seven Books, far more than a Q&A bot's knowledge base needs at this stage.
We selected 33 articles from the three Books our target users (workers, HR staff,
small-business owners) ask about most — Book One (recruitment/pre-employment),
Book Three (hours of work, wages, conditions of employment), and Book Six
(termination, retirement, separation pay) — rather than embedding the entire code
indiscriminately. We also kept the stopword list intentionally small; legal text
relies on function words ("shall," "any," "notwithstanding") that carry real
meaning, so an aggressive generic stopword list would have stripped signal a
retrieval system actually needs.
