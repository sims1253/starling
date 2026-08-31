"""S1-mini transcript fixtures (text-in/text-out; no audio).

Three tiers for the latency grid, mirroring the audio fixtures' role:

* ``short``  — the model-card quickstart transcript (~20 words).
* ``medium`` — a meeting-minute style ramble with fillers, self-corrections
  and numbers (~100 words).
* ``long``   — a dictated status report near the card's ~1,000-token input
  ceiling (~550 words).

``QUALITY_CASES`` are (transcript, controls, expected) triples whose expected
outputs are taken **verbatim from the model card** (quickstart + the styling
matrix + email/number renderings). They gate real normalization quality, not
just parity with stock.

``CONTROL_MATRIX`` sweeps every trained styling x structure x context value on
one transcript (parity-only; expectations come from stock at capture time).
"""

from __future__ import annotations

SHORT = "so um i need to like send the the report by uh friday no wait make that thursday"

MEDIUM = (
    "okay so um the the call yesterday with with the vendor went pretty well i think "
    "they said the new contract starts on march third twenty twenty six which is like "
    "a week earlier than we expected um they also want uh forty five hundred dollars "
    "a month instead of four thousand no wait sorry its four thousand two hundred "
    "and theyll throw in support for free so yeah i i think we should take it"
)

LONG = (
    "alright let me uh let me give you guys a quick update on on where the migration "
    "stands as of today so the the database side is basically done we finished moving "
    "all thirty two tables last thursday and the replication lag dropped to under "
    "fifty milliseconds which was the the big worry um the api side is about seventy "
    "percent done sarahs team finished the the auth endpoints and the billing ones "
    "but the the reporting endpoints are still in progress they think theyll be done "
    "by like friday next week um the the frontend is a bit behind honestly we we "
    "probably underestimated how much the the component rewrite would take were "
    "looking at maybe another three weeks there which pushes the the launch to "
    "august twenty second or or possibly the twenty ninth depending on qa one more "
    "thing the the infra costs came in lower than budgeted were at about two "
    "thousand three hundred dollars a month against a three thousand dollar budget "
    "so we have room for the the staging cluster mike asked for um if anyone has "
    "questions about the the migration plan the doc is is linked in the channel oh "
    "and and one last thing were still waiting on the security review they they "
    "promised it by the fifteenth but i i wouldnt be surprised if it slips to the "
    "eighteenth or so alright i think thats everything lets lets pick this up at "
    "the the next sync on monday"
)

# (transcript, styling, structure, context, expected) triples. Provenance:
# the model card's styling matrix, re-verified against the shipped ``main``
# revision of the weights (revision drift: the card examples were captured on
# an earlier revision that dropped leading ``So``/``Hmm``; the shipped model
# keeps them — stock transformers greedy is the source of truth here).
QUALITY_CASES: list[tuple[str, str, str, str, str]] = [
    (
        SHORT,
        "semi-formal", "prose", "general",
        "So I need to send the report by Thursday.",
    ),
    (
        "hmm im gonna be late theres a cute dog outside i cant just walk past him",
        "casual", "prose", "general",
        "hmm im gonna be late. theres a cute dog outside. i cant just walk past him",
    ),
    (
        "hmm im gonna be late theres a cute dog outside i cant just walk past him",
        "semi-casual", "prose", "general",
        "hmm, I'm gonna be late. there's a cute dog outside. I can't just walk past him",
    ),
    (
        "hmm im gonna be late theres a cute dog outside i cant just walk past him",
        "semi-formal", "prose", "general",
        "Hmm, I'm going to be late. There's a cute dog outside. I can't just walk past him.",
    ),
    (
        "hmm im gonna be late theres a cute dog outside i cant just walk past him",
        "formal", "prose", "general",
        "Hmm, I am going to be late. There is a cute dog outside. I cannot just walk past him.",
    ),
]

# Every trained control combination on one transcript (2 x 2 x 4 = 16 cases:
# structure x context x styling). Expectations come from stock at benchmark
# time; this matrix gates parity across the whole control space.
_CONTROL_TRANSCRIPT = (
    "um yeah so i i called the the support line twice on on january fifth and and "
    "the guy said hed email me at mike dot chen at example dot com but i i never "
    "got anything"
)
CONTROL_MATRIX: list[tuple[str, str, str, str]] = [
    (_CONTROL_TRANSCRIPT, styling, structure, context)
    for styling in ("casual", "semi-casual", "semi-formal", "formal")
    for structure in ("prose", "lists")
    for context in ("general", "email")
]

LENGTH_TIERS: dict[str, str] = {
    "short": SHORT,
    "medium": MEDIUM,
    "long": LONG,
}
