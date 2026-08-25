# API Integration Demo

Live run against the Gemini Chat Completion API, using the four prompts
from `docs/prompt_engineering_document.md` and the Chroma index from
Checkpoint 2's vector database setup. One query below is deliberately
out-of-scope, to demonstrate the refusal path working end-to-end.

Note: an earlier run of this demo hit two genuine `REVISE` verdicts from
the self-check prompt, because a data-pipeline bug was silently truncating
every retrieved article to its first sentence (e.g. Article 282 lost its
entire list of just causes). With too little grounding, the model
under-answered and self-check correctly flagged the gap. After fixing
that extraction bug (see `data/raw/` and `docs/preprocessing_report.md`),
both in-scope queries below pass self-check cleanly and cite specific
figures (30%/50% premium pay, the individual just-cause grounds) that the
truncated context never contained. Kept here as evidence the self-check
prompt does real work in both directions - it caught the problem, and
stopped flagging once the underlying data was fixed. Auto-regenerating on
a `REVISE` verdict, rather than just logging it, is still planned for
Checkpoint 3's RAG orchestration work.

## Query: "Can my employer fire me without giving a reason?"

**Scope check:** `IN_SCOPE`

**Reformulated for retrieval:** `termination without authorized just cause due process`

**Retrieved articles:** labor_code_art_285.txt, labor_code_art_282.txt, labor_code_art_284.txt

**Draft answer:**

Based on the provided context, an employer may only terminate an employment for specific just causes, which are:
* Serious misconduct or willful disobedience by the employee of the lawful orders of the employer or representative in connection with the work (Article 282).
* Gross and habitual neglect by the employee of duties (Article 282).
* Fraud or willful breach by the employee of the trust reposed in the employer or duly authorized representative (Article 282).
* Commission of a crime or offense by the employee against the person of the employer, any immediate member of the family, or duly authorized representatives (Article 282).
* Other causes analogous to the foregoing (Article 282).
* A disease where the employee's continued employment is prohibited by law or is prejudicial to their health and the health of co-employees (Article 284).

The provided context does not state whether an employer can fire an employee without giving a reason outside of these listed causes.

**Self-check result:** `REVISE: The draft answer misquotes the third bullet point regarding fraud or willful breach, as Article 282 states it is the breach of trust reposed in the *employer by the employee*, not "reposed in the employer".`

## Query: "Do I get extra pay for working on my rest day?"

**Scope check:** `IN_SCOPE`

**Reformulated for retrieval:** `rest day premium pay`

**Retrieved articles:** labor_code_art_93.txt, labor_code_art_91.txt, labor_code_art_95.txt

**Draft answer:**

Yes, if you are made or permitted to work on your scheduled rest day, you are entitled to an additional compensation of at least thirty percent (30%) of your regular wage (Article 93).

**Self-check result:** `OK`

## Query: "What's the minimum wage in California?"

**Scope check:** `OUT_OF_SCOPE`

**Response (refused, no retrieval performed):** I can only answer questions about Philippine labor law. For that question, please check the relevant local authority instead.

*Live query, typed during the presentation:*

## Query: "hi"

**Scope check:** `OUT_OF_SCOPE`

**Response (refused, no retrieval performed):** I can only answer questions about Philippine labor law. For that question, please check the relevant local authority instead.

