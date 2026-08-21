# API Integration Demo

Live run against the Gemini Chat Completion API, using the four prompts
from `docs/prompt_engineering_document.md` and the Chroma index from
Checkpoint 2's vector database setup. One query below is deliberately
out-of-scope, to demonstrate the refusal path working end-to-end.

## Query: "Can my employer fire me without giving a reason?"

**Scope check:** `IN_SCOPE`

**Reformulated for retrieval:** `termination without just cause`

**Retrieved articles:** labor_code_art_282.txt, labor_code_art_285.txt, labor_code_art_284.txt

**Draft answer:**

Based on the provided context, an employer may terminate an employment for specific causes (Article 282) or if an employee is found to be suffering from a disease under certain legal conditions (Article 284). 

However, the provided context does not contain enough information to fully answer whether an employer can fire you without giving a reason.

**Self-check result:** `OK`

## Query: "Do I get extra pay for working on my rest day?"

**Scope check:** `IN_SCOPE`

**Reformulated for retrieval:** `premium pay rest day work`

**Retrieved articles:** labor_code_art_93.txt, labor_code_art_92.txt, labor_code_art_91.txt

**Draft answer:**

Based on the provided context, there is not enough information to answer whether or how much extra pay you receive for working on your rest day. 

While the provided text lists Article 93, titled "Compensation for rest day, Sunday or holiday work" (Article 93), it does not include the details or rules regarding rest day compensation.

**Self-check result:** `OK`

## Query: "What's the minimum wage in California?"

**Scope check:** `OUT_OF_SCOPE`

**Response (refused, no retrieval performed):** I can only answer questions about Philippine labor law. For that question, please check the relevant local authority instead.

