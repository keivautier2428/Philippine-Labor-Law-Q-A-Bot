# Preprocessing Report — Before / After Examples

Processed **33** documents from `data/raw/` into `data/processed/`.

Pipeline steps applied to each document:
1. Strip the manifest-style metadata header (source citation block).
2. Normalize whitespace, collapse HTML entity leftovers (`&nbsp;`), fix spacing before punctuation.
3. Tokenize into lowercase word/number tokens, preserving hyphenated legal terms (e.g. `thirteenth-month`) and removing a small stopword list.

## Corpus Statistics

- Total documents: 33
- Total tokens across corpus: 931
- Average tokens per document: 28.2

## Sample Before/After

### labor_code_art_12.txt

**Before (raw, 227 chars):**
```
Presidential Decree No. 442 - Labor Code of the Philippines BOOK ONE PRE-EMPLOYMENT   Article 12 ============================================================  Article 12. Statement of objectives. It is the policy of the State:
```

**After (cleaned, 67 chars):**
```
Article 12. Statement of objectives. It is the policy of the State:
```

**Tokens (6 total, first 20 shown):** `['article', '12', 'statement', 'objectives', 'policy', 'state']`

### labor_code_art_280.txt

**Before (raw, 857 chars):**
```
Presidential Decree No. 442 - Labor Code of the Philippines BOOK SIX POST EMPLOYMENT Title I TERMINATION OF EMPLOYMENT  Article 280 ============================================================  Article 280. Regular and casual employment. The provisions of written agreement to the contrary notwithstanding and regardless of the oral agreement of the parties, an employment shall be deemed to be regul...
```

**After (cleaned, 662 chars):**
```
Article 280. Regular and casual employment. The provisions of written agreement to the contrary notwithstanding and regardless of the oral agreement of the parties, an employment shall be deemed to be regular where the employee has been engaged to perform activities which are usually necessary or desirable in the usual business or trade of the employer, except where the employment has been fixed f...
```

**Tokens (57 total, first 20 shown):** `['article', '280', 'regular', 'casual', 'employment', 'provisions', 'written', 'agreement', 'contrary', 'notwithstanding', 'regardless', 'oral', 'agreement', 'parties', 'employment', 'shall', 'deemed', 'regular', 'where', 'employee']`

### labor_code_art_85.txt

**Before (raw, 460 chars):**
```
Presidential Decree No. 442 - Labor Code of the Philippines BOOK THREE CONDITIONS OF EMPLOYMENT Title I WORKING CONDITIONS AND REST PERIODS Chapter I HOURS OF WORK Article 85 ============================================================  Article 85. Meal periods. Subject to such regulations as the Secretary of Labor may prescribe, it shall be the duty of every employer to give his employees not les...
```

**After (cleaned, 222 chars):**
```
Article 85. Meal periods. Subject to such regulations as the Secretary of Labor may prescribe, it shall be the duty of every employer to give his employees not less than sixty (60) minutes time-off for their regular meals.
```

**Tokens (28 total, first 20 shown):** `['article', '85', 'meal', 'periods', 'subject', 'such', 'regulations', 'secretary', 'labor', 'may', 'prescribe', 'shall', 'duty', 'every', 'employer', 'give', 'his', 'employees', 'not', 'less']`

### labor_code_art_95.txt

**Before (raw, 326 chars):**
```
Presidential Decree No. 442 - Labor Code of the Philippines BOOK THREE CONDITIONS OF EMPLOYMENT Title I WORKING CONDITIONS AND REST PERIODS Chapter III HOLIDAYS, SERVICE INCENTIVE LEAVES AND SERVICE CHARGES Article 95 ============================================================  Article 95. Right to service incentive leave.
```

**After (cleaned, 45 chars):**
```
Article 95. Right to service incentive leave.
```

**Tokens (6 total, first 20 shown):** `['article', '95', 'right', 'service', 'incentive', 'leave']`

