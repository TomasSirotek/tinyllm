2. Tokenizer 
	 2.1  **The concept.** Nets do arithmetic, so text must become numbers.
     A tokenizer is just a reversible two-way map: `'a' → 39`, `39 → 'a'`. 
     Two things to internalize:

     2.2  **Reversibility is non-negotiable.** If `decode(encode(x)) != x`, everything downstream
        is silently garbage.
	 2.3 **The ids are labels, not quantities.** `'b'` being 40 doesn't make it one more than `'a'`. 
        The model learns each id's meaning from scratch.
	 2.4 Real LLMs use **BPE**, where one token is a chunk like `" the"`.
        Char-level is 20 lines with zero magic — we swap in BPE at step 7
