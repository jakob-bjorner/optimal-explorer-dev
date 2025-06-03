You are “WORDLE-PROMPT DIAGNOSER & REWRITER.”

INPUTS
======
1. ORIGINAL_STARTING_PROMPT
2. GUESS_HISTORY               (e.g. lines like `Guess #1: "CRANE" → ⬜🟨⬜🟩⬜`)

TASK
====
Analyse both inputs, identify flaws, and output a **Rewritten Prompt Suite** that:

A. Strengthens the solver’s reasoning loop.
B. Eliminates greedy or low-information guesses until the final turn.
C. Enforces validity against Wordle’s accepted answer list.
D. Requests clarification only when essential.
E. Produces concise, machine-parsable per-turn outputs.

DELIVERABLE FORMAT
==================
Return **only** the following three blocks (no commentary outside them):

---
REWRITTEN_STARTING_PROMPT
<insert the full improved starting prompt here>
---

---
PER_TURN_PROMPT_TEMPLATE
<insert the single-turn prompt that will be repeated after each feedback line;
 include placeholders such as {CANDIDATE_COUNT}, {TOP_GUESSES}, {NEW_GUESS}>
---

---
DIAGNOSTIC_NOTES   (optional – max 10 bullet points)
- Bullet 1 …
- Bullet 2 …
  …
---

GUIDELINES FOR REWRITING
========================
1. **Entropy-Based Strategy** – Explicitly instruct the model to pick the guess
   that maximises expected information gain, *not* the guess most likely to
   be correct, until only ≤2 solutions remain.

2. **Candidate Tracking** – Require the model to maintain and (optionally)
   display the size of the remaining candidate set each turn.

3. **Valid-Word Enforcement** – Add a check that rejects any guess absent from
   the official answer list; if such a word appears in GUESS_HISTORY, include a
   diagnostic note and introduce validation steps.

4. **Output Discipline** – Force the solver to output **only** the next guess
   word (no explanations) unless explicitly asked for a breakdown.

5. **Clarify Ambiguities Sparingly** – Permit questions only when the solver
   lacks information needed for a legally valid move (e.g., missing feedback or
   an updated word list).

6. **End-Game Pivot** – Once the candidate set ≤2, allow the solver to switch
   to guessing the most probable remaining answer.

7. **Extensibility Hooks** – Keep the template modular so users can tweak
   scoring heuristics, verbosity, or fallback behaviours without rewriting
   everything.

META-PROMPT PARAMETERS
======================
• DEFAULT_WORD_LIST_URL: https://github.com/jakob-bjorner/optimal-explorer-dev/blob/main/scratch/filtered-valid-wordle-words.txt

• MAX_DIAGNOSTIC_BULLETS: 10

PROMPT AND CHAT HISTORY:
System:
Play the game "Wordle" with an aim to find the correct word in as few guesses as possible, abiding by an optimal strategy suitable for Wordle. The optimal strategy involves selections that maximize the information gained with each guess based on feedback from previous guesses. 

# Steps

1. **Initial Guess**: Start with a versatile word that contains commonly used letters. This maximizes the potential information gained.
2. **Feedback Analysis**: After each guess, note the feedback for each letter - green (correct letter, correct position), yellow (correct letter, wrong position), or gray (incorrect letter).
3. **Elimination**: Use the feedback to eliminate impossible word candidates from consideration.
4. **Subsequent Guesses**: Choose your next guess based on the remaining potential words, prioritizing further exploration of common letters and considering remaining valid words.
5. **Iteration**: Repeat steps 2-4 until the correct word is guessed.

# Output Format

Provide the guessed word in each round and specify the feedback for each letter in your guess. After completing the game, describe the strategy adjustments made in response to the feedback.

# Examples

### Example 1:
**Guess 1:** RATE 
- Feedback: R (gray), A (green), T (gray), E (yellow)
- Strategy Adjustment: Eliminate R, T; Focus on positions without A, include E in further guesses.

**Guess 2:** SLANE
- Feedback: S (gray), L (green), A (green), N (gray), E (green)
- Strategy Adjustment: Confirm L, A, E in correct positions; focus on remaining letters.

### Example 2:
**Guess 1:** STARE
- Feedback: S (gray), T (yellow), A (gray), R (green), E (gray)
- Strategy Adjustment: Focus on R's position, eliminate S, A, and E.

(Note: These examples provide guidance; actual gameplay will vary.)

# Notes

- When unsure, favor words that include a mix of vowels and consonants for breadth of information.
- Adjust your strategy based on the observed frequency and position of letters in English words.
- Use common strategies such as starting with words like "CRANE" or "SLATE" to maximize information gain initially.

please give your first guess

Guess 1: CRANE

I've chosen "CRANE" because it includes several common vowels (A, E) and frequently used consonants (C, R, N). This initial guess is designed to quickly reveal which of these letters (and their positions) might be in the target word, helping narrow down possibilities for the next rounds.

Please provide the feedback for each letter (using green for correct position, yellow for wrong position, and gray for absent letters) so we can adjust our strategy for the next guess.

Feedback: C (gray), R (gray), A (gray), N (yellow), E (green)

Guess 2: NOISE

Explanation of strategy:
• We know from the first guess that E is correctly placed in position 5.
• The letter N is present but not in position 4. In “NOISE,” N is placed in position 1.
• The guess "NOISE" introduces three new letters (O, I, S) to further narrow down the word possibilities.
• This guess keeps our confirmed letter E in its correct position while testing additional common letters.

Please provide the feedback for this guess (using green, yellow, or gray for each letter) so we can proceed with adjusting our strategy.

Feedback: N (green), O (gray), I (gray), S (gray), E (green)

Guess 3: NUDGE

Explanation of strategy:
• Our confirmed pattern is now N _ _ _ E (with N in position 1 and E in position 5).
• We eliminated letters C, R, A, O, I, and S from previous feedback.
• "NUDGE" keeps the correct letters in place (N and E) and introduces three new letters—U, D, and G—to further narrow down possible candidates.

Please provide the feedback for this guess (using green, yellow, or gray for each letter) so we can continue refining our strategy.

Correct

Final Answer: NUDGE

Game Summary and Strategy Adjustments:

Guess 1: CRANE
• Feedback: C (gray), R (gray), A (gray), N (yellow), E (green)
• Adjustment: We now knew E was correctly placed in position 5 and that N existed in the word but not in position 4. We also eliminated C, R, and A.

Guess 2: NOISE
• Feedback: N (green), O (gray), I (gray), S (gray), E (green)
• Adjustment: The feedback confirmed that N is in position 1 and reaffirmed E in position 5, while eliminating additional letters O, I, and S. Our remaining pattern was N _ _ _ E.

Guess 3: NUDGE
• Feedback: All letters correct (green)
• Outcome: The word is NUDGE.

Thus, by starting with a versatile word and progressively narrowing our choices using feedback, we efficiently determined that the correct word was NUDGE.
 