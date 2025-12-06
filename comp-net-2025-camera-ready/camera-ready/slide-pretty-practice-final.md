# Practice Script: Cognitive MRI Presentation (FINAL VERSION)
## File: slide-pretty.tex | Target Time: 11:00-11:30 (12-minute slot)

**Core Principle:** Let the work speak for itself. These are smart people—they'll see the limitations without you belaboring them. Focus on clarity, honesty, and forward momentum.

**Delivery Philosophy:** Measured pacing. Pause for complex visuals. Speed through transitions. The network visualization (Slide 8) is your centerpiece—give it room to breathe.

---

## REFINED TIME BUDGET

| Section | Slides | Time | Notes |
|---------|--------|------|-------|
| Setup | 1-4 | 2:57 | Includes embedding definition, concrete example |
| Methods | 5-7 | 3:15 | Weighting (1:00), edges (0:45), ablation (1:30) |
| Results | 8-10 | 4:32 | Network gets 1:42, observations ~1:25 each |
| Vision & Close | 11-12 | 1:35 | Crisp ending |
| **Target Total** | **12** | **12:19** | **Tight—practice will tell** |

**Contingency:** If running over at 10:30 mark, trim Slide 11 to 30 seconds (skip application details, just say "navigation by topic rather than timeline").

---

## Slide 1: Title (25 seconds)

**[VISUAL: Plain frame with gradient background, conference logo, GitHub link]**

**[Stand still, make eye contact, wait 2 seconds before speaking]**

> "Good morning. This talk is about a simple question: what happens if you treat your AI conversation history as a dataset?"

**[Pause 1 second]**

> "Most of us have hundreds—maybe thousands—of conversations with LLMs by now. They're usually just buried in a scroll. We wanted to know whether there's any interesting structure hiding in there."

**[Slight smile]**

> "We're calling this a 'Cognitive MRI.' That's a metaphor, not a literal claim—the idea is to extract something resembling a knowledge map from conversation logs."

**[Transition: move to next slide immediately, no verbal bridge needed]**

---

## Slide 2: Scale & Stakes (40 seconds)

**[VISUAL: Globe with user icons (left), comparison table (right)]**

**[POINT to globe] (10s):**
> "ChatGPT alone has 1.7 billion users. That's a lot of conversational data."

**[GESTURE to table on RIGHT] (12s):**
> "What's potentially interesting is that conversation logs capture something different from traditional datasets."

**[POINT to first two table rows briefly] (8s):**
> "Citation networks capture outputs. Social networks capture connections."

**[POINT to highlighted third row with green checkmark] (10s):**
> "Conversation logs might capture something closer to the *process* of thinking—the iteration, the back-and-forth."

**[Beat—let that land for 1 second]**

> "That's an empirical question. Today I'll show one case study to see if there's signal."

---

## Slide 3: The Big Picture - Externalized Cognition (55 seconds)

**[VISUAL: LEFT column has theory text with two bullet points ("Thinking Out Loud", "The Iterative Loop") and orange alert block ("The Cognitive Dark Matter"). RIGHT column has iceberg diagram: "Visibility Threshold" waterline, paper icon labeled "The Product (Linear, Polished)" above, glowing network labeled "The Process (Networked, Exploratory)" below, dashed arrow "Generated From" connecting them]**

**Opening (12s):**
**[GESTURE to left column header]:**
> "We're framing this through Distributed Cognition—thinking happens not just in your head, but between you and your tools."

**[POINT to "Thinking Out Loud" bullet] (10s):**
> "When you use an LLM, you're thinking out loud. Offloading cognitive work to the machine."

**[POINT to "The Iterative Loop" bullet] (8s):**
> "Ideas get constructed through dialogue, not just retrieved."

**[MOVE to RIGHT—point to paper icon ABOVE "Visibility Threshold"] (8s):**
> "What usually gets archived? The product. Linear, polished."

**[SWEEP hand DOWN past waterline to glowing network] (8s):**
> "The process underneath—networked, exploratory—usually invisible."

**[Concrete example—conversational tone] (12s):**
> "Think of a bug fix. Twenty iterations of debugging—false leads, backtracking, finally the insight. The commit message? One line. Or a mathematician filling notebooks, redefining the problem three times before the proof clicks. We only see the final theorem."

**[POINT to orange "Cognitive Dark Matter" alert block] (5s):**
> "That's the cognitive dark matter. LLM logs might actually capture it."

**[Transition:]**
> "So here's what we did."

---

## Slide 4: From Log to a "Cognitive MRI" (47 seconds)

**[VISUAL: Two-column comparison. LEFT: "1. The Linear Log (Chronological Sequence)" - vertical timeline with colored dots. MIDDLE: Arrow labeled "Embed & Link". RIGHT: "2. The Cognitive MRI (Semantic Topology)" - network with labeled clusters. BOTTOM: "The Insight" block]**

**[POINT to LEFT column title, trace down timeline] (12s):**
> "Start with the linear log. Chronological sequence. January: Python error. February: Banana bread. March: More debugging. April: Ethics."

**[Note: blue dots (Coding) are separated in time by gray (Cooking) and purple (Philosophy)]**

**[GESTURE to center arrow with "Embed & Link" label] (15s):**
> "We embed each conversation—turn it into a point in high-dimensional space. 768 dimensions. Semantically similar conversations end up near each other geometrically."

> "Then we link by similarity, not time."

**[POINT to RIGHT column - "The Cognitive MRI"] (15s):**
> "The result: a semantic topology. The two coding sessions—months apart—snap together."

**[POINT to "Linked!" annotation between blue nodes, then to isolated gray "Cooking" node]:**
> "Banana bread stays isolated. Philosophy forms its own region."

**[POINT to bottom "The Insight" block] (5s):**
> "Distance in time doesn't equal distance in thought."

---

## Slide 5: Method - Weighting (1:00)

**[VISUAL: Flowchart—"Conversation" at top splits into thick blue arrow (User ×2) and thin orange arrow (AI ×1), converging to green "Final Embedding" box with formula]**

**The Problem (12s):**
> "One practical issue: AI responses are verbose. Lots of boilerplate. If you embed everything equally, AI phrasing might dominate."

**[POINT to "Conversation" box, trace descending arrows] (15s):**
> "So we separated user prompts from AI responses and weighted them differently."

**[POINT to thick blue arrow and "×2" label] (15s):**
> "The intuition: user prompts carry more of the *intent*—what you actually wanted to know."

**[POINT to thin orange arrow] (8s):**
> "AI response is context but shouldn't dominate."

**[POINT to green formula box at bottom] (12s):**
> "We embed user turns and AI turns separately, then take the mean of each. Then we combine those two mean vectors with the 2:1 weighting. One final vector per conversation."

> "But how do we turn embeddings into a network?"

---

## Slide 6: From Embeddings to Edges (45 seconds)

**[VISUAL: LEFT side has vector diagram showing three vectors (e₁, e₂, e₃) from origin. e₁ and e₂ point similar direction (small angle φ) with "Edge!" annotation. e₃ points differently with "No edge (too different)" annotation. Cosine formula box at bottom. RIGHT side has two blocks: "Cosine Similarity" explaining angle measurement, "Edge Formation Rule" with threshold formula, and "Two Key Parameters" listing α and θ]**

**Opening (10s):**
> "Now we have embeddings—vectors in high-dimensional space. How do we turn that into a network?"

**[POINT to vector diagram—trace e₁ and e₂] (15s):**
> "Cosine similarity measures the angle between vectors. Same direction means similar content—cosine of 1. Orthogonal is 0. Since we normalize to unit length, this is just the dot product."

**[POINT to "Edge Formation Rule" block] (12s):**
> "If the similarity is at or above our threshold θ, we connect them. The edge weight is the similarity score itself—stronger connections for more similar conversations."

**[POINT to "Two Key Parameters" list] (8s):**
> "So we have two key parameters: α controls how we embed, θ controls how we connect. Both need validation."

---

## Slide 7: Parameter Selection (1:30)

**[VISUAL: Two stacked plots (left)—top shows threshold vs modularity phase transition, bottom shows weight ratio vs modularity peak at 2:1. Bullet summary (right)]**

**Opening (12s):**
> "We ran a 2D parameter sweep—63 configurations, varying both θ and α together. We optimized for modularity: how cleanly communities separate."

**[POINT to TOP plot—trace curve] (30s):**
> "The threshold dimension. There's a phase transition around 0.875. Below that, too many edges—you get a hairball. Above that, things fragment."

**[POINT to 0.9 on curve]:**
> "θ = 0.9 gave reasonable structure. Not objectively 'correct'—a reasonable choice we're explicit about."

**[MOVE DOWN to BOTTOM plot] (30s):**
> "The weight ratio dimension. Modularity peaked at 2:1, user-to-AI."

**[POINT to peak]:**
> "This supported our intuition that user prompts carry more signal. The joint optimum: θ = 0.9, α = 2:1."

**Conclusion (8s):**
> "The ablation gives us confidence the findings aren't artifacts of arbitrary choices. But this is still N=1."

---

## Slide 8: The Network (1:42) **← CENTERPIECE**

**[VISUAL: Full-color network (cluster-vis-topics-better.png) showing 449 nodes, colored by community. On-slide callouts: "AI Theory →" (right), "↓ Coding" (bottom). Stats on right: 449 nodes, 1,615 edges, Q=0.750, 15 communities]**

**[Let image appear—PAUSE 4 SECONDS. Let them absorb it. Don't speak yet.]**

> "So here's the network. 449 conversations from about two years."

**The Numbers (15s):**
**[POINT to stats on right]:**
> "1,615 edges. 15 communities. Modularity 0.750—reasonably high, suggesting non-random structure."

**The Structure (30s):**
**[GESTURE across overall structure]:**
> "The communities roughly correspond to topics I'd recognize."

**[USE on-slide callout "AI Theory →"—point to dense blue/purple cluster RIGHT]:**
> "AI and machine learning here—dense cluster with lots of internal connections. Neural networks, probability, embeddings."

**[USE callout "↓ Coding"—point to pink/green clusters BOTTOM]:**
> "Coding projects down here. More fragmented—different projects, different sub-clusters."

**[SWEEP across other visible clusters]:**
> "Philosophy elsewhere. Writing. Math."

**Core-Periphery (12s):**
> "The network isn't uniform. A quarter forms a dense core—broadly connected topics. The periphery is specialized. And the average path length—about 6 hops between any two conversations—gives you a sense of cognitive distance."

**The Interpretation (30s):**
**[Step back from screen]:**
> "I'm the one labeling these after the fact. The algorithm finds structure; interpretation is mine."

**[Pause 1 second]**

> "That said—I did preliminary tests where an LLM labeled communities based on conversation content. Results were reasonable. In principle, this whole pipeline could be automated: embed, cluster, label. No human required."

**Key Point (15s):**
> "What's interesting is the algorithm found *something*. Whether these communities are meaningful beyond my recognition—harder question we can't fully answer with N=1."

---

## Slide 9: Observation 1 - Heterogeneity (1:25)

**[VISUAL: Two columns—LEFT "Theoretical Domains" with dense blue mesh diagram (C ≈ 0.58), RIGHT "Practical Domains" with sparse green tree diagram (C ≈ 0.39)]**

**Opening (12s):**
> "One thing we noticed: different topic areas have different network structure."

**[POINT to clustering coefficients below diagrams] (15s):**
> "Theoretical topics—math, ML theory—have higher clustering, about 0.58. Practical coding is lower, 0.39."

**[POINT to LEFT blue mesh] (30s):**
> "The interpretation: theoretical work involves returning to core concepts, refining definitions, lots of cross-referencing."

**[Trace connections between peripheral nodes]:**
> "Everything connects. Dense local structure."

**[MOVE to RIGHT green tree] (25s):**
> "Coding projects are more linear. Solve one bug, move to next. Less backtracking."

**[Trace tree from root to leaves]:**
> "More tree-like. Not much connection between branches."

**Caveat (3s—quick):**
> "Suggestive, not definitive. Need more data to know if this generalizes."

---

## Slide 10: Observation 2 - Bridges (1:25)

**[VISUAL: Bridge visualization (left) with high-betweenness nodes highlighted. Right side lists three types: blue "Evolutionary", teal "Integrative", orange "Pure Bridges"]**

**Opening (10s):**
> "We also looked at high-betweenness nodes—conversations connecting different clusters."

**[POINT to visualization LEFT] (12s):**
> "Qualitatively, we noticed a few patterns in these bridging conversations."

**[POINT to blue "Evolutionary Bridges" text] (25s):**
> "Some conversations *drift* between topics. Start in one area, organically evolve into another."

**[GESTURE to visualization—trace path across communities]:**
> "'Evolutionary' bridges. Like geometric means drifting from pure math into neural network loss functions."

**[POINT to teal "Integrative Bridges" text] (25s):**
> "Others are deliberate—explicitly connecting two fields. Ethics of AI, for example."

> "'Integrative.' Consciously synthesizing."

**[POINT to orange "Pure Bridges" text] (13s):**
> "Occasionally a single conversation connects distant clusters. A 'pure' bridge—maybe a Linux question linking gaming to work."

**[Wrap—don't apologize]:**
> "This is a taxonomy we're proposing based on what we observed."

---

## Slide 11: Potential Applications (45 seconds)

**[VISUAL: "The Scroll" (left)—gray bars fading up labeled "Ephemeral & Buried" → arrow → "The Map" (right)—small network with labeled nodes and "Synthesis" path. Example query box at bottom]**

**Opening (8s):**
> "Why might this matter?"

**[POINT to scroll LEFT] (12s):**
> "Right now, conversation history is an infinite scroll. Finding something from months ago is painful."

**[Trace finger up fading bars briefly]**

**[GESTURE to network map RIGHT] (15s):**
> "If this works more generally, you could navigate by topic rather than by date."

**[POINT to query box]:**
> "'Show me everywhere I discussed entropy.'"

**[Trace network connections—Entropy to Biology, AI, Coding]:**
> "Network lights up connections."

**Broader View (10s):**
> "This paper focused on network topology—structure, communities, bridges. But once you have this structure, it enables other things: semantic search, recommendations, gap detection. We haven't built those yet."

**[Beat—don't oversell]:**
> "That's speculative. But it's the direction."

---

## Slide 12: Conclusion (50 seconds)

**[VISUAL: Three columns—LEFT "Key Findings" (green) with network image and bullets, MIDDLE "Limitations" (orange) with N=1 icon and camera, RIGHT "Future Directions" (blue) with growth diagram and magnifying glass]**

**Summary (15s):**
> "To summarize: we took one user's conversation logs, built a semantic network, found what appears to be meaningful community structure."

**[POINT to LEFT green "Key Findings"]:**
> "User weighting helps. Structural differences between topic types. Taxonomy of bridge conversations."

**[POINT to MIDDLE orange "Limitations" with N=1 icon] (12s):**
> "But this is exploratory. N=1."

**[POINT to camera icon]:**
> "One platform. Snapshot in time. No ground truth."

**[POINT to RIGHT blue "Future" with growth diagram] (10s):**
> "Obvious next steps: more users, longitudinal analysis, validation."

**[POINT to magnifying glass]:**
> "We'd welcome collaborators with larger datasets."

**Closing (3s):**
**[Make eye contact, smile slightly]:**
> "Thanks. Happy to discuss."

**[Hold position for 2 seconds before stepping back]**

**[GitHub link visible at bottom for those who want to follow up]**

---

## BACKUP SLIDES (Q&A Only) — **DO NOT ADVANCE UNLESS ASKED**

### Backup 1: Technical Details
**Trigger:** "What embedding model?" / "How did you detect communities?"
- nomic-embed-text, 768 dimensions
- 500-token chunks, 50-token overlap
- Louvain algorithm, resolution 1.0, Q = 0.750
- 1,908 conversations → 449 in giant component after θ = 0.9 filtering

### Backup 2: Core Formulas
**Trigger:** "Can you show the math?"
- Weighted embedding formula
- Newman's modularity Q
- Betweenness centrality
- Clustering coefficient

### Backup 3: Privacy & Data Handling
**Trigger:** "What about privacy?"
- This study: author's own data
- Framework runs locally—no data leaves machine
- Future: IRB, informed consent, anonymization required

### Backup 4: Methodology Alternatives
**Trigger:** "Why cosine?" / "Why not k-NN?"
- Comparison table: Cosine vs Euclidean vs Jaccard
- Hard threshold vs soft clustering
- nomic vs OpenAI vs Sentence-BERT
- 2:1 vs 1:1 vs user-only rationale

---

## TIMING CHECKPOINTS & RECOVERY STRATEGIES

| Clock | You should be at... | If behind... | If ahead... |
|-------|---------------------|--------------|-------------|
| 3:00 | Starting Slide 5 | Trim Slide 3 example | Add 5s pause on Slide 4 |
| 4:45 | Starting Slide 7 (Ablation) | Speed through Slide 6 | On track |
| 6:45 | Starting Slide 8 (Network) | Don't rush—this is key | Can linger on network |
| 8:15 | Finishing Slide 8 | Critical—don't cut network | Perfect—save buffer |
| 10:30 | Starting Slide 11 | **CONTINGENCY:** Trim Slide 11 | You have buffer |
| 11:30 | Starting Slide 12 | Cut to conclusion fast | Excellent—confident close |

**CRITICAL RECOVERY:** If at 10:30 you're still on Slide 10, cut Slide 11 to 20 seconds: "Imagine navigating by topic rather than timeline. That's the vision." Then jump to Slide 12.

---

## IF RUNNING AHEAD (Finishing early is fine—here's how to use extra time well)

**At 9:30 and starting Slide 11?** You have ~2 minutes of buffer. Here's what to do:

### Where to Expand (Gracefully)

1. **Slide 8 — The Network (best place to linger)**
   - Take an extra 10-15 seconds just *looking* at the network with them
   - Point to additional clusters: "There's also a philosophy cluster here... writing over here..."
   - More deliberately trace a path between communities
   - "Take a moment to find patterns you see"

2. **Slide 3 — The Iceberg**
   - The concrete example is already there; deliver it more slowly

3. **Slide 10 — Bridges**
   - Give an extra example for each bridge type
   - More slowly trace paths across the network visualization

4. **Pauses and Eye Contact**
   - Before each slide transition, make eye contact for 2-3 seconds instead of 1
   - After making a key point, let it land—don't rush to the next thought
   - Stillness reads as confidence

### What NOT to Do

- **Don't ramble** — Adding filler words or tangents sounds unprepared
- **Don't over-explain** — If you've made your point, stop
- **Don't apologize for being brief** — 10 minutes of clear content beats 12 minutes of padding
- **Don't add new claims** — Stay within what you can defend

### The Golden Rule

**Better to finish at 10:30 with a crisp close than to pad to 11:45 with meandering.**

Finishing 1-2 minutes early shows confidence. It leaves room for a longer Q&A, which is often where the best conversations happen. The moderator will appreciate not running behind.

**If you hit Slide 12 at 10:00:** Slow down slightly on the conclusion. Make deliberate eye contact with different sections of the audience. Let your final "Thanks. Happy to discuss." breathe.

---

## VISUAL REFERENCE: Network Structure (Slide 8)

```
                    ┌────────────────────────────────────────┐
                    │                                        │
    Math/Stats      │                                        │
    (upper left)    │       ┌─────────────┐                 │
                    │       │  AI Theory  │ ← "AI Theory →" │
    Philosophy/     │       │ (blue/purple│    callout      │
    Ethics          │       │   DENSE)    │                 │
    (middle left)   │       └─────────────┘                 │   Stats box:
                    │                                        │   • 449 nodes
                    │                                        │   • 1,615 edges
                    │                                        │   • Q = 0.750
                    │                                        │   • 15 communities
                    │       ┌─────────────┐                 │
                    │       │   Coding    │ ← "↓ Coding"    │
                    │       │ (pink/green │    callout      │
                    │       │  FRAGMENTED)│                 │
                    │       └─────────────┘                 │
                    └────────────────────────────────────────┘
```

**Pointing strategy:**
1. Stats first (15s) — establish numbers
2. Dense AI cluster RIGHT (15s) — lots of internal edges
3. Fragmented Coding BOTTOM (10s) — separate silos
4. Step back, talk interpretation (30s)
5. Don't rush—this is your best visual

---

## ANTICIPATED QUESTIONS & RESPONSES

**Q: "Isn't this just clustering text?"**
> "Fair point. Embedding and clustering are standard. The data source—conversation logs as cognitive process—is what's potentially interesting. Whether that's valuable is what we're exploring."

**Q: "How do you know communities are meaningful?"**
> "Honestly, we don't have strong validation. I recognize them, but that's not rigorous. Proper validation would need user studies or retrieval benchmarks. Future work."

**Q: "N=1 is pretty limited."**
> "Agreed. This is exploratory. We wanted to see if there was signal before scaling up. Answer seems to be yes, but we need more data for stronger claims."

**Q: "What about privacy for multi-user studies?"**
> "Critical issue. Any multi-user study needs IRB approval, informed consent, anonymization. Framework runs locally—no data leaves your machine. But studying others' conversations raises real ethical questions."

**Q: "Could structure be a parameter artifact?"**
> "That's why we did the ablation study. Structure persists across parameter ranges, which gives confidence. But different choices would give different results."

**Q: "Why nomic-embed-text?"**
> "Open weights, reproducibility, 8k context window. Wanted something others could replicate without API costs. Reasonable choice, not necessarily optimal."

**Q: "What would falsify the heterogeneity hypothesis?"**
> "If we saw same clustering coefficient across all topic types in a larger sample, that would suggest the difference was noise or specific to my patterns. Testable with more data."

**Q: "How did you label communities?"**
> "Manually, by reading representative conversations. But I also tested LLM-based labeling—reasonable results. Whole pipeline could be automated."

**Q: "Applications beyond visualization?"**
> "This paper focused on topology—structure, communities, bridges. But network enables semantic search, recommendation, gap detection, maybe tracking evolution of thinking over time. Haven't built those yet."

---

## DELIVERY NOTES

### Pacing Rhythm
- **Slides 1-4:** Moderate pace, building context
- **Slides 5-6:** Slightly faster through methods (they can read the paper for details)
- **Slide 7:** SLOW DOWN. This is the payoff. Let them look.
- **Slides 8-9:** Moderate—observations are interesting but don't drag
- **Slides 10-11:** Pick up pace—vision and wrap

### Gesture Economy
- **Don't over-gesture.** Point when directing attention to specific elements.
- **Use pauses instead of filler gestures.** Stillness = confidence.
- **On Slide 7:** Step back after pointing to let them see the whole network.

### Voice Modulation
- **Slide 1:** Conversational, inviting
- **Slides 2-4:** Building momentum
- **Slide 5-6:** Professional, methodical (this is the "we did our homework" section)
- **Slide 7:** Slightly more energy—this is your reveal
- **Slides 8-9:** Analytical but not dry
- **Slide 10:** Forward-looking, optimistic but measured
- **Slide 11:** Crisp, confident close

### What to Emphasize
- **"1.7 billion users"** — scale matters
- **"Distributed Cognition"** — theoretical anchor
- **"2:1 user weighting"** — design choice validated
- **"Q = 0.750"** — quantitative validation
- **"N=1"** — honest limitation
- **"15 communities"** — concrete finding

### What NOT to Do
- Don't apologize for limitations more than once (Slide 11 is enough)
- Don't rush the network visualization
- Don't read bullets—talk around them
- Don't say "um" or "so" as filler—pause instead
- Don't pre-answer questions you think they'll ask (save for Q&A)

---

## PRE-TALK MINDSET

You're not selling. You're sharing.

You found something interesting. You're being honest about its limitations. You're inviting others to explore this direction.

**The network is your evidence.** The modularity score backs it up. The ablation study shows it's not arbitrary.

If someone challenges your methodology, that's *collaboration*, not criticism. Thank them.

**Goal:** Walk out of this room with 2-3 people who want to talk more. That's success.

---

## PRE-TALK CHECKLIST

- [ ] **PDF loaded** — test arrow keys, ensure no black screen on first slide
- [ ] **Timer visible** — phone or watch, easy to glance at
- [ ] **Water nearby** — stay hydrated, use pauses for sips
- [ ] **Opening line memorized** — first 10 seconds should be automatic
- [ ] **Backup slides accessible** — know they're there but don't advance unless asked
- [ ] **Slide 7 clarity verified** — colors, callouts, stats all visible from back of room
- [ ] **Contingency plan clear** — know what to cut if running over at 10:00

---

## THE OPENING (MEMORIZE THIS)

**[Stand still. Make eye contact. Wait 2 seconds. Then begin.]**

> "Good morning. This talk is about a simple question: what happens if you treat your AI conversation history as a dataset?"

**[If you remember nothing else, remember this opening. It sets the tone for everything that follows.]**

---

## THE CLOSING (KNOW THIS COLD TOO)

**[After "we'd welcome collaborators with larger datasets"...]**

**[Pause 1 second. Make eye contact.]**

> "Thanks. Happy to discuss."

**[Smile slightly. Hold position 2 seconds. Step back.]**

**[Do NOT add "any questions?" or "I'll take questions now"—the moderator will handle that.]**

---

**You've got this. The work is solid. The network is beautiful. Just tell the story.**
