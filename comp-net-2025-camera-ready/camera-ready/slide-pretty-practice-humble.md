# Practice Script: Cognitive MRI Presentation (Humble Version)
## File: slide-pretty.tex | Total Time: 12 minutes

**Tone:** Intellectually honest. Exploratory. N=1 case study. Interesting preliminary findings, not revolutionary claims.

**Guiding Principle:** These are smart people. Don't oversell. Let the work speak for itself. Acknowledge limitations upfront, not as an afterthought.

---

## TIME BUDGET

| Section | Slides | Time |
|---------|--------|------|
| Setup | 1-4 | 3:00 |
| Methods | 5-6 | 3:00 |
| Results | 7-9 | 4:30 |
| Vision & Close | 10-11 | 1:30 |
| **Total** | **11** | **12:00** |

---

## Slide 1: Title (30 seconds)

**[VISUAL: Plain frame with gradient background, conference logo on right, GitHub link at bottom]**

> "Good morning. This talk is about a simple question: what happens if you treat your AI conversation history as a dataset?"

> "Most of us have hundreds, maybe thousands of conversations with LLMs by now. They're usually just... forgotten. Buried in a scroll. We were curious whether there's any interesting structure hiding in there."

> "We're calling this a 'Cognitive MRI'—that's a metaphor, obviously, not a literal claim. The idea is: can we extract something resembling a knowledge map from conversation logs?"

**TRANSITION:** "Let me give you some context on why we thought this was worth exploring."

---

## Slide 2: Scale & Stakes (45 seconds)

**[VISUAL: Left side has stylized globe with user icons scattered across it. Right side has comparison table.]**

**[POINT to globe with distributed user icons] (15s):**
> "ChatGPT alone has something like 1.7 billion users. That's a lot of conversational data being generated."

**[GESTURE across to table on RIGHT] (15s):**
> "What's potentially interesting here—and I want to be careful not to overstate this—is that conversation logs capture something different from traditional datasets."

**[POINT to first two rows: Citation Networks, Social Networks]:**
> "Citation networks capture outputs. Social networks capture connections."

**[POINT to highlighted third row with green checkmark]:**
> "But conversation logs might capture something closer to the *process* of thinking—the back-and-forth, the iteration."

> "I say 'might' because that's an empirical question. Today I'll show one case study—my own data—to see if there's anything there."

---

## Slide 3: The Iceberg (1 minute)

**[VISUAL: Left column has text/theory. Right column has iceberg diagram—paper icon above blue waterline, network with glowing nodes below it. Dashed arrow connects them labeled "Generated From".]**

**Opening (15s):**
> "There's a framing from cognitive science called Distributed Cognition—the idea that thinking happens not just in your head, but between you and your tools."

**[POINT to paper icon ABOVE the dashed waterline] (20s):**
> "Usually what gets preserved is the final output. The paper, the code, the finished product."

*[Note the paper has a folded corner and text lines—it's meant to look like a polished document]*

**[SWEEP hand DOWN past the waterline to the glowing network below] (20s):**
> "The messy process underneath—the false starts, the refinements, the dead ends—that usually disappears."

*[The network below has blue glowing nodes with hub-and-spoke structure—meant to evoke exploratory, non-linear thinking]*

**[POINT to the dashed arrow connecting them]:**
> "LLM logs are interesting because they might preserve some of that process. Whether that's actually useful is what we wanted to find out."

**TRANSITION:** "So here's what we did."

---

## Slide 4: From Log to Network (45 seconds)

**[VISUAL: Three-part layout. LEFT: Vertical timeline with colored dots (blue=Python, gray=Banana Bread, purple=Ethics). CENTER: Arrow labeled "Embed & Link". RIGHT: Network where similar topics cluster together.]**

**[POINT to LEFT timeline, trace finger down the dots] (15s):**
> "Start with the raw conversation log. Chronological. January: Python error. February: Banana bread recipe. March: More Python. April: Ethics discussion."

*[Note the blue dots (Python) are separated by time but will connect in the network]*

**[GESTURE to center arrow] (10s):**
> "We embed each conversation semantically and connect conversations by similarity rather than time."

**[POINT to RIGHT network—specifically where the two blue nodes are now adjacent] (15s):**
> "The Python conversations—months apart—end up as neighbors. The banana bread stays isolated over here."

*[Point to the gray isolated node, then to the purple Ethics node in its own area]*

> "This is a standard technique. The question is whether the resulting structure tells us anything interesting."

---

## Slide 5: Method - Weighting (1 minute)

**[VISUAL: Flowchart. Top: "Conversation" box. Two arrows descend—thick blue arrow (labeled "Signal ×2") to "User" box, thin orange arrow (labeled "Context ×1") to "AI" box. Both arrows converge to green "Final Embedding" box at bottom with formula.]**

**The Problem (15s):**
> "One practical issue: AI responses are verbose. Lots of boilerplate. If you embed everything equally, the AI's generic phrasing might dominate."

**[POINT to top "Conversation" box, then trace the two descending arrows] (20s):**
> "So we tried separating user prompts from AI responses and weighting them differently."

**[POINT specifically to the thick blue arrow and "×2" label] (15s):**
> "The intuition is that the user's prompts carry more of the *intent*—what you actually wanted to know."

**[POINT to thinner orange arrow and "×1" label]:**
> "The AI response is helpful context but maybe shouldn't dominate the embedding."

**[POINT to green result box at bottom showing the formula $\vec{e}_{conv} = 2 \cdot \vec{e}_{user} + 1 \cdot \vec{e}_{AI}$] (10s):**
> "We tested different ratios. More on that in a moment."

---

## Slide 6: Parameter Selection (2 minutes)

**[VISUAL: Left side has two stacked plots. Right side has bullet summary. TOP PLOT: Threshold vs modularity curve showing phase transition. BOTTOM PLOT: Weight ratio vs modularity showing peak at 2:1.]**

**Opening (30s):**
> "I want to be transparent about methodology here. There are two key parameters: the similarity threshold for connecting conversations, and the user-to-AI weight ratio. Both are choices that affect results."

**[POINT to TOP plot—trace the curve with finger] (45s):**
> "For the threshold: we swept across values and looked at modularity—how cleanly the network separates into communities."

**[POINT to the sharp rise around 0.875]:**
> "There's a transition around 0.875. Below that, everything connects into one blob—you can see modularity is low."

**[POINT to right side of curve where it plateaus/fragments]:**
> "Above that, things fragment. We chose $\theta$ = 0.9, which seemed to give reasonable community structure."

> "I won't claim this is the objectively 'correct' threshold. It's a reasonable choice that we're being explicit about."

**[MOVE hand DOWN to BOTTOM plot] (45s):**
> "For the weighting: modularity peaked at 2:1, user-to-AI."

**[POINT to the peak in the curve]:**
> "This supported our intuition that user prompts carry more signal, but it's one metric on one dataset."

**Conclusion (10s):**
> "The ablation study gives us some confidence that our findings aren't artifacts of arbitrary parameter choices. But this is still N=1."

---

## Slide 7: The Network (1.5 minutes)

**[VISUAL: This is the main visualization. Full-color network diagram (cluster-vis-topics-better.png) showing 449 nodes colored by community. Dense clusters visible. On-slide callouts: "AI Theory →" pointing to right side, "↓ Coding" pointing to bottom. Stats listed on right: 449 nodes, 1,615 edges, Q=0.750, 15 communities.]**

**[Let image appear - pause 3 seconds to let audience absorb it]**

> "So here's what the network looks like. 449 conversations from about two years of use."

**The Numbers (20s):**
**[POINT to stats on right side of slide]:**
> "1,615 edges. 15 communities detected by the Louvain algorithm. Modularity of 0.750—that's reasonably high, suggesting the communities aren't random."

**[GESTURE to the overall structure] (30s):**
> "The communities roughly correspond to topics I'd recognize."

**[USE the on-slide callout "AI Theory →" — point to the dense blue/purple cluster on the RIGHT side of the network]:**
> "AI and machine learning stuff over here—you can see it's a fairly dense cluster with lots of internal connections. Conversations about neural networks, probability theory, embeddings."

**[USE the on-slide callout "↓ Coding" — point to the pink/green clusters at the BOTTOM]:**
> "Coding projects down here. More fragmented—different projects in different sub-clusters."

**[SWEEP hand across other visible clusters]:**
> "Philosophy and ethics elsewhere. Writing projects. Math."

**[Step back from screen] (15s):**
> "I want to be clear: I'm the one labeling these clusters after the fact. The algorithm finds structure; the interpretation is mine."

> "That said—I did some preliminary tests where I had an LLM label the communities based on conversation content. The results were quite reasonable. So in principle, this whole pipeline could be fully automated: embed, cluster, label. No human in the loop required."

**Key Point (15s):**
> "What's interesting is that the algorithm found *something*. Whether these communities are meaningful beyond my own recognition—that's a harder question we can't fully answer with N=1."

---

## Slide 8: Observation 1 - Heterogeneity (1.5 minutes)

**[VISUAL: Two-column layout. LEFT: "Theoretical Domains" header with blue small-world diagram (dense mesh, central hub, peripheral nodes all interconnected) and block showing C ≈ 0.58. RIGHT: "Practical Domains" header with green tree diagram (hierarchical branching, no cross-connections) and block showing C ≈ 0.39.]**

**Opening (15s):**
> "One thing we noticed: different topic areas seem to have different network structure."

**[POINT to the clustering coefficients in the blocks below each diagram] (20s):**
> "Theoretical topics—math, ML theory—have higher clustering. About 0.58. Practical coding projects are lower, around 0.39."

**[POINT to LEFT diagram—the blue mesh with glowing nodes] (30s):**
> "The interpretation we'd offer: theoretical work involves returning to core concepts, refining definitions, lots of cross-referencing."

**[Trace the connections between peripheral nodes]:**
> "See how everything connects? That would create denser local structure."

**[MOVE hand to RIGHT diagram—the green tree] (30s):**
> "Coding projects are more linear. Solve one bug, move to the next. Less backtracking."

**[Trace the tree structure from root down to leaves]:**
> "More tree-like. Parent to child. Not much connection between branches."

**Caveat (15s):**
> "This is suggestive, not definitive. We'd need more users and more rigorous analysis to know if this pattern generalizes."

---

## Slide 9: Observation 2 - Bridges (1.5 minutes)

**[VISUAL: Left side shows bridge visualization (bridge-better.png) highlighting high-betweenness nodes. Right side lists three bridge types with color-coded headers: blue "Evolutionary", teal "Integrative", orange "Pure Bridges".]**

**Opening (15s):**
> "We also looked at high-betweenness nodes—conversations that connect different clusters."

**[POINT to visualization on LEFT—the nodes highlighted as bridges between communities] (15s):**
> "Qualitatively, we noticed a few different patterns in these bridging conversations."

**[POINT to blue "Evolutionary Bridges" text on right] (30s):**
> "Some conversations seem to *drift* between topics. They start in one area and organically evolve into another."

**[GESTURE back to visualization—trace a path that crosses communities]:**
> "We're calling these 'evolutionary' bridges. Like a conversation about geometric means that drifted from pure math into neural network loss functions."

**[POINT to teal "Integrative Bridges" text] (30s):**
> "Others are more deliberate—explicitly trying to connect two fields. Like discussing the ethics of AI."

> "We're calling these 'integrative.' You're consciously trying to synthesize."

**[POINT to orange "Pure Bridges" text] (20s):**
> "And occasionally there's a single conversation that connects otherwise distant clusters. A 'pure' bridge—maybe a random Linux question that happens to link gaming to work projects."

> "This is a taxonomy we're proposing, not a proven framework. It seemed like a useful way to categorize what we observed."

---

## Slide 10: Potential Applications (50 seconds)

**[VISUAL: Left side shows "The Scroll"—stacked gray bars fading upward representing buried conversations, labeled "Ephemeral & Buried". Center has transformation arrow. Right side shows "The Map"—a small network with labeled nodes (Biology, Coding, Entropy, AI, Ethics) connected by edges, with a dashed "Synthesis" path highlighted. Below is a tcolorbox with example query.]**

**Opening (10s):**
> "Why might this matter?"

**[POINT to LEFT scroll graphic] (20s):**
> "Right now, conversation history is basically an infinite scroll. Finding something from months ago is painful."

**[Trace finger up the fading bars]:**
> "That insight you had? Buried somewhere. Lost in the timeline."

**[GESTURE across the arrow to the RIGHT network map] (15s):**
> "If this kind of analysis works more generally, you could imagine a tool that lets you navigate your conversation history by topic rather than by date."

**[POINT to the example query box at bottom]:**
> "Query your own knowledge. 'Show me everywhere I discussed entropy.'"

**[Trace the connections in the small network—Entropy connecting to Biology, AI, Coding]:**
> "The network lights up the connections."

> "Now, this paper focused on analyzing the network structure itself—the topology, the communities, the bridges. But once you have this structure, there are other applications beyond visualization."

> "Semantic search across your history. Recommendation—'you asked about X, you might want to revisit Y.' Gap detection—finding topics you've explored separately but never connected. We haven't built those yet, but the network structure enables them."

> "That's speculative. But it's the direction we're interested in."

---

## Slide 11: Conclusion (1 minute)

**[VISUAL: Three-column layout. LEFT "Key Findings" (green block) with small network image and bullet points. MIDDLE "Limitations" (orange block) with single-user icon (N=1) and camera icon. RIGHT "Future Directions" (blue block) with growth diagram showing one node becoming many, plus magnifying glass icon.]**

**Summary (20s):**
> "To summarize: we took one user's conversation logs, built a semantic network, and found what appears to be meaningful community structure."

**[POINT to LEFT column—the green "Key Findings" block] (15s):**
> "User weighting seems to help. We observed structural differences between topic types. We proposed a taxonomy of bridge conversations."

**[POINT to MIDDLE column—the orange "Limitations" block with N=1 icon] (15s):**
> "But this is exploratory work. N=1."

**[POINT to camera icon]:**
> "One platform. Snapshot in time. No ground truth validation."

**[POINT to RIGHT column—the blue "Future" block with growth diagram] (10s):**
> "The obvious next steps: more users, longitudinal analysis, proper validation."

**[POINT to magnifying glass icon]:**
> "We'd welcome collaborators who have access to larger datasets."

**Closing (5s):**
> "Thanks. Happy to discuss."

**[GitHub link visible at bottom of slide for reference]**

---

## BACKUP SLIDES (Q&A Only)

**Navigation:** After Slide 11, press → to access backup slides.

### Backup 1: Technical Details
**When someone asks:** "What embedding model?" / "How did you detect communities?"
- nomic-embed-text, 768 dimensions, 500-token chunks with 50-token overlap
- Louvain algorithm at resolution 1.0, Q = 0.750
- Started with 1,908 conversations → 449 in giant component after $\theta$ = 0.9 filtering

### Backup 2: Core Formulas
**When someone asks:** "Can you show the math?"
- Weighted embedding: $\vec{e}_{conv} = \frac{\alpha \vec{e}_{user} + \vec{e}_{AI}}{\|\alpha \vec{e}_{user} + \vec{e}_{AI}\|}$
- Newman's modularity Q
- Betweenness centrality formula
- Clustering coefficient formula

### Backup 3: Privacy & Data Handling
**When someone asks:** "What about privacy?"
- This study: author's own data, official ChatGPT export
- Framework runs locally—no data leaves user's machine
- Future studies: IRB required, informed consent, anonymization, differential privacy

### Backup 4: Methodology Alternatives
**When someone asks:** "Why cosine similarity?" / "Why not k-NN?"
- Table comparing: Cosine vs Euclidean vs Jaccard
- Hard threshold vs soft clustering
- nomic-embed-text vs OpenAI vs Sentence-BERT
- 2:1 vs 1:1 vs user-only

---

## HUMBLE PHRASING CHEAT SHEET

**Instead of...** → **Say...**

- "reveals" → "suggests"
- "shows" → "we observed"
- "proves" → "is consistent with"
- "the first time" → "an opportunity"
- "exceptionally high" → "reasonably high"
- "unique to" → "we noticed in"
- "demonstrates" → "provides preliminary evidence"
- "the answer is" → "one interpretation is"
- "clearly" → [just remove it]
- "definitely" → [just remove it]

---

## KEY NUMBERS

| Number | What it means | Where to use |
|--------|---------------|--------------|
| 1.7 billion | ChatGPT users (context) | Slide 2 |
| $\theta$ = 0.9 | Similarity threshold | Slides 6, Backup 1 |
| 2:1 | User-to-AI weight ratio | Slides 5, 6 |
| Q = 0.750 | Modularity | Slides 6, 7 |
| 449 | Nodes (conversations) | Slide 7 |
| 1,615 | Edges | Slide 7 |
| 15 | Communities | Slide 7 |
| 0.58 | Clustering (theoretical) | Slide 8 |
| 0.39 | Clustering (practical) | Slide 8 |
| 1,908 → 449 | Before/after filtering | Backup 1 |

---

## VISUAL REFERENCE: Main Network (Slide 7)

```
                    ┌─────────────────────────────────┐
                    │                                 │
    Math/Stats      │      ┌──────────────┐          │
    cluster         │      │  AI Theory   │ ← "AI Theory →" callout
    (upper left)    │      │  (blue/purple│          │
                    │      │   dense)     │          │
                    │      └──────────────┘          │
                    │                                 │
    Philosophy/     │                                 │   Stats:
    Ethics          │                                 │   • 449 nodes
    (middle left)   │                                 │   • 1,615 edges
                    │                                 │   • Q = 0.750
                    │      ┌──────────────┐          │   • 15 communities
                    │      │   Coding     │          │
                    │      │  (pink/green │ ← "↓ Coding" callout
                    │      │  fragmented) │          │
                    │      └──────────────┘          │
                    └─────────────────────────────────┘
```

**What to point at:**
1. Dense AI Theory cluster (RIGHT side) — lots of internal edges
2. Fragmented Coding clusters (BOTTOM) — separate project silos
3. Overall modularity — clear boundaries between colors/communities

---

## TIMING CHECKPOINTS

| Time | You should be at... | If behind... |
|------|---------------------|--------------|
| 3:00 | Starting Slide 5 | Tighten Slide 3 |
| 5:00 | Middle of Slide 6 | You're fine |
| 8:00 | Finishing Slide 7 | On track |
| 10:00 | Finishing Slide 9 | Skip Slide 10 |
| 11:00 | Starting Slide 11 | Perfect |

---

## ANTICIPATED TOUGH QUESTIONS

**Q: "Isn't this just clustering text? What's novel?"**
> "Fair point. The embedding and clustering are standard. What's potentially interesting is the *data source*—conversation logs as a window into cognitive process. Whether that's actually valuable is an empirical question we're exploring."

**Q: "How do you know the communities are meaningful?"**
> "Honestly, we don't have strong validation. I recognize them as meaningful, but that's not rigorous. Proper validation would require user studies or retrieval benchmarks. That's future work."

**Q: "N=1 is pretty limited."**
> "Agreed. This is exploratory. We wanted to see if there was any signal before scaling up. The answer seems to be yes, but we'd need more data to make stronger claims."

**Q: "What about privacy concerns for multi-user studies?"**
> "Critical issue. Any multi-user study would need IRB approval, informed consent, anonymization. The framework we built runs locally—no data leaves your machine. But studying others' conversations raises real ethical questions."

**Q: "Could the structure just be an artifact of your parameter choices?"**
> "That's why we did the ablation study. The structure persists across a range of parameters, which gives us some confidence. But you're right that different choices would give different results."

**Q: "Why nomic-embed-text instead of OpenAI embeddings?"**
> "Open weights, reproducibility, and the 8k context window. We wanted something others could replicate without API costs. It's a reasonable choice, not necessarily the optimal one."

**Q: "What would falsify your hypothesis about structural heterogeneity?"**
> "If we saw the same clustering coefficient across all topic types in a larger sample, that would suggest the difference we observed was noise or specific to my usage patterns. That's testable with more data."

**Q: "How did you label the communities?"**
> "Manually, by reading representative conversations from each cluster. But I also ran preliminary tests with LLM-based labeling—feeding conversation samples to a model and asking it to propose a topic label. Results were reasonable. The whole pipeline could be automated."

**Q: "What applications beyond visualization?"**
> "This paper focused on network analysis—structure, communities, bridges. But the network enables other things: semantic search across your history, recommendation systems ('you discussed X, revisit Y'), gap detection (topics explored separately but never connected), maybe even identifying when your thinking has evolved on a topic over time."

---

## PRE-TALK MINDSET

You're presenting interesting preliminary work to smart colleagues. You're not selling anything. You're sharing what you found and inviting feedback.

If someone pokes a hole in your methodology, that's *useful*. Thank them.

The goal isn't to convince everyone this is groundbreaking. The goal is to share an interesting direction and see if others find it worth pursuing.

**The network visualization is your best asset.** Let people look at it. Don't rush past Slide 7.

---

## PRE-TALK CHECKLIST

- [ ] PDF loaded, tested arrow keys
- [ ] Timer visible (phone/watch)
- [ ] Water nearby
- [ ] Know your opening line cold
- [ ] Backup slides accessible (4 after main 11)
- [ ] Laser pointer working (if using)

**First line:** "Good morning. This talk is about a simple question: what happens if you treat your AI conversation history as a dataset?"

**Good luck.**
