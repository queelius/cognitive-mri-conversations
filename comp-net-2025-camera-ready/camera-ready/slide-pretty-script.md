# Cognitive MRI Presentation - Raw Script

## Slide 1: Title

"Good morning. This talk is about a simple question: what happens if you treat your AI conversation history as a dataset?"

"Most of us have hundreds—maybe thousands—of conversations with LLMs by now. They're usually just buried in a scroll. We wanted to know whether there's any interesting structure hiding in there."

"We're calling this a 'Cognitive MRI.' That's a metaphor, not a literal claim—the idea is to extract something resembling a knowledge map from conversation logs."

---

## Slide 2: Scale & Stakes

"ChatGPT alone has 1.7 billion users. That's a lot of conversational data."

"What's potentially interesting is that conversation logs capture something different from traditional datasets."

"Citation networks capture outputs. Social networks capture connections."

"Conversation logs might capture something closer to the *process* of thinking—the iteration, the back-and-forth."

"That's an empirical question. Today I'll show one case study to see if there's signal."

---

## Slide 3: Externalized Cognition

"We're framing this through Distributed Cognition—thinking happens not just in your head, but between you and your tools."

"When you use an LLM, you're thinking out loud. Offloading cognitive work to the machine."

"Ideas get constructed through dialogue, not just retrieved."

"What usually gets archived? The product. Linear, polished."

"The process underneath—networked, exploratory—usually invisible."

"Think of a bug fix. Twenty iterations of debugging—false leads, backtracking, finally the insight. The commit message? One line. Or a mathematician filling notebooks, redefining the problem three times before the proof clicks. We only see the final theorem."

"That's the cognitive dark matter. LLM logs might actually capture it."

"So here's what we did."

---

## Slide 4: From Log to MRI

"Start with the linear log. Chronological sequence. January: Python error. February: Banana bread. March: More debugging. April: Ethics."

"We embed each conversation—turn it into a point in high-dimensional space. 768 dimensions. Semantically similar conversations end up near each other geometrically."

"Then we link by similarity, not time."

"The result: a semantic topology. The two coding sessions—months apart—snap together."

"Banana bread stays isolated. Philosophy forms its own region."

"Distance in time doesn't equal distance in thought."

---

## Slide 5: Method - Weighting

"One practical issue: AI responses are verbose. Lots of boilerplate. If you embed everything equally, AI phrasing might dominate."

"So we separated user prompts from AI responses and weighted them differently."

"The intuition: user prompts carry more of the *intent*—what you actually wanted to know."

"AI response is context but shouldn't dominate."

"We embed user turns and AI turns separately, then take the mean of each. Then we combine those two mean vectors with the 2:1 weighting. One final vector per conversation."

"But how do we turn embeddings into a network?"

---

## Slide 6: From Embeddings to Edges

"Now we have embeddings—vectors in high-dimensional space. How do we turn that into a network?"

"Cosine similarity measures the angle between vectors. Same direction means similar content—cosine of 1. Orthogonal is 0. Since we normalize to unit length, this is just the dot product."

"If the similarity is at or above our threshold $\theta$, we connect them. The edge weight is the similarity score itself—stronger connections for more similar conversations."

"So we have two key parameters: $\alpha$ controls how we embed, $\theta$ controls how we connect. Both need validation."

---

## Slide 7: Parameter Selection

"We ran a 2D parameter sweep—63 configurations, varying both $\theta$ and $\alpha$ together. We optimized for modularity: how cleanly communities separate."

"The threshold dimension. There's a phase transition around 0.875. Below that, too many edges—you get a hairball. Above that, things fragment."

"$\theta$ = 0.9 gave reasonable structure. Not objectively 'correct'—a reasonable choice we're explicit about."

"The weight ratio dimension. Modularity peaked at 2:1, user-to-AI."

"This supported our intuition that user prompts carry more signal. The joint optimum: $\theta$ = 0.9, $\alpha$ = 2:1."

"The ablation gives us confidence the findings aren't artifacts of arbitrary choices. But this is still N=1."

---

## Slide 8: The Network

*[Pause 4 seconds—let them absorb it]*

"So here's the network. 449 conversations from about two years."

"1,615 edges. 15 communities. Modularity 0.750—reasonably high, suggesting non-random structure."

"The communities roughly correspond to topics I'd recognize."

"AI and machine learning here—dense cluster with lots of internal connections. Neural networks, probability, embeddings."

"Coding projects down here. More fragmented—different projects, different sub-clusters."

"Philosophy elsewhere. Writing. Math."

"The network isn't uniform. A quarter forms a dense core—broadly connected topics. The periphery is specialized. And the average path length—about 6 hops between any two conversations—gives you a sense of cognitive distance."

"I'm the one labeling these after the fact. The algorithm finds structure; interpretation is mine."

"That said—I did preliminary tests where an LLM labeled communities based on conversation content. Results were reasonable. In principle, this whole pipeline could be automated: embed, cluster, label. No human required."

"What's interesting is the algorithm found *something*. Whether these communities are meaningful beyond my recognition—harder question we can't fully answer with N=1."

---

## Slide 9: Heterogeneity

"One thing we noticed: different topic areas have different network structure."

"Theoretical topics—math, ML theory—have higher clustering, about 0.58. Practical coding is lower, 0.39."

"The interpretation: theoretical work involves returning to core concepts, refining definitions, lots of cross-referencing."

"Everything connects. Dense local structure."

"Coding projects are more linear. Solve one bug, move to next. Less backtracking."

"More tree-like. Not much connection between branches."

"Suggestive, not definitive. Need more data to know if this generalizes."

---

## Slide 10: Bridges

"We also looked at high-betweenness nodes—conversations connecting different clusters."

"Qualitatively, we noticed a few patterns in these bridging conversations."

"Some conversations *drift* between topics. Start in one area, organically evolve into another."

"'Evolutionary' bridges. Like geometric means drifting from pure math into neural network loss functions."

"Others are deliberate—explicitly connecting two fields. Ethics of AI, for example."

"'Integrative.' Consciously synthesizing."

"Occasionally a single conversation connects distant clusters. A 'pure' bridge—maybe a Linux question linking gaming to work."

"This is a taxonomy we're proposing based on what we observed."

---

## Slide 11: The Vision: Personal Knowledge Cartography

"Why might this matter?"

"Right now, conversation history is an infinite scroll. Finding something from months ago is painful."

"If this works more generally, you could navigate by topic rather than by date."

"'Show me everywhere I discussed entropy.'"

"Network lights up connections."

"This paper focused on network topology—structure, communities, bridges. But once you have this structure, it enables other things: semantic search, recommendations, gap detection. We haven't built those yet."

"That's speculative. But it's the direction."

---

## Slide 12: Cognitive MRI: A Proof of Concept

"To summarize: we took one user's conversation logs, built a semantic network, found what appears to be meaningful community structure."

"User weighting helps. Structural differences between topic types. Taxonomy of bridge conversations."

"But this is exploratory. N=1."

"One platform. Snapshot in time. No ground truth."

"Obvious next steps: more users, longitudinal analysis, validation."

"We'd welcome collaborators with larger datasets."

"Thanks. Happy to discuss."
