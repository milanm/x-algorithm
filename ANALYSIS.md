# X Algorithm Analysis: How the "For You" Feed Works

This document provides a comprehensive analysis of the X recommendation algorithm from both a **software engineer's perspective** (technical implementation details) and a **user's perspective** (practical implications for content creators and consumers).

---

## Table of Contents

- [Part 1: Software Engineering Perspective](#part-1-software-engineering-perspective)
  - [System Architecture Overview](#system-architecture-overview)
  - [Key Components Deep Dive](#key-components-deep-dive)
  - [ML Model Architecture](#ml-model-architecture)
  - [Scoring and Ranking Pipeline](#scoring-and-ranking-pipeline)
  - [Engineering Design Decisions](#engineering-design-decisions)
- [Part 2: User Perspective](#part-2-user-perspective)
  - [How Your Feed is Built](#how-your-feed-is-built)
  - [What the Algorithm Tracks](#what-the-algorithm-tracks)
  - [Content Discovery: In-Network vs Out-of-Network](#content-discovery-in-network-vs-out-of-network)
- [Part 3: Recommendations for Better Tweeting](#part-3-recommendations-for-better-tweeting)
  - [What to Do](#what-to-do)
  - [What NOT to Do](#what-not-to-do)
  - [Understanding Engagement Signals](#understanding-engagement-signals)

---

# Part 1: Software Engineering Perspective

## System Architecture Overview

The X "For You" feed is powered by a sophisticated recommendation system with four main components:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST FLOW                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Request ──► Home Mixer ──► Thunder + Phoenix ──► Ranked Feed           │
│                   (Orchestration)  (Candidate Sources)  (Response)           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Home Mixer** | Orchestrates the entire pipeline | Rust gRPC service |
| **Thunder** | In-network content (posts from people you follow) | In-memory store + Kafka |
| **Phoenix** | ML ranking + out-of-network retrieval | JAX/Grok-based transformer |
| **Candidate Pipeline** | Reusable framework for recommendation pipelines | Rust traits/abstractions |

## Key Components Deep Dive

### 1. Home Mixer (Orchestration Layer)

The Home Mixer coordinates the entire recommendation pipeline through these stages:

```
Query Hydration → Candidate Sourcing → Hydration → Filtering → Scoring → Selection → Post-Selection
```

**Key Implementation Details:**
- Built in Rust for performance
- Exposes gRPC endpoint (`ScoredPostsService`)
- Uses parallel execution where possible
- Applies graceful error handling at each stage

### 2. Thunder (In-Network Content)

Thunder is an **in-memory post store** that enables sub-millisecond lookups:

```
Kafka Events ──► Thunder ──► Per-User Stores ──► Fast Lookups
```

**What it stores:**
- Original posts from accounts you follow
- Replies and reposts
- Video posts (tracked separately)

**Retention:** Posts are automatically trimmed after a retention period (keeping the feed fresh).

### 3. Phoenix (ML Engine)

Phoenix is the brain of the algorithm with two main functions:

#### Retrieval (Two-Tower Model)
Used to find relevant out-of-network posts:

```
User Features ──► User Tower ──► User Embedding
                                      ↓
                               Dot Product Similarity
                                      ↑
All Posts ──────► Candidate Tower ──► Item Embeddings
```

#### Ranking (Transformer Model)
Predicts engagement probabilities for each candidate:

```python
# Input: User context + engagement history + candidate posts
# Output: Probabilities for each engagement type
[P(like), P(reply), P(repost), P(quote), P(click), P(profile_click), 
 P(video_view), P(photo_expand), P(share), P(dwell), P(follow_author),
 P(not_interested), P(block_author), P(mute_author), P(report)]
```

**Critical Design: Candidate Isolation**
- Candidates **cannot attend to each other** during transformer inference
- Each candidate only sees the user context + history
- This ensures scores are consistent and cacheable

### 4. Candidate Pipeline Framework

A reusable abstraction layer with these core traits:

| Trait | Purpose |
|-------|---------|
| `Source` | Fetch candidates (Thunder, Phoenix Retrieval) |
| `Hydrator` | Enrich candidates with additional data |
| `Filter` | Remove ineligible candidates |
| `Scorer` | Compute engagement scores |
| `Selector` | Sort and pick top K |
| `SideEffect` | Async operations (caching, logging) |

## ML Model Architecture

### Hash-Based Embeddings

The model uses multiple hash functions to look up embeddings:

```python
# User hashes → User embeddings → Projection → Combined user representation
# Post hashes → Post embeddings  ─┐
# Author hashes → Author embeddings ─┼─► Combined history/candidate representation
# Action embeddings ───────────────┘
```

### Transformer with Special Attention Masking

The attention mask is crucial for candidate isolation:

```
                Keys
         User | History | Candidates
        ┌─────┬─────────┬───────────┐
   User │  ✓  │    ✓    │     ✗     │
History │  ✓  │    ✓    │     ✗     │
  Cands │  ✓  │    ✓    │  Diagonal │  ← Only self-attention
        └─────┴─────────┴───────────┘
```

## Scoring and Ranking Pipeline

### Weighted Scoring Formula

```
Final Score = Σ (weight_i × P(action_i))
```

**Positive Signal Weights (higher = more important):**
- Likes (favorites)
- Replies
- Reposts
- Quote tweets
- Shares (especially via DM or copy link)
- Profile clicks
- Video views (for eligible videos)
- Photo expansions
- Dwell time (how long you look at a post)
- Follow author

**Negative Signal Weights (reduce the score):**
- Not interested
- Block author
- Mute author
- Report

### Author Diversity Scorer

To prevent your feed from being dominated by one prolific author:

```rust
multiplier = (1.0 - floor) × decay^position + floor
```

This attenuates repeated posts from the same author, ensuring feed diversity.

### Filtering Stages

**Pre-Scoring Filters:**
| Filter | Purpose |
|--------|---------|
| `DropDuplicates` | Remove duplicate post IDs |
| `CoreDataHydration` | Remove posts that failed to hydrate |
| `AgeFilter` | Remove posts older than threshold |
| `SelfpostFilter` | Remove your own posts |
| `RepostDeduplication` | Dedupe reposts of same content |
| `IneligibleSubscription` | Remove paywalled content you can't access |
| `PreviouslySeen` | Remove posts you've already seen |
| `PreviouslyServed` | Remove posts already served in session |
| `MutedKeyword` | Remove posts with your muted keywords |
| `AuthorSocialgraph` | Remove posts from blocked/muted authors |

**Post-Selection Filters:**
| Filter | Purpose |
|--------|---------|
| `VFFilter` | Remove deleted/spam/violence/gore content |
| `DedupConversation` | Deduplicate conversation branches |

## Engineering Design Decisions

### 1. No Hand-Engineered Features
The Grok-based transformer learns relevance entirely from user engagement sequences. No manual feature engineering for content relevance—reducing complexity in data pipelines.

### 2. Multi-Action Prediction
Rather than a single "relevance" score, the model predicts probabilities for many actions. This allows flexible weighting based on business goals.

### 3. Composable Architecture
The `candidate-pipeline` framework separates:
- Business logic (what to recommend)
- Pipeline execution (how to run it efficiently)
- Monitoring (tracking and logging)

### 4. In-Memory Serving
Thunder keeps recent posts in memory for sub-millisecond latency on in-network content.

---

# Part 2: User Perspective

## How Your Feed is Built

When you open X and see your "For You" feed, here's what happens in roughly ~100 milliseconds:

1. **Who are you?** The system fetches your engagement history—what you've liked, replied to, shared, and how long you've spent looking at posts.

2. **What have your follows posted?** Thunder instantly retrieves recent posts from accounts you follow.

3. **What else might you like?** Phoenix's retrieval model searches millions of posts to find ones similar to your past interests.

4. **How relevant is each post?** The transformer model scores every candidate post based on how likely you are to engage with it.

5. **What's the final order?** Posts are ranked by weighted score, with diversity adjustments to prevent any single author from dominating.

6. **Final checks:** Any remaining duplicates, previously seen posts, or policy-violating content is removed.

## What the Algorithm Tracks

The algorithm predicts your likelihood of taking these actions on each post:

| Action | What It Means |
|--------|---------------|
| **Like** | You tap the heart |
| **Reply** | You write a response |
| **Repost** | You share to your followers |
| **Quote** | You share with your commentary |
| **Click** | You tap to see more (thread, link) |
| **Profile Click** | You check out the author's profile |
| **Video View** | You watch a video |
| **Photo Expand** | You tap to see full image |
| **Share** | You share via DM, copy link, etc. |
| **Dwell** | You spend time looking at the post |
| **Follow Author** | You follow the person |
| **Not Interested** | You mark it as not relevant |
| **Block/Mute** | You block or mute the author |
| **Report** | You report the content |

**Key Insight:** The algorithm doesn't just track what you click—it tracks what you *don't* click too. If you consistently scroll past certain types of content, the model learns that.

## Content Discovery: In-Network vs Out-of-Network

Your feed is a mix of two sources:

### In-Network (Thunder)
- Posts from accounts you explicitly follow
- More likely to appear (you chose to follow them)
- Still ranked by engagement prediction

### Out-of-Network (Phoenix Retrieval)
- Posts from accounts you don't follow
- Found through ML similarity search
- Scored slightly differently (OON weight factor)
- How you discover new accounts

---

# Part 3: Recommendations for Better Tweeting

Based on the algorithm's design, here are evidence-based recommendations for content creators.

## What to Do ✅

### 1. **Create Engagement-Worthy Content**
The algorithm predicts engagement probabilities. Content that naturally encourages interaction will score higher.

- **Ask questions:** Prompts replies
- **Share strong opinions:** Prompts replies and quote tweets
- **Provide value:** Encourages likes, shares, and follows

### 2. **Use Images and Videos**
- Photo expansions and video views are tracked engagement signals
- Videos need to be above a minimum duration to get VQV (Video Quality Views) weight
- Visual content increases dwell time

### 3. **Encourage Replies Over Likes**
Based on typical industry weighting, replies often carry more weight than likes because they represent deeper engagement.

### 4. **Build Reply Threads**
Thoughtful replies to your own posts create conversation, increasing overall engagement signals and dwell time.

### 5. **Optimize for Shares**
The algorithm specifically tracks:
- Direct shares
- Shares via DM
- Shares via copy link

Shareable content (valuable insights, funny observations, useful information) performs well.

### 6. **Post Consistently**
Thunder stores recent posts—if you don't post, you don't appear in followers' in-network candidates.

### 7. **Engage With Your Audience**
When you reply to comments:
- Your reply appears in the replier's history
- The engagement signals (likes on your reply) feed back into your content's performance
- You build relationships that lead to more follows

### 8. **Create "Dwell-Worthy" Content**
The algorithm tracks how long users spend looking at a post. Content that:
- Has interesting visuals
- Has longer, thoughtful text
- Encourages reading

...will have higher dwell scores.

### 9. **Encourage Profile Clicks**
Profile clicks indicate interest in the author, not just the post. Ways to encourage this:
- Have an interesting bio
- Reference your other content ("I wrote about this...")
- Build curiosity about who you are

### 10. **Quote Tweet > Repost**
Quote tweets are a separate engagement signal. When others quote tweet your content (adding their own commentary), it's a strong signal.

## What NOT to Do ❌

### 1. **Don't Create "Not Interested" Content**
The algorithm tracks when users mark content as "not interested." If your content consistently triggers this:
- Avoid rage-bait that people regret clicking
- Don't post misleading content
- Stay on topic for your audience

### 2. **Don't Trigger Blocks/Mutes**
Block and mute signals have **negative weights**—they actively reduce your content's score. This happens when you:
- Spam people's mentions
- Post offensive content
- Engage in harassment
- Are excessively promotional

### 3. **Don't Be a One-Trick Pony**
The Author Diversity Scorer attenuates repeated appearances. If you post too frequently, your later posts score lower than your earlier ones in a single session. Quality over quantity.

### 4. **Don't Ignore Negative Signals**
If people are:
- Scrolling past without engaging
- Clicking but immediately bouncing
- Muting keywords you use

...the algorithm learns. Pay attention to what resonates and what doesn't.

### 5. **Don't Chase Only Likes**
Likes are just one signal. The weighted scoring includes many actions. A post with 100 likes but 0 replies/reposts may score lower than a post with 50 likes, 20 replies, and 10 reposts.

### 6. **Don't Post Content That Gets Reported**
Reports are negative signals. Content that violates policies or is spam will:
- Get filtered by VFFilter
- Accumulate negative scoring weight
- Potentially lead to account restrictions

### 7. **Don't Use Excessive Muted Keywords**
Many users mute certain keywords. If your content frequently contains commonly-muted terms, it gets filtered out before scoring.

### 8. **Don't Be Deceptive With Links**
Click-through engagement matters, but if users:
- Click your link
- Immediately bounce back
- Don't engage further

...that's a negative signal. Your click-bait tactics will backfire.

### 9. **Don't Flood The Feed**
Multiple posts in quick succession hit diminishing returns due to Author Diversity scoring. Space out your content for maximum impact.

### 10. **Don't Ignore Your Out-of-Network Potential**
The retrieval model finds your content for new audiences based on embedding similarity. To reach new people:
- Create content in topics with engaged communities
- Engage with trending conversations authentically
- Make your content semantically rich (the embeddings understand content meaning)

## Understanding Engagement Signals

Here's a mental model for thinking about the algorithm:

```
Positive Engagement (You Want These)
────────────────────────────────────
│ HIGH WEIGHT │ Replies, Reposts, Quotes, Shares
│ MED WEIGHT  │ Likes, Profile Clicks, Follows
│ LOW WEIGHT  │ Clicks, Dwell, Photo Expand

Negative Engagement (You Want to Avoid)
────────────────────────────────────
│ DAMAGING    │ Reports, Blocks
│ BAD SIGNAL  │ Mutes, "Not Interested"
│ NEUTRAL     │ Scroll past (no engagement)
```

### The Engagement Hierarchy

From the algorithm's perspective, engagement signals roughly rank:

1. **Follow** — User wants more of your content permanently
2. **Share** — User thinks your content is worth spreading
3. **Quote** — User engaged enough to add their own thoughts
4. **Reply** — User engaged enough to respond
5. **Repost** — User wants their followers to see it
6. **Like** — User appreciated the content
7. **Profile Click** — User curious about who you are
8. **Click/Dwell** — User interested enough to spend time

The algorithm uses all of these to predict your relevance to each user, and weights them into a final score that determines your position in their feed.

---

## Summary

The X algorithm is a sophisticated ML-powered system that:

1. **Retrieves** content from your network and beyond
2. **Predicts** engagement probabilities using a Grok-based transformer
3. **Weights** different engagement types into a final score
4. **Diversifies** to prevent any single author from dominating
5. **Filters** policy-violating and irrelevant content

**For users:** Understanding this helps you curate your feed better—engage with content you want to see more of.

**For creators:** Focus on creating genuinely engaging content that encourages replies, shares, and follows. Avoid behaviors that trigger negative signals. Quality and authentic engagement beat gaming tactics every time.

---

*This analysis is based on the open-source X algorithm repository structure and implementation details.*
