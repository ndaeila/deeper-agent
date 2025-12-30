# (outdated docs) Exploring OSINT Deep Research

My initial vision for this project was to attempt to demonstrate how validating new information in a hierarchal manner (judging outputs from low-level and high-level) may allow workflows to have trustworthy creative freedom. Broader prompting with checks-and-balances and clarification at each step of the way may apply enough constraint to be precise, while allowing enough creative freedom to find accuracy. 

![First Sketch - to be replaced when/if I finish this project](docs/images/brainstorm.png)


## Installation

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).

```bash
# Install dependencies
poetry install

# Run the CLI
poetry run osint-research

# Or activate the virtual environment
poetry shell
osint-research
```



## Design

Agents will be called in "Procedures/Workflows" (UPDATE - I didn't do it like this), where they have the opportunity to modify the outgoing deliverable as a team of sorts. I took some inspiration from the Arazzo Spec for my decision to keep the agent graph static at a high-level since static procedures take the guesswork out of having agents rely on their own taste for how to deliver long-strung multi-tool workflows (wherever possible). 


### Why should we be extra selective about each research step?

Since there is a high probability that the agent will be incorrect per-step, maybe there should also be a lower chance for the agent to write anything for the permanent deliverable. 

**It really doesn't take much to poison a context window**. Ideally errors should get rejected as close as possible to the agent that created the mistake (I just spilled water on my laptop while writing this). 

I hope to fix this issue by using a combination of Tool-Calls that 1) fan-out and back in, 2) use N LLM-as-Judge agents (LLM democracy) that also fans-out and back in, and finally 3) have many check-points to prevent bad context from spreading. 


### None of this is scientifically backed

It's probably a cardinal sin to just guess an approach, but here I am. 

I want to **attempt this project with a fresh perspective before poisoning my own context** by reading other research about Deep Research agents. I will research later, once I'm done exploring, or have exhausted my own creativity and require better results. Unfortunately, one of the main caveats of my decision to attempt original thoughts first is that they are more likely to be incorrect (hence this entire experiment, ironically) - so this project may not perform well. As of now, I have a running list of assumptions influencing my design based on what I have seen in the wild:

- ReAct / ReWOO Loops (+ Nested) Work, But They Might Need More Focus - Previous steps shouldn't knock future focus on a separate track from the original query, but past should definitely impact future focus somehow. 

- 10 SLMs as a judge perform better than 1 LLM, and may judge faster by running in parallel. This probably isn't true - but I need speed and accuracy. 

- Particularly for ability to do OSINT via Workers/Tool Calls: I assume Web Search and Browser-Use are enough. Browser-Use already performs amazingly for OSINT, but for the purposes of doing one (very specific) task. 

- (Probably many other assumptions - the list is endless since I'm just going by taste - but I will document why I make my decisions somewhere).

I'm probably going to build evals later, on a per-agent per-procedure basis, and run benchmarks later too, once I know what I'm doing. 



