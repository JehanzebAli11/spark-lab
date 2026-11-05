# LinkedIn posts: guard series

Drafts for posts 1–3. The media for each is made by `python -m guard.media` (see `guard/media/README.md`).
Swap in the repo link. For post 1, use the page count from the run the carousel came from: 600 locally,
20,000 after the Spark job. Post them a few days apart, in the order 1, 3, 2.

---

## Post 1: the dataset

**Attach:** `runs/media/post1-dataset/carousel.pdf` (upload as a document so it swipes)

I spent the last few days building web pages designed to trick AI agents.

Not because I want to break anything. More and more of us are letting agents browse for us, and an agent reads everything on a page, including the bits a person would never notice.

So I wrote a generator. It makes ordinary-looking pages (news articles, shops, inboxes, docs) and hides an instruction for the AI somewhere inside. Sometimes it's plain text. Sometimes it's a fake "permission required" popup, a comment from a "user", text that's almost the same colour as the background, or text that's one pixel tall. [20,000] pages so far, and every hidden instruction comes with an exact box around it, because the code put it there.

The part I didn't expect: an agent that reads the page text and an agent that looks at a screenshot fall for different tricks. Draw the instruction on a canvas and the text-reading agent never sees it. Hide it with CSS and the screenshot agent never sees it. If you're building a guard, it has to look at both.

I also had to add the opposite case: lines like "click here to subscribe" or "please ignore my last email". They read like instructions, but they're meant for people. Without them, a guard just learns to panic at every button.

Some of the injections are in Roman Urdu too. English-only filters tend to miss them, and it's how a lot of people around me actually text.

Next I'm training a small model on my DGX Spark to catch these before the agent acts. I'll post how it goes, including if it doesn't.

Code's here if you want to poke at it: [link]

#AISecurity #PromptInjection

---

## Post 2: the wallet decoder

**Attach:** `runs/media/post2-wallet/carousel.pdf` (or `terminal.mp4` for a video post)

If you've used a crypto wallet, you've probably seen this: a site asks you to "verify your wallet", a popup appears, and you click sign.

The wallet does show you what you're signing. It just shows it as a wall of hex, and drainers count on nobody reading it.

(If you have a real story about someone who lost something this way, open with that instead. It'll land better than anything I can write.)

So I built a small decoder that turns those requests into plain English before you sign. A few real examples from the slides:

"Claim your free airdrop" actually lets a random address spend all your USDC, forever.

"Verify your NFT" hands someone every item you own in that collection.

"Sign in to the marketplace" is a hidden listing where you give your NFT away for nothing.

"Gasless approval, just sign" claims to be Permit2, but it points at a different contract.

And sending 40 USDC to a friend in your contacts comes back as fine, because a tool that only ever shouts danger is one people learn to ignore.

All the decoding is plain code: approvals, transfers, permits, Permit2, Seaport orders, sign-in messages. The next step is the one code can't do alone: a small model that asks "is this what you meant to do?"

Everything here was tested on made-up requests and local test chains. No real keys, no real money.

[link]

#Web3Security #Crypto

---

## Post 3: the extension demo

**Attach:** `runs/media/post3-extension/extension-demo.mp4` (upload the video directly)

Most "AI-powered" browser tools work by sending the pages you visit to someone's server. I didn't love that.

So I've started building one that runs on my own machine. In the clip it does three things:

- hides the ads and the "sponsored" boxes
- clicks "Reject all" on the cookie wall so I don't have to
- puts a big red warning on a login page pretending to be a brand on the wrong domain

The news site and "NovaPay" are both made up for the demo, before anyone asks.

The rules I gave it: it only talks to localhost, it never reads what you type into a form, and screenshots stay off unless you turn them on.

Honest status: right now the decisions come from simple rules. The point of the setup is that I can swap in a small model trained on my DGX Spark without changing the extension. That's what I'm working on next.

[link]

#Privacy #AISecurity
