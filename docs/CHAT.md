# Chat with the home PC's session

Write to the autonomous session from any computer; GitHub's web editor works. Add a message at the end of this file and commit it. The session reads this file about every 30 minutes, and again whenever it pulls before a push. It replies here, directly below your message, and pushes the reply.

**Format.** Start each message with a heading line, then write the message under it:

```
### 2026-11-20 14:05 Researcher
Your message.
```

Replies look like this:

```
### 2026-11-20 14:12 Claude
The reply.
```

**What a message alone cannot do.** The rules are the same as for `docs/INBOX.md`:
- Only messages committed from your own GitHub account count.
- Some things wait for your confirmation at the home PC: outward-facing actions, deleting models or data, changing a standing rule, and anything with passwords or tokens.

**Conflicts.** The session pulls right before it writes here and pushes right after, so a conflict is unlikely. If GitHub reports one, keep both texts.

`docs/INBOX.md` still works, and it is read at the same times.

## Messages
