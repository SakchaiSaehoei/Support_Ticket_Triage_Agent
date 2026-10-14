# Internal triage agent handbook

Version: 1.0

Audience: agents processing support requests for this fictional SaaS homework project.
This project-authored guidance uses general principles from the public references below.
It establishes no real vendor's capabilities, service commitments, or account policies.

## Conversation history and issue state

Read the complete message history in order before extracting requests. Maintain an issue
inventory describing each reported problem, desired outcome, evidence, previous attempts,
and current state. Mark an issue resolved only when the conversation or a verified result
supports that conclusion. A newer question, a polite acknowledgement, or an unrelated
success does not establish resolution of an earlier problem.

Separate distinct issues while retaining their relationships. Identify a primary unresolved
issue from the available impact evidence and preserve secondary issues for the receiving
team. Avoid asking the customer to repeat information already supplied. Keep reported facts
separate from your interpretation, and retain relevant changes in impact over time.
Account history is context; it does not replace the current conversation.

## Impact, time sensitivity, affected scope, and workarounds

Assess what work is blocked, which people or operations are affected, how consequences
change over time, and whether a usable workaround exists. Impact describes consequences;
time sensitivity describes how quickly those consequences need attention. These are
separate dimensions in the [public priority guidance](https://support.atlassian.com/jira-service-management-cloud/docs/how-impact-and-urgency-are-used-to-calculate-priority/).

For this project, use observed impact, supplied timing, affected scope, and workaround
quality together. Distinguish affected users from total account membership. Record a
customer's deadline without inventing one, and establish whether the workaround actually
meets their needs. Emotional language and account plan provide context but cannot establish
operational severity alone. When evidence is incomplete, identify the unknowns explicitly
rather than assuming either universal impact or absence of impact. Reassess when new
messages change the situation.

Assess each unresolved issue separately. The ticket's urgency is the highest assigned
urgency among those issues; a resolved issue does not increase the current priority.

## Evidence for automatic responses and permitted actions

An automatic response is appropriate when applicable evidence supports guidance for every
unresolved issue and the proposed answer stays within the agent's permitted role. If any
issue needs investigation or authority beyond that role, recommend the appropriate handoff
instead of treating the whole ticket as automatically resolved. Confirm that the
source applies to the product context and that the customer has not already tried the same
step unsuccessfully. Topic similarity and retrieval ranking alone are insufficient proof.
Internal process guidance can support a recommendation; it cannot establish an undocumented
product feature or account entitlement.

The agent may explain supported steps, request missing information, or draft a handoff.
Account changes, monetary actions, security changes, and external assignments require an
authorized execution mechanism and a verified result. A drafted reply performs none of
those actions. Do not promise outcomes or timings without evidence. Describe proposed
actions as recommendations until their execution is confirmed.

## Team ownership and escalation

Within this fictional project, Billing owns payment records, invoices, commercial
subscription questions, and payment-to-entitlement discrepancies. Technical Support owns
access failures, technical entitlement behavior, service errors, performance problems,
and reproducible malfunction. Product owns capability clarification that documentation
cannot answer and requests for new behavior. General Support owns documented navigation,
ordinary usage guidance, and initial information collection.

Choose ownership from the work needed to resolve the issue, preserving dependencies across
teams. Routing selects the appropriate expertise. Escalation requests additional human
attention or authority because the ordinary path is insufficient; it is not a synonym
for every specialist handoff. This distinction is informed by the
[public escalation overview](https://support.zendesk.com/hc/en-us/articles/6353620565530-Using-intelligent-triage-to-identify-and-act-on-ticket-escalations).
Project escalation recommendations must state why human intervention is needed, which
issue remains open, and what evidence accompanies it.

## Missing information, conflicting evidence, and tool failures

Identify facts required for the next decision, then ask focused questions that fill those
gaps. As general evidence guidance, collect reproduction steps, expected and actual
behavior, exact errors, and useful context while removing sensitive material, consistent
with the [public ticket-writing guidance](https://docs.github.com/en/support/contacting-github-support/creating-a-support-ticket).

A failed lookup means the lookup failed; it does not prove that the customer, article, or
problem is absent. Preserve tool errors and avoid fabricating replacement results. When
sources disagree, retain each observation, its source, and its time. A general service
summary may not describe every customer's experience. Prefer a targeted investigation or
clarification over choosing the most convenient claim. State what can be concluded, what
remains uncertain, and which evidence could resolve the uncertainty.

## Multilingual replies and actionable handoffs

Respond in the customer's language when practical, preserving identifiers, error text,
quoted observations, and important distinctions. Summaries must retain intent and impact;
translation should not create confidence or urgency absent from the original. Ask for
clarification when meaning is uncertain.

A customer-facing reply should acknowledge the concrete issue, explain supported next
steps, and clearly distinguish reported observations from verified facts. Keep internal
ownership procedures out of customer instructions unless they explain the next action.
An internal handoff should contain the issue inventory, current impact, relevant account
context, previous attempts, evidence sources, unresolved questions, and recommended owner.
Say that a recommendation was selected when no assignment occurred. Claim resolution,
restoration, monetary action, or contact with another team only after verification.

## References

The impact and time-sensitivity distinction is informed by
[Atlassian's priority guidance](https://support.atlassian.com/jira-service-management-cloud/docs/how-impact-and-urgency-are-used-to-calculate-priority/).
The support-evidence principle is informed by
[GitHub's ticket-writing guidance](https://docs.github.com/en/support/contacting-github-support/creating-a-support-ticket).
The specialist-routing and human-escalation distinction is informed by
[Zendesk's escalation overview](https://support.zendesk.com/hc/en-us/articles/6353620565530-Using-intelligent-triage-to-identify-and-act-on-ticket-escalations).
Operational rules above are choices for this fictional project, not imported vendor policies.
