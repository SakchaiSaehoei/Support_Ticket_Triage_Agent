# Internal SaaS support playbooks

Version: 1.0

Audience: agents investigating requests for this fictional SaaS homework project.
These reusable, project-authored playbooks provide investigation structure rather than
product-specific answers. Public principles informing the guidance appear in References.
No real vendor's feature set, commercial terms, or service commitments are established here.

## Account access

### Information to collect

Identify the account or workspace involved, the sign-in method, and whether the customer
can reach the sign-in page or complete authentication. Preserve the reported failure
and when it began without requesting passwords or authentication secrets.

### Checks

Compare the supplied identifier with available account records. Distinguish authentication
failure from a later authorization failure. Review available evidence of account state
without interpreting a failed lookup as a missing account.

### Supported next steps

Use the product's documented recovery or sign-in instructions when available and applicable.
Ask one targeted question when the failed stage is unclear. Retain attempts already made.

### Escalation

Recommend Technical Support for unresolved access behavior. Request human authority when
account ownership or a security-sensitive change must be verified.

### Unknowns

Do not infer account suspension, compromise, or successful recovery without supporting evidence.

## Permissions and workspace access

### Information to collect

Record the resource, workspace, attempted action, user's known role, and the access result.
Determine whether the customer previously had access and whether other users are affected.

### Checks

Compare documented permission requirements with verified role information. Check whether
the correct workspace and resource were selected. Treat unavailable role data as unknown.

### Supported next steps

Explain applicable permission requirements from authoritative documentation. Where supported,
direct the customer to the authorized workspace administrator for an access review.

### Escalation

Recommend Technical Support for unexplained permission behavior and human review for access
changes beyond the agent's authority. Do not grant permissions through a suggested reply.

### Unknowns

Account membership does not establish access to every resource, and missing visibility
does not establish that the resource was deleted.

## Subscriptions and entitlements

### Information to collect

Identify the relevant account, recorded subscription, requested change, expected entitlement,
and the behavior observed. Keep commercial questions distinct from technical access symptoms.

### Checks

Compare verified subscription records with documented entitlement requirements. Confirm that
the relevant account context matches the request. Separate change requests from completed changes.

### Supported next steps

Explain documented subscription requirements and request clarification about any unmatched
records. Suggest only supported, authorized procedures for managing subscription preferences.

### Escalation

Recommend Billing for commercial records or terms and Technical Support for behavior that
conflicts with verified entitlements. Include the dependency when both teams are needed.

### Unknowns

Do not assume activation timing, entitlement availability, cancellation effects, or eligibility
from an unverified payment attempt or a plan name alone.

## Payments and billing clarification

### Information to collect

Collect relevant invoice or transaction references, amounts, dates, the customer's reported
payment state, and the requested outcome. Remove sensitive payment details from shared evidence.

### Checks

Compare available billing records with the reported transactions. Keep authorization,
settlement, adjustment, and refund states separate when those states are actually provided.
Avoid treating similar records as confirmed duplicate payments.

### Supported next steps

Explain verified records in plain language. Ask for the smallest missing reference needed
to investigate. Use documented billing procedures when available and authorized.

### Escalation

Recommend Billing for reconciliation, disputed records, or monetary action requiring review.
Recommend additional human authority when the normal owner cannot resolve the request.

### Unknowns

Do not invent commercial policy, processing times, refund eligibility, or completed monetary actions.

## Service availability and errors

### Information to collect

Record the failing operation, error details, observation time, affected scope, environment,
and whether access fails consistently. Distinguish direct observation from reported broader impact.

### Checks

Compare relevant service information with the customer's observations. Establish which
operations remain usable and whether the failure is isolated or shared using available evidence.

### Supported next steps

Offer documented, reversible troubleshooting that matches the failure and avoids repeating
unsuccessful steps. Preserve conflicting observations for investigation rather than dismissing them.

### Escalation

Recommend Technical Support when the failure remains unexplained, blocks required work,
or needs service-side investigation. Explain why additional human attention is warranted.

### Unknowns

A reported error does not confirm an outage, and a service summary does not prove that
every operation or customer is unaffected.

## Performance and intermittent behavior

### Information to collect

Identify the slow or inconsistent operation, its frequency, observed duration, environment,
and expected behavior. Ask whether the issue is ongoing and which work remains possible.

### Checks

Compare repeated observations when available. Review documented limits and verified service
information. Distinguish a measured delay from an estimate and a recurring pattern from one event.

### Supported next steps

Suggest relevant documented diagnostics or a verified workaround. Ask for a small reproducible
case when it would narrow the investigation without changing important customer data.

### Escalation

Recommend Technical Support for recurring or unexplained degradation, supplying the observed
pattern, remaining workaround, and impact rather than asserting a root cause.

### Unknowns

Do not assign fault to the customer's connection, device, workload, or the service without evidence.

## Configuration and preferences

### Information to collect

Identify the setting the customer wants to change, current behavior, expected behavior,
product context, and available options. Record previous attempts and whether changes persisted.

### Checks

Use applicable product documentation to distinguish supported configuration from observed
malfunction. Check whether requirements, permissions, or version differences are documented.

### Supported next steps

Explain documented configuration steps when the source applies. Ask for the observed options
if instructions and interface behavior disagree. Do not invent a menu path or missing control.

### Escalation

Recommend General Support for documented usage, Technical Support for reproducible configuration
behavior, and Product for undocumented capability clarification or enhancement requests.

### Unknowns

Absence of an observed option does not establish that a feature is unavailable everywhere,
and a preference request does not establish an existing capability.

## Data workflows and transfers

### Information to collect

Identify the operation, input or output format, data scope, expected result, actual result,
and any error. Ask whether the original data remains intact and whether the operation is repeatable.

### Checks

Compare documented format and size requirements with supplied context. Review filters,
resource selection, and permissions where relevant. Do not request sensitive production data
when sanitized evidence would suffice.

### Supported next steps

Offer documented validation or a small safe reproduction using disposable data. Explain
supported workflow requirements without promising compatibility that has not been established.

### Escalation

Recommend Technical Support for failed or inconsistent processing and human review when the
customer reports possible data loss or needs an account operation beyond agent authority.

### Unknowns

Do not assert successful transfer, recoverability, retention guarantees, or irreversible loss
without verified results and applicable documentation.

## Capability questions and unsupported requests

### Information to collect

Clarify the outcome the customer wants, relevant product context, current workaround,
and whether the request describes missing functionality or a malfunction of documented behavior.

### Checks

Search applicable documentation and available product information. An unsuccessful search
means supporting evidence was not found; it does not prove the capability cannot exist.

### Supported next steps

Describe a documented capability or limitation with its applicable conditions. Offer a
supported alternative when it meets the stated need. Request clarification where evidence is incomplete.

### Escalation

Recommend Product for capability clarification or enhancement requests and Technical Support
for behavior contradicting documented support. Preserve any unresolved operational issue alongside the request.

### Unknowns

Do not promise a roadmap, release timing, future support, or a workaround whose suitability
has not been established.

## References

Evidence collection in these project-authored playbooks follows the general principle of
providing actionable context while protecting sensitive details, informed by
[GitHub's ticket-writing guidance](https://docs.github.com/en/support/contacting-github-support/creating-a-support-ticket).
Impact assessment and routing principles are explained in the companion handbook with links to
[Atlassian's priority guidance](https://support.atlassian.com/jira-service-management-cloud/docs/how-impact-and-urgency-are-used-to-calculate-priority/)
and [Zendesk's escalation overview](https://support.zendesk.com/hc/en-us/articles/6353620565530-Using-intelligent-triage-to-identify-and-act-on-ticket-escalations).
The playbooks do not reproduce vendor procedures, feature promises, or commercial contracts.
